//! Extra outputs for the Rust MTO2: `segmentation_map.fits`, `binary_mask.fits`
//! and `parameters.csv`.
//!
//! * `segmentation_map.fits`: int32 image, 0 = background, 1..K = segment id
//!   (same layout / header as the input image, like the Python `segmentation_map.fits`).
//!   Ids follow `tree.order` (a parent always has a smaller id than its children),
//!   as in the Python version where id = (n_vertices - 1 - vertex index).
//! * `binary_mask.fits`: uint8 image, 0 = background, 1 = any segment.
//!   Same header as `segmentation_map.fits`.
//! * `parameters.csv`: `ID,Area,Flux,R_eff,Min,Max,X,Y,RA,Dec,P_ID`, one row per segment.
//!     P_ID   = ID of the parent segment: the nearest segment above this one in
//!              the max-tree. 0 when the segment has no parent segment.
//!     Area   = max-tree node area of the segment (pixels, includes nested segments),
//!              as `area[n_map_segments]` in the Python version.
//!     Flux   = sum of the UNSMOOTHED image over the same pixels
//!              (`hg.accumulate_sequential(tree_of_segments, image, sum)` in Python).
//!     R_eff  = half-light radius, as Python `uts.half_light_radius`: the distance
//!              from the UNWEIGHTED geometric centroid of the segment's pixels at
//!              which the cumulative UNSMOOTHED flux (pixels sorted by distance)
//!              first reaches half of the total. The pixel set is the segment's
//!              OWN pixels (nested child segments excluded), i.e. the pixels whose
//!              innermost segment is this one (`seg.segment_of_pixel == id`).
//!     Min/Max = min/max of the UNSMOOTHED image over the segment's subtree pixels.
//!     X, Y   = the node centroid already computed by `compute_attributes`
//!              (`tree.cx[node]`, `tree.cy[node]`). No extra work.
//!     RA,Dec = world coordinates of (X, Y) from the FITS WCS header.
//!              NaN when the required WCS keywords are absent.
//!   Everything is O(pixels + nodes + header); R_eff is O(own pixels) expected
//!   (weighted-median selection on squared radii, no per-pixel sqrt/division),
//!   parallel over segments; it falls back to an exact sort for tiny segments and
//!   segments with negative flux.

use crate::maxtree::MaxTree;
use crate::segment::Segmentation;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

const BLOCK: usize = 2880;
const CARD: usize = 80;

// ---------------------------------------------------------------------------
// Minimal linear WCS (TAN-like), read from FITS header cards.
// ---------------------------------------------------------------------------

/// Linear WCS: pixel (1-based) → intermediate world (deg) → sky (deg).
/// Supports CD_i_j and (CDELT + PC) forms; assumes TAN (or any projection where
/// the linear part dominates near the reference point) — good enough for source
/// catalogues. Returns `None` if any required keyword is missing.
struct Wcs {
    crpix1: f64,
    crpix2: f64,
    crval1: f64, // deg
    crval2: f64, // deg
    cd11: f64,
    cd12: f64,
    cd21: f64,
    cd22: f64,
}

impl Wcs {
    fn from_cards(cards: &[[u8; CARD]]) -> Option<Self> {
        let f = |key: &str| -> Option<f64> {
            cards
                .iter()
                .find(|c| c[..8].starts_with(format!("{key:<8}").as_bytes()))
                .and_then(|c| std::str::from_utf8(&c[10..]).ok())
                .and_then(|s| s.split('/').next()?.trim().parse::<f64>().ok())
        };
        let crpix1 = f("CRPIX1")?;
        let crpix2 = f("CRPIX2")?;
        let crval1 = f("CRVAL1")?;
        let crval2 = f("CRVAL2")?;
        // CD matrix preferred; fall back to CDELT + PC.
        let cd11;
        let cd12;
        let cd21;
        let cd22;
        if let (Some(a), Some(b), Some(c), Some(d)) =
            (f("CD1_1"), f("CD1_2"), f("CD2_1"), f("CD2_2"))
        {
            cd11 = a; cd12 = b; cd21 = c; cd22 = d;
        } else {
            let cdelt1 = f("CDELT1")?;
            let cdelt2 = f("CDELT2")?;
            let pc11 = f("PC1_1").unwrap_or(1.0);
            let pc12 = f("PC1_2").unwrap_or(0.0);
            let pc21 = f("PC2_1").unwrap_or(0.0);
            let pc22 = f("PC2_2").unwrap_or(1.0);
            cd11 = cdelt1 * pc11; cd12 = cdelt1 * pc12;
            cd21 = cdelt2 * pc21; cd22 = cdelt2 * pc22;
        }
        Some(Wcs {
            crpix1, crpix2, crval1, crval2,
            cd11, cd12, cd21, cd22,
        })
    }

    /// (x, y) are 0-based pixel coordinates; FITS is 1-based.
    /// Returns (ra_deg, dec_deg). For small offsets near the reference point
    /// the linear intermediate-world coords ARE the tangent-plane coords;
    /// we do the standard TAN inverse so the result is exact for TAN.
    #[inline]
    fn sky(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x + 1.0 - self.crpix1;
        let dy = y + 1.0 - self.crpix2;
        let xi = self.cd11 * dx + self.cd12 * dy; // deg
        let eta = self.cd21 * dx + self.cd22 * dy; // deg
        let xi = xi.to_radians();
        let eta = eta.to_radians();
        let ra0 = self.crval1.to_radians();
        let dec0 = self.crval2.to_radians();
        let (sin_d0, cos_d0) = dec0.sin_cos();
        let denom = cos_d0 - eta * sin_d0;
        let ra = ra0 + xi.atan2(denom);
        let dec = (sin_d0 + eta * cos_d0).atan2((xi * xi + denom * denom).sqrt());
        (ra.to_degrees().rem_euclid(360.0), dec.to_degrees())
    }
}

// ---------------------------------------------------------------------------
// R_eff (half-light radius)
// ---------------------------------------------------------------------------
//
// Same quantity as Python `half_light_radius`, computed with:
//   * one fused pass over the image that groups each segment's own pixels
//     (counting sort, row-major) and accumulates centroid sums + total flux with
//     no per-pixel division (x, y come from the loop counters);
//   * squared distances throughout (sqrt only on the final answer);
//   * O(n) expected weighted-median selection (3-way quickselect on squared
//     radius) instead of an O(n log n) sort, with an exact sort fallback for
//     tiny segments, negative fluxes (cumsum not monotone) and degenerate pivots;
//   * segments processed in parallel (std scoped threads, largest first,
//     dynamic task queue).
//
// The scratch buffer is stored as two parallel `Vec<f64>` (squared radius and
// flux) — a structure-of-arrays layout so the partition swaps only touch the
// radius slice for comparisons while flux lookups stay sequential.

/// Segments at or below this many pixels are simply sorted.
const SORT_CUTOFF: usize = 24;
/// Below this many segmented pixels, threads cost more than they save.
const PAR_MIN_PIXELS: usize = 1 << 15;
/// A task block claims this many single-segment tasks before batching.
const SINGLE_TASK_LIMIT: usize = 64;
/// Granularity of the batched tail of small segments.
const BLOCK_TASK_SIZE: usize = 256;

/// Per-segment own-pixel lists plus the sums R_eff needs.
struct SegPixels {
    start: Vec<usize>,
    members: Vec<u32>,
    sx: Vec<u64>,
    sy: Vec<u64>,
    flux: Vec<f64>,
    has_neg: Vec<bool>,
}

fn group_pixels(seg: &Segmentation, raw: &[f32], w: usize) -> SegPixels {
    let n_seg = seg.segment_seed.len();
    let mut start = vec![0usize; n_seg + 1];
    for i in 0..n_seg {
        start[i + 1] = start[i] + seg.segment_area[i] as usize;
    }
    let mut fill = start[..n_seg].to_vec();
    let mut members = vec![0u32; start[n_seg]];
    let mut sx = vec![0u64; n_seg];
    let mut sy = vec![0u64; n_seg];
    let mut flux = vec![0.0f64; n_seg];
    let mut has_neg = vec![false; n_seg];

    for (y, (srow, rrow)) in seg
        .segment_of_pixel
        .chunks_exact(w)
        .zip(raw.chunks_exact(w))
        .enumerate()
    {
        let base = y * w;
        for (x, (&s, &v)) in srow.iter().zip(rrow).enumerate() {
            if s == u32::MAX {
                continue;
            }
            let s = s as usize;
            members[fill[s]] = (base + x) as u32;
            fill[s] += 1;
            sx[s] += x as u64;
            sy[s] += y as u64;
            flux[s] += v as f64;
            has_neg[s] |= !(v >= 0.0); // NaN counts as "unsafe" too
        }
    }
    SegPixels { start, members, sx, sy, flux, has_neg }
}

// ---------------------------------------------------------------------------
// SoA scratch buffer for R_eff
// ---------------------------------------------------------------------------

/// Structure-of-arrays scratch: squared radius and flux in parallel vectors.
/// The partition only compares/swaps `d2`; flux is looked up by index.
#[derive(Default)]
struct Scratch {
    d2: Vec<f64>,
    fl: Vec<f64>,
}

impl Scratch {
    #[inline]
    fn clear(&mut self) {
        self.d2.clear();
        self.fl.clear();
    }

    #[inline]
    fn push(&mut self, d2: f64, fl: f64) {
        self.d2.push(d2);
        self.fl.push(fl);
    }

    #[inline]
    fn len(&self) -> usize {
        self.d2.len()
    }

    /// Swap element `i` and `j` in both arrays.
    #[inline]
    fn swap(&mut self, i: usize, j: usize) {
        self.d2.swap(i, j);
        self.fl.swap(i, j);
    }
}

/// Sort by squared radius, return the squared radius of the first pixel whose
/// cumulative flux reaches `need` (largest radius if never reached).
fn first_crossing_sorted(s: &mut Scratch, need: f64) -> f64 {
    // Sort indices by d2; simplest correct approach is to sort a zipped temp.
    // We sort `d2` and carry `fl` along using a small index permutation.
    let n = s.len();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_unstable_by(|&a, &b| s.d2[a as usize].total_cmp(&s.d2[b as usize]));
    let mut cum = 0.0f64;
    let mut last_d2 = 0.0f64;
    for &i in &idx {
        let i = i as usize;
        cum += s.fl[i];
        last_d2 = s.d2[i];
        if cum >= need {
            return last_d2;
        }
    }
    last_d2
}

/// Squared half-light radius over `s` with total flux `total`.
/// `max_d2` is the largest d² in `s`. With `allow_select` and non-negative
/// fluxes this is an O(n) weighted-median selection; the result is identical to
/// the sort-and-scan answer (tied radii are grouped, so tie order is irrelevant).
fn half_radius_d2(
    s: &mut Scratch,
    total: f64,
    max_d2: f64,
    allow_select: bool,
) -> f64 {
    let mut need = total * 0.5;
    if !allow_select || s.len() <= SORT_CUTOFF {
        return first_crossing_sorted(s, need);
    }
    let (mut lo, mut hi) = (0usize, s.len());
    // introselect-style budget: fall back to sorting if partitions are unlucky
    let mut budget = 2 * (usize::BITS - s.len().leading_zeros()) as usize + 4;
    while lo < hi {
        if budget == 0 || hi - lo <= SORT_CUTOFF {
            return first_crossing_sorted_slice(s, lo, hi, need);
        }
        budget -= 1;

        // median-of-three pivot
        let (a, b, c) = (s.d2[lo], s.d2[(lo + hi) / 2], s.d2[hi - 1]);
        let pivot = if (a <= b) == (b <= c) { b } else if (b <= a) == (a <= c) { a } else { c };

        // 3-way partition: [lo,lt) < pivot, [lt,gt) == pivot, [gt,hi) > pivot
        let (mut lt, mut i, mut gt) = (lo, lo, hi);
        let (mut s_lt, mut s_eq) = (0.0f64, 0.0f64);
        while i < gt {
            let d = s.d2[i];
            if d < pivot {
                // move the `< pivot` element to `lt`; flux follows via swap
                let fl = s.fl[i];
                s.swap(lt, i);
                s_lt += fl;
                lt += 1;
                i += 1;
            } else if d > pivot {
                gt -= 1;
                s.swap(i, gt);
            } else {
                s_eq += s.fl[i];
                i += 1;
            }
        }

        if s_lt >= need {
            hi = lt; // crossing happens among the smaller radii
        } else if s_lt + s_eq >= need {
            return pivot;
        } else {
            need -= s_lt + s_eq;
            lo = gt; // crossing happens among the larger radii
        }
    }
    max_d2 // only reachable through float drift: "never reached" => last radius
}

/// Same as `first_crossing_sorted` but restricted to `[lo, hi)`, used by the
/// quickselect fallback so we sort only the surviving sub-range.
fn first_crossing_sorted_slice(s: &mut Scratch, lo: usize, hi: usize, need: f64) -> f64 {
    let mut idx: Vec<u32> = (lo..hi).map(|i| i as u32).collect();
    idx.sort_unstable_by(|&a, &b| s.d2[a as usize].total_cmp(&s.d2[b as usize]));
    let mut cum = 0.0f64;
    let mut last_d2 = 0.0f64;
    for &i in &idx {
        let i = i as usize;
        cum += s.fl[i];
        last_d2 = s.d2[i];
        if cum >= need {
            return last_d2;
        }
    }
    last_d2
}

/// Half-light radius of segment `i`.
fn segment_half_light_radius(
    g: &SegPixels,
    i: usize,
    raw: &[f32],
    w: usize,
    buf: &mut Scratch,
) -> f64 {
    let members = &g.members[g.start[i]..g.start[i + 1]];
    let total = g.flux[i];
    if members.is_empty() || total == 0.0 {
        return 0.0;
    }
    let n = members.len() as f64;
    let (cx, cy) = (g.sx[i] as f64 / n, g.sy[i] as f64 / n);

    // `members` ascends in row-major order, so the row index is tracked with
    // comparisons instead of a division per pixel.
    buf.clear();
    let mut y = members[0] as usize / w;
    let mut row_start = y * w;
    let mut row_end = row_start + w;
    let mut max_d2 = 0.0f64;
    for &p in members {
        let p = p as usize;
        while p >= row_end {
            row_start = row_end;
            row_end += w;
            y += 1;
        }
        let dx = (p - row_start) as f64 - cx;
        let dy = y as f64 - cy;
        let d2 = dx * dx + dy * dy;
        if d2 > max_d2 {
            max_d2 = d2;
        }
        buf.push(d2, raw[p] as f64);
    }
    half_radius_d2(buf, total, max_d2, !g.has_neg[i]).sqrt()
}

/// Build the task list: largest segments first as single tasks, then blocks.
fn build_tasks(seg: &Segmentation) -> (Vec<u32>, Vec<(usize, usize)>) {
    let n_seg = seg.segment_seed.len();
    let mut order: Vec<u32> = (0..n_seg as u32).collect();
    order.sort_unstable_by_key(|&i| std::cmp::Reverse(seg.segment_area[i as usize]));
    let mut tasks: Vec<(usize, usize)> = Vec::new();
    let mut k = 0usize;
    while k < n_seg {
        let len = if tasks.len() < SINGLE_TASK_LIMIT { 1 } else { BLOCK_TASK_SIZE };
        tasks.push((k, (k + len).min(n_seg)));
        k += len;
    }
    (order, tasks)
}

/// R_eff for every segment (index = segment id). `threads == 1` runs serially.
fn compute_r_eff_with(seg: &Segmentation, raw: &[f32], w: usize, threads: usize) -> Vec<f64> {
    let n_seg = seg.segment_seed.len();
    let g = group_pixels(seg, raw, w);
    let mut out = vec![0.0f64; n_seg];

    let threads = threads.min(n_seg).max(1);
    if threads == 1 {
        let mut buf = Scratch::default();
        for i in 0..n_seg {
            out[i] = segment_half_light_radius(&g, i, raw, w, &mut buf);
        }
        return out;
    }

    let (order, tasks) = build_tasks(seg);

    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, f64)>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut buf = Scratch::default();
                    let mut res = Vec::new();
                    loop {
                        let t = next.fetch_add(1, Ordering::Relaxed);
                        if t >= tasks.len() {
                            break;
                        }
                        let (a, b) = tasks[t];
                        for &i in &order[a..b] {
                            let i = i as usize;
                            res.push((i, segment_half_light_radius(&g, i, raw, w, &mut buf)));
                        }
                    }
                    res
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for part in parts {
        for (i, r) in part {
            out[i] = r;
        }
    }
    out
}

/// R_eff for every segment, using all available cores for large inputs.
fn compute_r_eff(seg: &Segmentation, raw: &[f32], w: usize) -> Vec<f64> {
    let covered: usize = seg.segment_area.iter().map(|&a| a as usize).sum();
    let threads = if covered < PAR_MIN_PIXELS {
        1
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    };
    compute_r_eff_with(seg, raw, w, threads)
}

// ---------------------------------------------------------------------------
// parameters.csv
// ---------------------------------------------------------------------------

/// Write `ID,Area,Flux,R_eff,Min,Max,X,Y,RA,Dec`.
/// `raw` is the image BEFORE smoothing.
/// `in_path`/`hdu_index` are used only to read the WCS keywords once.
pub fn write_parameters_csv(
    tree: &MaxTree,
    area: &[u32],
    seg: &Segmentation,
    raw: &[f32],
    in_path: &str,
    hdu_index: usize,
    path: &str,
) -> io::Result<()> {
    let root = tree.root();
    let n_nodes = tree.num_nodes();
    let w = tree.width;

    // WCS: read header once. None → RA/Dec = NaN.
    let wcs = read_header_cards(in_path, hdu_index)
        .ok()
        .and_then(|cards| Wcs::from_cards(&cards));

    // ---- one bottom-up pass accumulating everything we need per node ----
    let mut flux = vec![0.0f64; n_nodes];
    let mut mn = vec![f64::INFINITY; n_nodes];
    let mut mx = vec![f64::NEG_INFINITY; n_nodes];

    for (p, &nd) in tree.node_of_pixel.iter().enumerate() {
        let nd = nd as usize;
        let v = raw[p] as f64;
        flux[nd] += v;
        if v < mn[nd] { mn[nd] = v; }
        if v > mx[nd] { mx[nd] = v; }
    }

    for &nd in tree.order.iter().rev() {
        let nd = nd as usize;
        if nd == root {
            continue;
        }
        let par = tree.parent_node[nd] as usize;
        flux[par] += flux[nd];
        if mn[nd] < mn[par] { mn[par] = mn[nd]; }
        if mx[nd] > mx[par] { mx[par] = mx[nd]; }
    }

    // ---- parent segment id (0 = no parent segment) ----
    // `anc_seg[nd]` = id (1-based) of the nearest segment node on the path
    // from `nd` to the root, inclusive; 0 if none. `order` lists parents first.
    let mut node_seg_id = vec![0u32; n_nodes];
    for (i, &nd) in seg.segment_seed.iter().enumerate() {
        node_seg_id[nd as usize] = i as u32 + 1;
    }
    let mut anc_seg = vec![0u32; n_nodes];
    for &nd in &tree.order {
        let nd = nd as usize;
        anc_seg[nd] = if node_seg_id[nd] != 0 {
            node_seg_id[nd]
        } else if nd == root {
            0
        } else {
            anc_seg[tree.parent_node[nd] as usize]
        };
    }

    // ---- R_eff for all segments (fused grouping + parallel selection) ----
    let r_eff_all = compute_r_eff(seg, raw, w);

    // ---- write rows ----
    let mut wtr = BufWriter::new(File::create(path)?);
    writeln!(wtr, "ID,Area,Flux,R_eff,Min,Max,X,Y,RA,Dec,P_ID")?;
    for (i, &nd) in seg.segment_seed.iter().enumerate() {
        let nd = nd as usize;
        let r_eff = r_eff_all[i];
        let seg_min = if mn[nd].is_finite() { mn[nd] } else { 0.0 };
        let seg_max = if mx[nd].is_finite() { mx[nd] } else { 0.0 };

        // X, Y: reuse the centroid already computed in `compute_attributes`.
        let x = tree.cx[nd] as f64;
        let y = tree.cy[nd] as f64;

        // RA, Dec: WCS transform of the same (x, y), or NaN when unavailable.
        let (ra, dec) = match &wcs {
            Some(w) => {
                let (r, d) = w.sky(x, y);
                (r, d)
            }
            None => (f64::NAN, f64::NAN),
        };

        // Parent segment: nearest segment strictly above this node (0 = none).
        let p_id = if nd == root { 0 } else { anc_seg[tree.parent_node[nd] as usize] };

        writeln!(
            wtr,
            "{},{},{},{:.6},{:.6e},{:.6e},{:.6},{:.6},{:.8},{:.8},{}",
            i + 1,
            area[nd],
            flux[nd],
            r_eff,
            seg_min,
            seg_max,
            x,
            y,
            ra,
            dec,
            p_id
        )?;
    }
    wtr.flush()
}

// ---------------------------------------------------------------------------
// FITS header helpers
// ---------------------------------------------------------------------------

/// Header cards of HDU `hdu_index` of `path`, verbatim (80 bytes each, END excluded).
fn read_header_cards(path: &str, hdu_index: usize) -> io::Result<Vec<[u8; CARD]>> {
    let mut f = File::open(path)?;
    let mut block = [0u8; BLOCK];
    for hdu in 0..=hdu_index {
        let mut cards: Vec<[u8; CARD]> = Vec::new();
        'header: loop {
            f.read_exact(&mut block)?;
            for c in block.chunks_exact(CARD) {
                if &c[..8] == b"END     " {
                    break 'header;
                }
                let mut card = [b' '; CARD];
                card.copy_from_slice(c);
                cards.push(card);
            }
        }
        if hdu == hdu_index {
            return Ok(cards);
        }
        // skip this HDU's data: |BITPIX|/8 * GCOUNT * (PCOUNT + prod(NAXISn))
        let int = |key: &str| -> Option<i64> {
            cards.iter().find(|c| c[..8].starts_with(format!("{key:<8}").as_bytes()))
                .and_then(|c| std::str::from_utf8(&c[10..]).ok())
                .and_then(|s| s.split('/').next()?.trim().parse::<i64>().ok())
        };
        let naxis = int("NAXIS").unwrap_or(0);
        let mut n: i64 = if naxis > 0 { 1 } else { 0 };
        for k in 1..=naxis {
            n *= int(&format!("NAXIS{k}")).unwrap_or(0);
        }
        let bytes = int("BITPIX").unwrap_or(8).abs() / 8
            * int("GCOUNT").unwrap_or(1)
            * (int("PCOUNT").unwrap_or(0) + n);
        let padded = (bytes as u64 + BLOCK as u64 - 1) / BLOCK as u64 * BLOCK as u64;
        f.seek(SeekFrom::Current(padded as i64))?;
    }
    unreachable!()
}

fn push_card(out: &mut Vec<u8>, key: &str, value: &str) {
    let s = format!("{key:<8}= {value:>20}");
    out.extend_from_slice(format!("{s:<80}").as_bytes());
}

/// Build a header for a 2-D image with `bitpix`, copying WCS/other cards
/// from the input image's HDU, dropping structural keywords.
fn build_image_header(
    in_path: &str,
    hdu_index: usize,
    bitpix: i32,
    width: usize,
    height: usize,
) -> io::Result<Vec<u8>> {
    const SKIP: [&str; 14] = [
        "SIMPLE", "XTENSION", "BITPIX", "NAXIS", "EXTEND", "PCOUNT", "GCOUNT", "GROUPS",
        "BZERO", "BSCALE", "BLANK", "CHECKSUM", "DATASUM", "BUNIT",
    ];
    let mut hdr: Vec<u8> = Vec::new();
    push_card(&mut hdr, "SIMPLE", "T");
    push_card(&mut hdr, "BITPIX", &bitpix.to_string());
    push_card(&mut hdr, "NAXIS", "2");
    push_card(&mut hdr, "NAXIS1", &width.to_string());
    push_card(&mut hdr, "NAXIS2", &height.to_string());
    for c in read_header_cards(in_path, hdu_index)? {
        let key = String::from_utf8_lossy(&c[..8]).trim_end().to_string();
        if key.is_empty() || SKIP.contains(&key.as_str()) || key.starts_with("NAXIS") {
            continue;
        }
        hdr.extend_from_slice(&c);
    }
    hdr.extend_from_slice(format!("{:<80}", "END").as_bytes());
    hdr.resize((hdr.len() + BLOCK - 1) / BLOCK * BLOCK, b' ');
    Ok(hdr)
}

/// Pad a data payload to a FITS 2880-byte boundary and write header + data.
fn write_fits_image<F>(
    in_path: &str,
    hdu_index: usize,
    bitpix: i32,
    width: usize,
    height: usize,
    out_path: &str,
    data_bytes: usize,
    write_data: F,
) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let hdr = build_image_header(in_path, hdu_index, bitpix, width, height)?;
    let mut w = BufWriter::with_capacity(1 << 20, File::create(out_path)?);
    w.write_all(&hdr)?;
    write_data(&mut w)?;
    w.write_all(&vec![0u8; (BLOCK - data_bytes % BLOCK) % BLOCK])?;
    w.flush()
}

// ---------------------------------------------------------------------------
// segmentation_map.fits
// ---------------------------------------------------------------------------

/// Write the segment-id map (int32, big endian) with the WCS/header of the input image.
pub fn write_segmentation_fits(
    in_path: &str,
    hdu_index: usize,
    seg: &Segmentation,
    width: usize,
    height: usize,
    out_path: &str,
) -> io::Result<()> {
    let data_bytes = seg.segment_of_pixel.len() * 4;
    write_fits_image(
        in_path, hdu_index, 32, width, height, out_path, data_bytes,
        |w| {
            let mut buf: Vec<u8> = Vec::with_capacity(4 << 16);
            for chunk in seg.segment_of_pixel.chunks(1 << 16) {
                buf.clear();
                for &s in chunk {
                    let id: i32 = if s == u32::MAX { 0 } else { s as i32 + 1 };
                    buf.extend_from_slice(&id.to_be_bytes());
                }
                w.write_all(&buf)?;
            }
            Ok(())
        },
    )
}

// ---------------------------------------------------------------------------
// binary_mask.fits
// ---------------------------------------------------------------------------

/// Write the binary mask (uint8, FITS BITPIX=8): 0 = background, 1 = any segment.
pub fn write_binary_mask_fits(
    in_path: &str,
    hdu_index: usize,
    seg: &Segmentation,
    width: usize,
    height: usize,
    out_path: &str,
) -> io::Result<()> {
    let data_bytes = seg.segment_of_pixel.len();
    write_fits_image(
        in_path, hdu_index, 8, width, height, out_path, data_bytes,
        |w| {
            let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);
            for chunk in seg.segment_of_pixel.chunks(1 << 16) {
                buf.clear();
                for &s in chunk {
                    buf.push(if s == u32::MAX { 0 } else { 1 });
                }
                w.write_all(&buf)?;
            }
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxtree::{build_max_tree, Connectivity};
    use crate::refined::build_segmentation_refined;

    fn xorshift(s: &mut u64) -> f64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        (*s >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Brute-force check of ID / Area / Flux / R_eff / Min / Max / X / Y
    /// against a flood fill of {pixel >= level}.
    #[test]
    fn all_columns_match_flood_fill() {
        let (w, h) = (23usize, 17usize);
        let mut s = 88172645463325252u64;
        let mut rnd = || xorshift(&mut s);
        let raw: Vec<f32> = (0..w * h).map(|_| (rnd() * 20.0) as f32).collect();
        let smooth: Vec<f32> = raw.iter().map(|&v| v * 0.5 + 1.0).collect();
        let mut tree = build_max_tree(&smooth, w, h, Connectivity::Four);
        crate::maxtree::compute_attributes(&mut tree, &smooth);
        let area = tree.area.clone();
        let corrected: Vec<bool> = (0..tree.num_nodes()).map(|nd| nd != 0 && rnd() < 0.3).collect();
        let seg = build_segmentation_refined(&tree, &corrected);
        assert!(seg.segment_seed.len() > 3);

        let dir = std::env::temp_dir().join("mto2_out_test");
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("parameters.csv");
        // Use a non-existent input to exercise the NaN-RA/Dec path.
        write_parameters_csv(&tree, &area, &seg, &raw, "no_such.fits", 0, csv.to_str().unwrap())
            .unwrap();
        let text = std::fs::read_to_string(&csv).unwrap();
        let mut rows = text.lines();
        assert_eq!(rows.next().unwrap(), "ID,Area,Flux,R_eff,Min,Max,X,Y,RA,Dec,P_ID");

        for (i, line) in rows.enumerate() {
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f[0].parse::<usize>().unwrap(), i + 1);
            let nd = seg.segment_seed[i] as usize;
            let seed = tree.seed_pixel[nd] as usize;
            let level = tree.level[nd];
            let mut seen = vec![false; w * h];
            let mut stack = vec![seed];
            seen[seed] = true;
            let (mut a, mut fl) = (0u32, 0.0f64);
            let (mut sx, mut sy) = (0.0f64, 0.0f64);
            let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
            while let Some(p) = stack.pop() {
                a += 1;
                let v = raw[p] as f64;
                fl += v;
                let (x, y) = ((p % w) as f64, (p / w) as f64);
                sx += x; sy += y;
                if v < mn { mn = v; }
                if v > mx { mx = v; }
                let (xi, yi) = ((p % w) as isize, (p / w) as isize);
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (xi + dx, yi + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize { continue; }
                    let q = ny as usize * w + nx as usize;
                    if !seen[q] && smooth[q] >= level { seen[q] = true; stack.push(q); }
                }
            }
            assert_eq!(f[1].parse::<u32>().unwrap(), a, "area of id {}", i + 1);
            assert!((f[2].parse::<f64>().unwrap() - fl).abs() < 1e-6 * fl.abs().max(1.0), "flux of id {}", i + 1);
            let cx = sx / a as f64;
            let cy = sy / a as f64;
            // R_eff: naive re-implementation of Python `half_light_radius` on the
            // segment's OWN pixels (innermost segment == this one).
            let own: Vec<usize> = (0..w * h)
                .filter(|&p| seg.segment_of_pixel[p] == i as u32)
                .collect();
            assert!(!own.is_empty());
            let tot: f64 = own.iter().map(|&p| raw[p] as f64).sum();
            let y0 = own.iter().map(|&p| (p / w) as f64).sum::<f64>() / own.len() as f64;
            let x0 = own.iter().map(|&p| (p % w) as f64).sum::<f64>() / own.len() as f64;
            let mut rf: Vec<(f64, f64)> = own
                .iter()
                .map(|&p| {
                    let dy = (p / w) as f64 - y0;
                    let dx = (p % w) as f64 - x0;
                    ((dx * dx + dy * dy).sqrt(), raw[p] as f64)
                })
                .collect();
            rf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let cum: Vec<f64> = rf
                .iter()
                .scan(0.0, |s, &(_, v)| { *s += v; Some(*s) })
                .collect();
            let idx = cum.partition_point(|&c| c < tot / 2.0); // numpy searchsorted(left)
            let r_eff = if idx < rf.len() { rf[idx].0 } else { rf[rf.len() - 1].0 };
            assert!((f[3].parse::<f64>().unwrap() - r_eff).abs() < 1e-4, "R_eff of id {}", i + 1);
            assert!((f[4].parse::<f64>().unwrap() - mn).abs() < 1e-4, "Min of id {}", i + 1);
            assert!((f[5].parse::<f64>().unwrap() - mx).abs() < 1e-4, "Max of id {}", i + 1);
            // X, Y from compute_attributes == centroid from the flood fill
            assert!((f[6].parse::<f64>().unwrap() - cx).abs() < 1e-4, "X of id {}", i + 1);
            assert!((f[7].parse::<f64>().unwrap() - cy).abs() < 1e-4, "Y of id {}", i + 1);
            assert!(f[8].parse::<f64>().unwrap().is_nan(), "RA should be NaN without WCS");
            assert!(f[9].parse::<f64>().unwrap().is_nan(), "Dec should be NaN without WCS");
            // P_ID: nearest segment strictly above in the max-tree (0 = none).
            let mut anc = tree.parent_node[nd] as usize;
            let mut expect = 0usize;
            if nd != tree.root() {
                loop {
                    if let Some(j) = seg.segment_seed.iter().position(|&s| s as usize == anc) {
                        expect = j + 1;
                        break;
                    }
                    if anc == tree.root() { break; }
                    anc = tree.parent_node[anc] as usize;
                }
            }
            let p_id = f[10].parse::<usize>().unwrap();
            if nd == tree.root() {
                assert_eq!(p_id, 0, "root P_ID must be 0");
            }
            assert_eq!(
                p_id, expect,
                "P_ID of id {} is {}, expected {} (0 = no strict-ancestor segment)",
                i + 1, p_id, expect
            );
        }
    }

    /// Straightforward reference: own pixels, unweighted centroid, full sort,
    /// first index with cumulative flux >= total/2.
    fn naive_r_eff(seg: &Segmentation, raw: &[f32], w: usize, id: usize) -> f64 {
        let own: Vec<usize> = (0..raw.len())
            .filter(|&p| seg.segment_of_pixel[p] == id as u32)
            .collect();
        if own.is_empty() {
            return 0.0;
        }
        let tot: f64 = own.iter().map(|&p| raw[p] as f64).sum();
        if tot == 0.0 {
            return 0.0;
        }
        let n = own.len() as f64;
        let y0 = own.iter().map(|&p| (p / w) as f64).sum::<f64>() / n;
        let x0 = own.iter().map(|&p| (p % w) as f64).sum::<f64>() / n;
        let mut rf: Vec<(f64, f64)> = own
            .iter()
            .map(|&p| {
                let (dy, dx) = ((p / w) as f64 - y0, (p % w) as f64 - x0);
                ((dx * dx + dy * dy).sqrt(), raw[p] as f64)
            })
            .collect();
        rf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut cum = 0.0;
        for &(r, f) in &rf {
            cum += f;
            if cum >= tot / 2.0 {
                return r;
            }
        }
        rf[rf.len() - 1].0
    }

    #[test]
    fn selection_matches_sort_scan() {
        let mut s = 0x9E3779B97F4A7C15u64;
        for n in [1usize, 2, 5, 24, 25, 100, 1000, 5000] {
            for trial in 0..20 {
                // integer-grid distances => many tied radii; ~20% zero fluxes
                let mut d2s: Vec<f64> = Vec::with_capacity(n);
                let mut fls: Vec<f64> = Vec::with_capacity(n);
                for _ in 0..n {
                    let dx = (xorshift(&mut s) * 15.0).floor();
                    let dy = (xorshift(&mut s) * 15.0).floor();
                    let f = if xorshift(&mut s) < 0.2 { 0.0 } else { xorshift(&mut s) * 10.0 };
                    d2s.push(dx * dx + dy * dy);
                    fls.push(f);
                }
                let total: f64 = fls.iter().sum();
                if total == 0.0 {
                    continue;
                }
                let max_d2 = d2s.iter().copied().fold(0.0, f64::max);
                let mut a = Scratch { d2: d2s.clone(), fl: fls.clone() };
                let mut b = Scratch { d2: d2s, fl: fls };
                let sel = half_radius_d2(&mut a, total, max_d2, true);
                let srt = half_radius_d2(&mut b, total, max_d2, false);
                assert_eq!(sel, srt, "n={n} trial={trial}");
            }
        }
    }

    fn big_random_segmentation(negatives: bool) -> (Segmentation, Vec<f32>, usize) {
        let (w, h) = (256usize, 200usize);
        let mut s = 424242u64;
        let raw: Vec<f32> = (0..w * h)
            .map(|_| {
                let v = (xorshift(&mut s) * 50.0) as f32;
                if negatives && xorshift(&mut s) < 0.05 { -v * 0.2 } else { v }
            })
            .collect();
        let smooth: Vec<f32> = raw.iter().map(|&v| v * 0.5 + 1.0).collect();
        let mut tree = build_max_tree(&smooth, w, h, Connectivity::Four);
        crate::maxtree::compute_attributes(&mut tree, &smooth);
        let corrected: Vec<bool> =
            (0..tree.num_nodes()).map(|nd| nd != 0 && xorshift(&mut s) < 0.02).collect();
        (build_segmentation_refined(&tree, &corrected), raw, w)
    }

    #[test]
    fn r_eff_parallel_serial_and_naive_agree() {
        for negatives in [false, true] {
            let (seg, raw, w) = big_random_segmentation(negatives);
            assert!(seg.segment_seed.len() > 50);
            assert!(seg.segment_area.iter().any(|&a| a as usize > SORT_CUTOFF * 4));
            let serial = compute_r_eff_with(&seg, &raw, w, 1);
            let par = compute_r_eff_with(&seg, &raw, w, 4);
            assert_eq!(serial, par, "serial vs parallel (negatives={negatives})");
            for id in 0..seg.segment_seed.len() {
                let want = naive_r_eff(&seg, &raw, w, id);
                assert!(
                    (serial[id] - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "segment {id} (negatives={negatives}): got {} want {want}",
                    serial[id]
                );
            }
        }
    }

    #[test]
    fn wcs_reference_pixel_maps_to_crval() {
        // Build a tiny fake header with a trivial WCS.
        let mk = |key: &str, val: &str| -> [u8; CARD] {
            let s = format!("{key:<8}= {val:>20}");
            let mut c = [b' '; CARD];
            c.copy_from_slice(format!("{s:<80}").as_bytes());
            c
        };
        let cards = vec![
            mk("CRPIX1", "10.0"),
            mk("CRPIX2", "20.0"),
            mk("CRVAL1", "150.0"),
            mk("CRVAL2", "2.0"),
            mk("CD1_1", "-0.001"),
            mk("CD1_2", "0.0"),
            mk("CD2_1", "0.0"),
            mk("CD2_2", "0.001"),
        ];
        let w = Wcs::from_cards(&cards).expect("valid wcs");
        // 0-based pixel (9, 19) is the reference pixel.
        let (ra, dec) = w.sky(9.0, 19.0);
        assert!((ra - 150.0).abs() < 1e-9);
        assert!((dec - 2.0).abs() < 1e-9);
        // One pixel to the right → RA decreases by 0.001 deg (negative CD1_1).
        let (ra2, _) = w.sky(10.0, 19.0);
        assert!((ra2 - 149.999).abs() < 1e-6);
    }

    #[test]
    fn binary_mask_is_binary_of_segmentation() {
        // Small synthetic image; we don't need real FITS files to check the
        // data bytes. Build the mask buffer the same way the writer does.
        let pix: Vec<u32> = vec![u32::MAX, 0, u32::MAX, 1, 2, u32::MAX, 0, 1, 2];
        let mut buf: Vec<u8> = Vec::new();
        for &s in &pix {
            buf.push(if s == u32::MAX { 0 } else { 1 });
        }
        let expected: Vec<u8> = pix.iter().map(|&s| if s == u32::MAX { 0 } else { 1 }).collect();
        assert_eq!(buf, expected);
        // Same rule as the segment-id map: 0 ↔ u32::MAX.
        for &s in &pix {
            let id: i32 = if s == u32::MAX { 0 } else { s as i32 + 1 };
            let bin: u8 = if s == u32::MAX { 0 } else { 1 };
            assert_eq!(id == 0, bin == 0);
        }
    }
}

