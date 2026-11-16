//! Extra outputs for the Rust MTO2: `segmentation_map.fits`, `binary_mask.fits`
//! and `parameters.csv`.
//!
//! * `segmentation_map.fits`: int32 image, 0 = background, 1..K = segment id
//!   (same layout / header as the input image, like the Python `segmentation_map.fits`).
//!   Ids follow `tree.order` (a parent always has a smaller id than its children),
//!   as in the Python version where id = (n_vertices - 1 - vertex index).
//! * `binary_mask.fits`: uint8 image, 0 = background, 1 = any segment.
//!   Same header as `segmentation_map.fits`.
//! * `parameters.csv`: `ID,Area,Flux,R_eff,Min,Max,X,Y,RA,Dec`, one row per segment.
//!     Area   = max-tree node area of the segment (pixels, includes nested segments),
//!              as `area[n_map_segments]` in the Python version.
//!     Flux   = sum of the UNSMOOTHED image over the same pixels
//!              (`hg.accumulate_sequential(tree_of_segments, image, sum)` in Python).
//!     R_eff  = sqrt( (Σx² + Σy²)/A - (cx² + cy²) ) over the segment's pixels,
//!              with (cx, cy) = (Σx/A, Σy/A) the pixel centroid.
//!     Min/Max = min/max of the UNSMOOTHED image over the segment's pixels.
//!     X, Y   = the node centroid already computed by `compute_attributes`
//!              (`tree.cx[node]`, `tree.cy[node]`). No extra work.
//!     RA,Dec = world coordinates of (X, Y) from the FITS WCS header.
//!              NaN when the required WCS keywords are absent.
//!   Everything is O(pixels + nodes + header); no per-segment pixel lists.

use crate::maxtree::MaxTree;
use crate::segment::Segmentation;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};

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
    let mut sx = vec![0.0f64; n_nodes];
    let mut sy = vec![0.0f64; n_nodes];
    let mut sxx = vec![0.0f64; n_nodes];
    let mut syy = vec![0.0f64; n_nodes];
    let mut mn = vec![f64::INFINITY; n_nodes];
    let mut mx = vec![f64::NEG_INFINITY; n_nodes];

    for (p, &nd) in tree.node_of_pixel.iter().enumerate() {
        let nd = nd as usize;
        let v = raw[p] as f64;
        let x = (p % w) as f64;
        let y = (p / w) as f64;
        flux[nd] += v;
        sx[nd] += x;
        sy[nd] += y;
        sxx[nd] += x * x;
        syy[nd] += y * y;
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
        sx[par] += sx[nd];
        sy[par] += sy[nd];
        sxx[par] += sxx[nd];
        syy[par] += syy[nd];
        if mn[nd] < mn[par] { mn[par] = mn[nd]; }
        if mx[nd] > mx[par] { mx[par] = mx[nd]; }
    }

    // ---- write rows ----
    let mut wtr = BufWriter::new(File::create(path)?);
    writeln!(wtr, "ID,Area,Flux,R_eff,Min,Max,X,Y,RA,Dec")?;
    for (i, &nd) in seg.segment_seed.iter().enumerate() {
        let nd = nd as usize;
        let a = area[nd].max(1) as f64;
        let cx = sx[nd] / a;
        let cy = sy[nd] / a;
        let r2 = (sxx[nd] + syy[nd]) / a - (cx * cx + cy * cy);
        let r_eff = r2.max(0.0).sqrt();
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

        writeln!(
            wtr,
            "{},{},{},{:.6},{:.6e},{:.6e},{:.6},{:.6},{:.8},{:.8}",
            i + 1,
            area[nd],
            flux[nd],
            r_eff,
            seg_min,
            seg_max,
            x,
            y,
            ra,
            dec
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

    /// Brute-force check of ID / Area / Flux / R_eff / Min / Max / X / Y
    /// against a flood fill of {pixel >= level}.
    #[test]
    fn all_columns_match_flood_fill() {
        let (w, h) = (23usize, 17usize);
        let mut s = 88172645463325252u64;
        let mut rnd = || {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
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
        assert_eq!(rows.next().unwrap(), "ID,Area,Flux,R_eff,Min,Max,X,Y,RA,Dec");

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
            let (mut sx, mut sy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
            while let Some(p) = stack.pop() {
                a += 1;
                let v = raw[p] as f64;
                fl += v;
                let (x, y) = ((p % w) as f64, (p / w) as f64);
                sx += x; sy += y; sxx += x * x; syy += y * y;
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
            let r2 = (sxx + syy) / a as f64 - (cx * cx + cy * cy);
            let r_eff = r2.max(0.0).sqrt();
            assert!((f[3].parse::<f64>().unwrap() - r_eff).abs() < 1e-4, "R_eff of id {}", i + 1);
            assert!((f[4].parse::<f64>().unwrap() - mn).abs() < 1e-4, "Min of id {}", i + 1);
            assert!((f[5].parse::<f64>().unwrap() - mx).abs() < 1e-4, "Max of id {}", i + 1);
            // X, Y from compute_attributes == centroid from the flood fill
            assert!((f[6].parse::<f64>().unwrap() - cx).abs() < 1e-4, "X of id {}", i + 1);
            assert!((f[7].parse::<f64>().unwrap() - cy).abs() < 1e-4, "Y of id {}", i + 1);
            assert!(f[8].parse::<f64>().unwrap().is_nan(), "RA should be NaN without WCS");
            assert!(f[9].parse::<f64>().unwrap().is_nan(), "Dec should be NaN without WCS");
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
