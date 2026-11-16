//! Extra outputs for the Rust MTO2: `segmentation_map.fits` and `parameters.csv`.
//!
//! * `segmentation_map.fits`: int32 image, 0 = background, 1..K = segment id
//!   (same layout / header as the input image, like the Python `segmentation_map.fits`).
//!   Ids follow `tree.order` (a parent always has a smaller id than its children),
//!   as in the Python version where id = (n_vertices - 1 - vertex index).
//! * `parameters.csv`: `ID,Area,Flux`, one row per segment.
//!     Area = max-tree node area of the segment (pixels, includes nested segments),
//!            as `area[n_map_segments]` in the Python version.
//!     Flux = sum of the UNSMOOTHED image over the same pixels
//!            (`hg.accumulate_sequential(tree_of_segments, image, sum)` in Python).
//!   Both are O(pixels + nodes); no per-segment pixel lists are built.

use crate::maxtree::MaxTree;
use crate::segment::Segmentation;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};

const BLOCK: usize = 2880;
const CARD: usize = 80;

// ---------------------------------------------------------------------------
// parameters.csv
// ---------------------------------------------------------------------------

/// Write `ID,Area,Flux`. `raw` is the image BEFORE smoothing.
pub fn write_parameters_csv(
    tree: &MaxTree,
    area: &[u32],
    seg: &Segmentation,
    raw: &[f32],
    path: &str,
) -> io::Result<()> {
    let root = tree.root();

    // Flux of every max-tree node = sum of raw pixels in its whole subtree.
    let mut flux = vec![0.0f64; tree.num_nodes()];
    for (p, &nd) in tree.node_of_pixel.iter().enumerate() {
        flux[nd as usize] += raw[p] as f64;
    }
    for &nd in tree.order.iter().rev() {
        let nd = nd as usize;
        if nd != root {
            flux[tree.parent_node[nd] as usize] += flux[nd];
        }
    }

    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "ID,Area,Flux")?;
    for (i, &nd) in seg.segment_seed.iter().enumerate() {
        let nd = nd as usize;
        writeln!(w, "{},{},{}", i + 1, area[nd], flux[nd])?;
    }
    w.flush()
}

// ---------------------------------------------------------------------------
// segmentation_map.fits
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

/// Write the segment-id map (int32, big endian) with the WCS/header of the input image.
pub fn write_segmentation_fits(
    in_path: &str,
    hdu_index: usize,
    seg: &Segmentation,
    width: usize,
    height: usize,
    out_path: &str,
) -> io::Result<()> {
    // structural / image-specific keywords are rewritten or dropped
    const SKIP: [&str; 14] = [
        "SIMPLE", "XTENSION", "BITPIX", "NAXIS", "EXTEND", "PCOUNT", "GCOUNT", "GROUPS",
        "BZERO", "BSCALE", "BLANK", "CHECKSUM", "DATASUM", "BUNIT",
    ];

    let mut hdr: Vec<u8> = Vec::new();
    push_card(&mut hdr, "SIMPLE", "T");
    push_card(&mut hdr, "BITPIX", "32");
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

    let mut w = BufWriter::with_capacity(1 << 20, File::create(out_path)?);
    w.write_all(&hdr)?;

    // data: 0 = background, segment index + 1 otherwise; chunked, no full-size byte buffer
    let mut buf: Vec<u8> = Vec::with_capacity(4 << 16);
    for chunk in seg.segment_of_pixel.chunks(1 << 16) {
        buf.clear();
        for &s in chunk {
            let id: i32 = if s == u32::MAX { 0 } else { s as i32 + 1 };
            buf.extend_from_slice(&id.to_be_bytes());
        }
        w.write_all(&buf)?;
    }
    let data_bytes = seg.segment_of_pixel.len() * 4;
    w.write_all(&vec![0u8; (BLOCK - data_bytes % BLOCK) % BLOCK])?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxtree::{build_max_tree, Connectivity};
    use crate::refined::build_segmentation_refined;

    /// Brute-force check of ID / Area / Flux against flood fill of {pixel >= level}.
    #[test]
    fn area_and_flux_match_flood_fill() {
        let (w, h) = (23usize, 17usize);
        let mut s = 88172645463325252u64;
        let mut rnd = || {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let raw: Vec<f32> = (0..w * h).map(|_| (rnd() * 20.0) as f32).collect();
        // "smoothed" image used for the tree differs from raw on purpose
        let smooth: Vec<f32> = raw.iter().map(|&v| v * 0.5 + 1.0).collect();
        let tree = build_max_tree(&smooth, w, h, Connectivity::Four);
        let area = tree.attribute_area();
        let corrected: Vec<bool> = (0..tree.num_nodes()).map(|nd| nd != 0 && rnd() < 0.3).collect();
        let seg = build_segmentation_refined(&tree, &corrected);
        assert!(seg.segment_seed.len() > 3);

        let dir = std::env::temp_dir().join("mto2_out_test");
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("parameters.csv");
        write_parameters_csv(&tree, &area, &seg, &raw, csv.to_str().unwrap()).unwrap();
        let text = std::fs::read_to_string(&csv).unwrap();
        let mut rows = text.lines();
        assert_eq!(rows.next().unwrap(), "ID,Area,Flux");

        for (i, line) in rows.enumerate() {
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f[0].parse::<usize>().unwrap(), i + 1);
            // flood fill of {smooth >= level} from the seed pixel
            let nd = seg.segment_seed[i] as usize;
            let seed = tree.seed_pixel[nd] as usize;
            let level = tree.level[nd];
            let mut seen = vec![false; w * h];
            let mut stack = vec![seed];
            seen[seed] = true;
            let (mut a, mut fl) = (0u32, 0.0f64);
            while let Some(p) = stack.pop() {
                a += 1;
                fl += raw[p] as f64;
                let (x, y) = ((p % w) as isize, (p / w) as isize);
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize { continue; }
                    let q = ny as usize * w + nx as usize;
                    if !seen[q] && smooth[q] >= level { seen[q] = true; stack.push(q); }
                }
            }
            assert_eq!(f[1].parse::<u32>().unwrap(), a, "area of id {}", i + 1);
            assert!((f[2].parse::<f64>().unwrap() - fl).abs() < 1e-6 * fl.abs().max(1.0), "flux of id {}", i + 1);
        }
    }
}