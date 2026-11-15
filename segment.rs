//! Minimal segmentation: each object's pixels are assigned to the
//! highest object seed in that pixel's ancestor chain.

use crate::maxtree::MaxTree;
use std::io::{self, Write};

pub struct Segmentation {
    pub segment_of_pixel: Vec<u32>,
    pub segment_seed: Vec<u32>,
    pub segment_area: Vec<u32>,
}

pub fn build_segmentation(tree: &MaxTree, obj: &[bool]) -> Segmentation {
    let n_pix = tree.num_pixels();
    let n_nodes = tree.num_nodes();

    // Map: node -> segment id (u32::MAX if none).
    let mut node_segment = vec![u32::MAX; n_nodes];
    let mut segment_seed = Vec::new();
    // Walk nodes in tree order so parents come first.
    for &nd in &tree.order {
        let nd = nd as usize;
        if obj[nd] && nd != tree.root() {
            let id = segment_seed.len() as u32;
            segment_seed.push(nd as u32);
            node_segment[nd] = id;
        }
    }
    // Propagate segment assignment: a pixel belongs to the nearest
    // ancestor that is a selected object (inclusive of self).
    let mut pixel_segment = vec![u32::MAX; n_pix];
    for p in 0..n_pix {
        let mut nd = tree.node_of_pixel[p] as usize;
        loop {
            if node_segment[nd] != u32::MAX {
                pixel_segment[p] = node_segment[nd];
                break;
            }
            let par = tree.parent_node[nd] as usize;
            if par == nd {
                break;
            }
            nd = par;
        }
    }

    let mut segment_area = vec![0u32; segment_seed.len()];
    for &s in &pixel_segment {
        if s != u32::MAX {
            segment_area[s as usize] += 1;
        }
    }

    Segmentation {
        segment_of_pixel: pixel_segment,
        segment_seed,
        segment_area,
    }
}

/// Very simple SVG: one randomly-colored rectangle per pixel, grouped by segment.
pub fn write_segmentation_svg<W: Write>(
    w: &mut W,
    seg: &Segmentation,
    title: &str,
) -> io::Result<()> {
    // We don't have width/height in Segmentation, so infer from the caller's
    // perspective is impossible — instead, draw a 1-D bar. If you have the
    // real image dimensions, replace this with a 2-D render.
    let n = seg.segment_of_pixel.len();
    let mut counts = vec![0u32; seg.segment_seed.len()];
    for &s in &seg.segment_of_pixel {
        if s != u32::MAX {
            counts[s as usize] += 1;
        }
    }
    writeln!(
        w,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="800" height="120"><text x="10" y="20">{}</text>"##,
        title.replace('&', "&amp;").replace('<', "&lt;")
    )?;
    let mut x = 10.0;
    for (i, &c) in counts.iter().enumerate() {
        let width = (c as f64 / n as f64) * 780.0;
        let hue = (i as f64 * 137.5) % 360.0;
        writeln!(
            w,
            r##"<rect x="{x:.2}" y="40" width="{width:.2}" height="40" fill="hsl({hue:.0}, 70%, 50%)"/>"##,
        )?;
        x += width;
    }
    writeln!(w, "</svg>")?;
    Ok(())
}