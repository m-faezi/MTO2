//! Segmentation from a contracted max-tree (aligned with MTO2 semantics).
//!
//! A pixel is assigned to the nearest object ancestor of its node on
//! the **contracted** tree (nearest significant ancestor chain), not the
//! raw max-tree parent chain.

use crate::maxtree::MaxTree;
use crate::statistical_tests::Selection;
use std::io::{self, Write};

pub struct Segmentation {
    pub segment_of_pixel: Vec<u32>,
    pub segment_seed: Vec<u32>,
    pub segment_area: Vec<u32>,
}

pub fn build_segmentation(tree: &MaxTree, sel: &Selection) -> Segmentation {
    let n_pix = tree.num_pixels();
    let n_nodes = tree.num_nodes();

    // Assign a segment id to each object node.
    let mut node_segment = vec![u32::MAX; n_nodes];
    let mut segment_seed: Vec<u32> = Vec::new();
    for &nd in &tree.order {
        let nd = nd as usize;
        if sel.objects[nd] && nd != tree.root() {
            let id = segment_seed.len() as u32;
            segment_seed.push(nd as u32);
            node_segment[nd] = id;
        }
    }

    // For each pixel: jump to nearest significant ancestor (contracted),
    // then walk up the contracted tree until we hit an object node.
    let mut pixel_segment = vec![u32::MAX; n_pix];
    for p in 0..n_pix {
        let orig = tree.node_of_pixel[p] as usize;
        let mut nd = sel.nearest_sig_ancestor[orig] as usize;
        loop {
            if node_segment[nd] != u32::MAX {
                pixel_segment[p] = node_segment[nd];
                break;
            }
            match sel.contracted_parent[nd] {
                Some(cp) => {
                    if cp as usize == nd {
                        break;
                    }
                    nd = cp as usize;
                }
                None => break,
            }
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

/// Render the segmentation as an actual 2-D image.
pub fn write_segmentation_svg<W: Write>(
    w: &mut W,
    seg: &Segmentation,
    width: usize,
    height: usize,
    title: &str,
) -> io::Result<()> {
    // Upscale factor so small images are visible.
    let scale = (600 / width.max(1)).max(1).min(8);
    let (wpx, hpx) = (width * scale, height * scale);
    let pad = 20;
    let svg_w = wpx + 2 * pad;
    let svg_h = hpx + 2 * pad + 30;

    let esc = |s: &str| -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };

    writeln!(
        w,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">"##
    )?;
    writeln!(
        w,
        r##"<rect width="{svg_w}" height="{svg_h}" fill="#111"/>"##
    )?;
    writeln!(
        w,
        r##"<text x="{pad}" y="18" font-family="sans-serif" font-size="13" fill="#eee">{}</text>"##,
        esc(title)
    )?;

    // Precompute per-segment colour.
    let mut colours: Vec<(u8, u8, u8)> = Vec::with_capacity(seg.segment_seed.len());
    for i in 0..seg.segment_seed.len() {
        let h = (i as f64 * 137.508) % 360.0;
        let (r, g, b) = hsl_to_rgb(h, 0.75, 0.55);
        colours.push((r, g, b));
    }

    let y0 = pad + 24;
    for y in 0..height {
        for x in 0..width {
            let s = seg.segment_of_pixel[y * width + x];
            let fill = if s == u32::MAX {
                "#222".to_string()
            } else {
                let (r, g, b) = colours[s as usize];
                format!("#{:02x}{:02x}{:02x}", r, g, b)
            };
            writeln!(
                w,
                r##"<rect x="{:.1}" y="{:.1}" width="{scale}" height="{scale}" fill="{fill}"/>"##,
                (pad + x * scale) as f64,
                (y0 + y * scale) as f64,
            )?;
        }
    }
    writeln!(w, "</svg>")?;
    Ok(())
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r1, g1, b1) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    (
        (((r1 + m) * 255.0) as i32).clamp(0, 255) as u8,
        (((g1 + m) * 255.0) as i32).clamp(0, 255) as u8,
        (((b1 + m) * 255.0) as i32).clamp(0, 255) as u8,
    )
}