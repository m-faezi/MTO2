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
///
/// Two safeguards keep the output in the MB range for any input size:
///   * the rendered image is capped at `MAX_W` pixels wide (aspect preserved)
///     by block-majority downsampling;
///   * horizontal runs of identical id are collapsed into a single `<rect>`.
pub fn write_segmentation_svg<W: Write>(
    w: &mut W,
    seg: &Segmentation,
    width: usize,
    height: usize,
    title: &str,
) -> io::Result<()> {
    const MAX_W: usize = 1000;

    // Output grid: downsample if the image is wider than MAX_W.
    let scale = ((width + MAX_W - 1) / MAX_W).max(1);
    let out_w = ((width + scale - 1) / scale).max(1);
    let out_h = ((height + scale - 1) / scale).max(1);

    let pad = 20;
    let svg_w = out_w + 2 * pad;
    let svg_h = out_h + 2 * pad + 30;

    let esc = |s: &str| -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
    };

    writeln!(
        w,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" shape-rendering="crispEdges">"##
    )?;
    writeln!(w, r##"<rect width="{svg_w}" height="{svg_h}" fill="#111"/>"##)?;
    writeln!(
        w,
        r##"<text x="{pad}" y="18" font-family="sans-serif" font-size="13" fill="#eee">{}</text>"##,
        esc(title)
    )?;

    // Precompute per-segment colour as a hex string once.
    let mut colours: Vec<String> = Vec::with_capacity(seg.segment_seed.len());
    for i in 0..seg.segment_seed.len() {
        let h = (i as f64 * 137.508) % 360.0;
        let (r, g, b) = hsl_to_rgb(h, 0.75, 0.55);
        colours.push(format!("#{:02x}{:02x}{:02x}", r, g, b));
    }
    const BG: &str = "#222";

    // Build the downsampled id grid: each output cell takes the most common
    // id in its `scale x scale` source block. O(pixels) with a small Vec
    // per block, no per-pixel allocation.
    let mut grid = vec![u32::MAX; out_w * out_h];
    if scale == 1 {
        grid.copy_from_slice(&seg.segment_of_pixel);
    } else {
        let mut keys: Vec<u32> = Vec::with_capacity(16);
        let mut vals: Vec<u32> = Vec::with_capacity(16);
        for oy in 0..out_h {
            let y0 = oy * scale;
            let y1 = (y0 + scale).min(height);
            for ox in 0..out_w {
                let x0 = ox * scale;
                let x1 = (x0 + scale).min(width);
                keys.clear();
                vals.clear();
                for y in y0..y1 {
                    let row = &seg.segment_of_pixel[y * width..(y + 1) * width];
                    for &s in &row[x0..x1] {
                        if let Some(k) = keys.iter().position(|&k| k == s) {
                            vals[k] += 1;
                        } else {
                            keys.push(s);
                            vals.push(1);
                        }
                    }
                }
                // mode; ties go to the lowest id for determinism
                let mut best_i = 0;
                for i in 1..keys.len() {
                    if vals[i] > vals[best_i]
                        || (vals[i] == vals[best_i] && keys[i] < keys[best_i])
                    {
                        best_i = i;
                    }
                }
                grid[oy * out_w + ox] =
                    if keys.is_empty() { u32::MAX } else { keys[best_i] };
            }
        }
    }

    let y_off = pad + 24;
    let mut buf = String::with_capacity(1 << 16);
    for oy in 0..out_h {
        let row = &grid[oy * out_w..(oy + 1) * out_w];
        let mut x = 0;
        while x < out_w {
            let s = row[x];
            let mut x_end = x + 1;
            while x_end < out_w && row[x_end] == s {
                x_end += 1;
            }
            let run = x_end - x;
            let fill = if s == u32::MAX { BG } else { colours[s as usize].as_str() };
            buf.push_str(&format!(
                r##"<rect x="{:.1}" y="{:.1}" width="{}" height="1" fill="{fill}"/>"##,
                (pad + x) as f64,
                (y_off + oy) as f64,
                run,
            ));
            buf.push('\n');
            if buf.len() >= 1 << 15 {
                w.write_all(buf.as_bytes())?;
                buf.clear();
            }
            x = x_end;
        }
    }
    w.write_all(buf.as_bytes())?;
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