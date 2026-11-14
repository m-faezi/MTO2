//! Segment tree from significant nodes + SVG map.

use crate::maxtree::MaxTree;
use std::io::{self, Write};

pub struct Segmentation {
    pub width: usize,
    pub height: usize,
    pub segment_of_pixel: Vec<u32>,
    pub segment_seed: Vec<u32>,
    pub segment_area: Vec<u32>,
}

pub fn build_segmentation(tree: &MaxTree, is_object: &[bool]) -> Segmentation {
    let n = tree.num_nodes();
    let root = tree.root();
    assert_eq!(is_object.len(), n, "is_object size mismatch");

    let mut obj_ancestor: Vec<u32> = vec![u32::MAX; n];
    obj_ancestor[root] = if is_object[root] { root as u32 } else { u32::MAX };
    for &nd in &tree.order {
        let nd = nd as usize;
        if nd == root { continue; }
        obj_ancestor[nd] = if is_object[nd] {
            nd as u32
        } else {
            obj_ancestor[tree.parent_node[nd] as usize]
        };
    }

    let mut compact: Vec<u32> = vec![u32::MAX; n];
    let mut segment_seed: Vec<u32> = Vec::new();
    for nd in 0..n {
        if is_object[nd] {
            compact[nd] = segment_seed.len() as u32;
            segment_seed.push(tree.seed_pixel[nd]);
        }
    }

    let mut segment_of_pixel = vec![u32::MAX; tree.num_pixels()];
    for p in 0..tree.num_pixels() {
        let nd = tree.node_of_pixel[p] as usize;
        let a = obj_ancestor[nd];
        if a != u32::MAX {
            segment_of_pixel[p] = compact[a as usize];
        }
    }

    let mut segment_area = vec![0u32; segment_seed.len()];
    for &s in &segment_of_pixel {
        if s != u32::MAX { segment_area[s as usize] += 1; }
    }

    Segmentation {
        width: tree.width,
        height: tree.height,
        segment_of_pixel,
        segment_seed,
        segment_area,
    }
}

fn segment_color(id: u32, sat: f64, val: f64) -> (u8, u8, u8) {
    let h = (id as f64 * 137.50776405) % 360.0;
    hsv_to_rgb(h, sat, val)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r1, g1, b1) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    (
        ((r1 + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((g1 + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((b1 + m) * 255.0).clamp(0.0, 255.0) as u8,
    )
}

pub fn write_segmentation_svg<Wr: Write>(
    w: &mut Wr,
    seg: &Segmentation,
    title: &str,
) -> io::Result<()> {
    let w_px = 760.0;
    let h_px = 760.0 * seg.height as f64 / seg.width.max(1) as f64;
    let pad = 20.0;

    writeln!(w, r##"<?xml version="1.0" encoding="UTF-8"?>"##)?;
    writeln!(
        w,
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"##,
        w_px + 2.0 * pad,
        h_px + 2.0 * pad + 40.0,
        w_px + 2.0 * pad,
        h_px + 2.0 * pad + 40.0
    )?;
    writeln!(w, r##"<rect width="100%" height="100%" fill="white"/>"##)?;
    writeln!(
        w,
        r##"<text x="{}" y="24" font-family="sans-serif" font-size="14" text-anchor="middle">{}</text>"##,
        pad + w_px / 2.0,
        xml_escape(title)
    )?;
    writeln!(
        w,
        r##"<rect x="{pad}" y="{}" width="{w_px}" height="{h_px}" fill="#1a1a1a"/>"##,
        pad + 30.0
    )?;

    let sx = w_px / seg.width.max(1) as f64;
    let sy = h_px / seg.height.max(1) as f64;
    let origin_y = pad + 30.0;

    let nseg = seg.segment_seed.len();
    let mut runs: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); nseg];
    for y in 0..seg.height as u32 {
        let mut prev = u32::MAX;
        let mut x0 = 0u32;
        for x in 0..seg.width as u32 {
            let id = seg.segment_of_pixel[y as usize * seg.width + x as usize];
            if id != prev {
                if prev != u32::MAX && x0 < x {
                    runs[prev as usize].push((y, x0, x));
                }
                prev = id;
                x0 = x;
            }
        }
        if prev != u32::MAX && x0 < seg.width as u32 {
            runs[prev as usize].push((y, x0, seg.width as u32));
        }
    }

    for (sid, seg_runs) in runs.iter().enumerate() {
        if seg_runs.is_empty() { continue; }
        let (r, g, b) = segment_color(sid as u32, 0.85, 0.95);
        let mut d = String::new();
        for (y, x0, x1) in seg_runs {
            let px0 = pad + *x0 as f64 * sx;
            let py = origin_y + *y as f64 * sy;
            let pw = (*x1 - *x0) as f64 * sx;
            let ph = sy;
            d.push_str(&format!("M{px0:.2},{py:.2}h{pw:.2}v{ph:.2}h-{pw:.2}z"));
        }
        writeln!(
            w,
            r##"<path d="{d}" fill="rgb({r},{g},{b})" fill-opacity="0.9" stroke="none"/>"##
        )?;
    }

    writeln!(w, "</svg>")?;
    Ok(())
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}