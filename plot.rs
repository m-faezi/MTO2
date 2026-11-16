//! Minimal SVG log-log scatter plot. No external crates.
//!
//! Points are rasterized into a fixed-resolution heatmap so the SVG size is
//! bounded by the plot area, not by the number of input points. This is what
//! keeps the file in the MB range even for images with millions of nodes.

use std::io::{self, Write};

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub area: f64,
    pub volume: f64,
    pub color: f64,
}

pub struct LogLogPlot<'a> {
    pub title: &'a str,
    pub xlabel: &'a str,
    pub ylabel: &'a str,
    pub points: Vec<Point>,
    pub slope1_refs: Vec<(String, f64)>,
}

const W: f64 = 820.0;
const H: f64 = 620.0;
const PAD_L: f64 = 78.0;
const PAD_R: f64 = 110.0;
const PAD_T: f64 = 56.0;
const PAD_B: f64 = 64.0;

pub fn color_of(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0) as f32;
    let r = (0.267 + 0.732 * smooth(t)).powf(1.0);
    let g = 0.005 + 0.995 * smooth(t.powf(1.2));
    let b = 0.329 + 0.671 * (1.0 - smooth(t));
    (
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn nice_log_ticks(lo: f64, hi: f64) -> Vec<f64> {
    let mut ticks = Vec::new();
    if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
        return ticks;
    }
    let mut decade = lo.log10().floor();
    let top = hi.log10().ceil();
    while decade <= top {
        let base = 10f64.powf(decade);
        for m in 1..=9 {
            let v = m as f64 * base;
            if v >= lo && v <= hi { ticks.push(v); }
        }
        decade += 1.0;
    }
    ticks
}

fn fmt_log_tick(v: f64) -> String {
    if v == 0.0 { return "0".into(); }
    let e = v.abs().log10().floor() as i32;
    let m = v / 10f64.powi(e);
    if (m - m.round()).abs() < 1e-9 {
        format!("{}e{}", m.round() as i64, e)
    } else {
        format!("{:.1}e{}", m, e)
    }
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

impl<'a> LogLogPlot<'a> {
    pub fn write_svg<Wr: Write>(&self, w: &mut Wr) -> io::Result<()> {
        // ---- collect valid points in log space ----
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut cs = Vec::new();
        for p in &self.points {
            if p.area > 0.0 && p.volume > 0.0 && p.area.is_finite() && p.volume.is_finite() {
                xs.push(p.area.log10());
                ys.push(p.volume.log10());
                cs.push(p.color);
            }
        }
        if xs.is_empty() {
            writeln!(
                w,
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><text x="10" y="20">no data</text></svg>"##
            )?;
            return Ok(());
        }
        let xmin = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let xmax = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ymin = ys.iter().copied().fold(f64::INFINITY, f64::min);
        let ymax = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let x0 = xmin - 0.25;
        let x1 = xmax + 0.25;
        let y0 = ymin - 0.25;
        let y1 = ymax + 0.25;
        let xspan = (x1 - x0).max(1e-9);
        let yspan = (y1 - y0).max(1e-9);

        let plot_w = W - PAD_L - PAD_R;
        let plot_h = H - PAD_T - PAD_B;
        let sx = |x: f64| PAD_L + (x - x0) / xspan * plot_w;
        let sy = |y: f64| H - PAD_B - (y - y0) / yspan * plot_h;

        let cmin = cs.iter().copied().fold(f64::INFINITY, f64::min);
        let cmax = cs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let cspan = (cmax - cmin).max(1e-12);

        // ---- bin into a fixed grid (heatmap) ----
        let gw = plot_w.max(1.0) as usize;
        let gh = plot_h.max(1.0) as usize;
        let mut sum_r = vec![0u64; gw * gh];
        let mut sum_g = vec![0u64; gw * gh];
        let mut sum_b = vec![0u64; gw * gh];
        let mut cnt = vec![0u32; gw * gh];
        for i in 0..xs.len() {
            let fx = (xs[i] - x0) / xspan;
            let fy = (ys[i] - y0) / yspan;
            if !(0.0..1.0).contains(&fx) || !(0.0..1.0).contains(&fy) {
                continue;
            }
            let gx = ((fx * gw as f64) as usize).min(gw - 1);
            let gy = ((fy * gh as f64) as usize).min(gh - 1);
            let gy = gh - 1 - gy; // flip so row 0 is top
            let idx = gy * gw + gx;
            let (r, g, b) = color_of((cs[i] - cmin) / cspan);
            sum_r[idx] += r as u64;
            sum_g[idx] += g as u64;
            sum_b[idx] += b as u64;
            cnt[idx] += 1;
        }

        // ---- emit SVG ----
        writeln!(w, r##"<?xml version="1.0" encoding="UTF-8"?>"##)?;
        writeln!(
            w,
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif" shape-rendering="crispEdges">"##
        )?;
        writeln!(w, r##"<rect width="{W}" height="{H}" fill="white"/>"##)?;
        writeln!(
            w,
            r##"<rect x="{PAD_L}" y="{PAD_T}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1"/>"##
        )?;

        let xticks = nice_log_ticks(10f64.powf(x0), 10f64.powf(x1));
        let yticks = nice_log_ticks(10f64.powf(y0), 10f64.powf(y1));
        for t in &xticks {
            let x = sx(t.log10());
            if x < PAD_L - 0.5 || x > W - PAD_R + 0.5 { continue; }
            writeln!(
                w,
                r##"<line x1="{x:.2}" y1="{PAD_T}" x2="{x:.2}" y2="{}" stroke="#eee" stroke-width="1"/>"##,
                H - PAD_B
            )?;
            writeln!(
                w,
                r##"<line x1="{x:.2}" y1="{}" x2="{x:.2}" y2="{}" stroke="#222" stroke-width="1"/>"##,
                H - PAD_B,
                H - PAD_B + 5.0
            )?;
            writeln!(
                w,
                r##"<text x="{x:.2}" y="{}" font-size="11" text-anchor="middle">{}</text>"##,
                H - PAD_B + 18.0,
                xml_escape(&fmt_log_tick(*t))
            )?;
        }
        for t in &yticks {
            let y = sy(t.log10());
            if y < PAD_T - 0.5 || y > H - PAD_B + 0.5 { continue; }
            writeln!(
                w,
                r##"<line x1="{PAD_L}" y1="{y:.2}" x2="{}" y2="{y:.2}" stroke="#eee" stroke-width="1"/>"##,
                W - PAD_R
            )?;
            writeln!(
                w,
                r##"<line x1="{}" y1="{y:.2}" x2="{PAD_L}" y2="{y:.2}" stroke="#222" stroke-width="1"/>"##,
                PAD_L - 5.0
            )?;
            writeln!(
                w,
                r##"<text x="{}" y="{:.2}" font-size="11" text-anchor="end" dominant-baseline="middle">{}</text>"##,
                PAD_L - 8.0,
                y,
                xml_escape(&fmt_log_tick(*t))
            )?;
        }

        for (label, c) in &self.slope1_refs {
            let xa = x0.max(y0 - c.log10());
            let xb = x1.min(y1 - c.log10());
            if xa < xb {
                let ya = xa + c.log10();
                let yb = xb + c.log10();
                writeln!(
                    w,
                    r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#c0392b" stroke-dasharray="6 4" stroke-width="1.2" stroke-opacity="0.9"/>"##,
                    sx(xa), sy(ya), sx(xb), sy(yb)
                )?;
                writeln!(
                    w,
                    r##"<text x="{:.2}" y="{:.2}" font-size="11" fill="#c0392b">{}</text>"##,
                    sx(xb) + 4.0,
                    sy(yb) - 4.0,
                    xml_escape(label)
                )?;
            }
        }

        // ---- heatmap cells ----
        for gy in 0..gh {
            for gx in 0..gw {
                let idx = gy * gw + gx;
                if cnt[idx] == 0 { continue; }
                let n = cnt[idx] as u64;
                let (r, g, b) = (
                    (sum_r[idx] / n) as u8,
                    (sum_g[idx] / n) as u8,
                    (sum_b[idx] / n) as u8,
                );
                let px = PAD_L + gx as f64;
                let py = PAD_T + gy as f64;
                writeln!(
                    w,
                    r##"<rect x="{px:.1}" y="{py:.1}" width="1" height="1" fill="rgb({r},{g},{b})"/>"##
                )?;
            }
        }

        writeln!(
            w,
            r##"<text x="{:.1}" y="{}" font-size="13" text-anchor="middle">{}</text>"##,
            PAD_L + plot_w / 2.0,
            H - 24.0,
            xml_escape(self.xlabel)
        )?;
        writeln!(
            w,
            r##"<text x="22" y="{:.1}" font-size="13" text-anchor="middle" transform="rotate(-90 22 {:.1})">{}</text>"##,
            PAD_T + plot_h / 2.0,
            PAD_T + plot_h / 2.0,
            xml_escape(self.ylabel)
        )?;
        writeln!(
            w,
            r##"<text x="{:.1}" y="28" font-size="15" text-anchor="middle">{}</text>"##,
            PAD_L + plot_w / 2.0,
            xml_escape(self.title)
        )?;

        // ---- colour bar ----
        let cb_x = W - PAD_R + 26.0;
        let cb_w = 16.0;
        let cb_y0 = PAD_T + 10.0;
        let cb_y1 = H - PAD_B - 10.0;
        let n_steps = 64;
        for s in 0..n_steps {
            let t = s as f64 / (n_steps - 1) as f64;
            let (r, g, b) = color_of(t);
            let y = cb_y1 - t * (cb_y1 - cb_y0);
            let h = (cb_y1 - cb_y0) / n_steps as f64 + 0.5;
            writeln!(
                w,
                r##"<rect x="{cb_x:.2}" y="{:.2}" width="{cb_w:.2}" height="{h:.2}" fill="rgb({r},{g},{b})"/>"##,
                y
            )?;
        }
        writeln!(
            w,
            r##"<rect x="{cb_x:.2}" y="{cb_y0:.2}" width="{cb_w:.2}" height="{:.2}" fill="none" stroke="#222"/>"##,
            cb_y1 - cb_y0
        )?;
        writeln!(
            w,
            r##"<text x="{:.2}" y="{:.2}" font-size="11" text-anchor="middle">max</text>"##,
            cb_x + cb_w / 2.0, cb_y0 - 6.0
        )?;
        writeln!(
            w,
            r##"<text x="{:.2}" y="{:.2}" font-size="11" text-anchor="middle">min</text>"##,
            cb_x + cb_w / 2.0, cb_y1 + 14.0
        )?;
        writeln!(
            w,
            r##"<text x="{:.2}" y="{:.2}" font-size="11" text-anchor="middle" transform="rotate(90 {:.2} {:.2})">node level</text>"##,
            cb_x + cb_w + 22.0,
            (cb_y0 + cb_y1) / 2.0,
            cb_x + cb_w + 22.0,
            (cb_y0 + cb_y1) / 2.0
        )?;

        writeln!(w, "</svg>")?;
        Ok(())
    }

    pub fn write_to_path(&self, path: &str) -> io::Result<()> {
        let f = std::fs::File::create(path)?;
        let mut w = io::BufWriter::new(f);
        self.write_svg(&mut w)?;
        w.flush()
    }
}