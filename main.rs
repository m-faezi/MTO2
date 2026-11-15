mod background;
mod maxtree;
mod plot;
mod refined;
mod segment;
mod statistical_tests;

use background::estimate_background;
use fitsio::{hdu::HduInfo, FitsFile};
use maxtree::{build_max_tree, replace_nan, Connectivity};
use plot::{LogLogPlot, Point};
use refined::build_segmentation_refined;
use segment::write_segmentation_svg;
use statistical_tests::{detect, move_up, Background};
use std::{collections::HashMap, env, error::Error, process, time::Instant};

fn read_fits(path: &str, hdu_index: usize) -> Result<(Vec<f32>, usize, usize), Box<dyn Error>> {
    let mut file = FitsFile::open(path)?;
    let hdu = file.hdu(hdu_index)?;
    let shape = match &hdu.info {
        HduInfo::ImageInfo { shape, .. } => shape.clone(),
        _ => return Err(format!("HDU {hdu_index} is not an image").into()),
    };
    if shape.len() != 2 {
        return Err(format!("expected 2D image, got {} axes", shape.len()).into());
    }
    let (height, width) = (shape[0], shape[1]);
    let data: Vec<f32> = hdu.read_image(&mut file)?;
    Ok((data, width, height))
}

/// Build a normalized 1-D Gaussian kernel. Returns `[1.0]` for sigma <= 0.
fn gaussian_kernel(sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    // scipy.ndimage.gaussian_filter: truncate=4.0, radius = int(truncate * sigma + 0.5)
    let radius = (4.0 * sigma + 0.5) as usize;
    let size = 2 * radius + 1;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    // Accumulate in f64, normalize, then cast to f32.
    let mut kernel64 = vec![0.0f64; size];
    let mut sum = 0.0f64;
    for i in 0..size {
        let x = i as f64 - radius as f64;
        let v = (-x * x * inv_2sigma2).exp();
        kernel64[i] = v;
        sum += v;
    }
    kernel64.iter().map(|&v| (v / sum) as f32).collect()
}

/// Mirror an index into [0, n) using scipy's 'reflect' mode: d c b a | a b c d | d c b a
#[inline]
fn reflect(i: isize, n: isize) -> usize {
    let period = 2 * n;
    let j = i.rem_euclid(period);
    (if j >= n { period - 1 - j } else { j }) as usize
}

/// Separable Gaussian blur matching `scipy.ndimage.gaussian_filter`
/// (mode='reflect', truncate=4.0).
fn gaussian_blur(img: &[f32], width: usize, height: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        return img.to_vec();
    }
    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;

    // Horizontal pass
    let mut tmp = vec![0.0f32; img.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for (k, &w) in kernel.iter().enumerate() {
                let sx = reflect(x as isize + k as isize - radius as isize, width as isize);
                acc += w * img[y * width + sx];
            }
            tmp[y * width + x] = acc;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f32; img.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for (k, &w) in kernel.iter().enumerate() {
                let sy = reflect(y as isize + k as isize - radius as isize, height as isize);
                acc += w * tmp[sy * width + x];
            }
            out[y * width + x] = acc;
        }
    }
    out
}

fn main() {
    const USAGE: &str = "usage: mto2 <image.fits> [--hdu 0] [--conn 4|8] \
[--s-sigma 2.0] [--alpha 1e-6] [--rejection-rate 0.05] \
[--move-factor 0.0] [--area-ratio 0.90]";
    const KNOWN: [&str; 7] = [
        "hdu", "conn", "s-sigma", "alpha", "rejection-rate",
        "move-factor", "area-ratio",
    ];

    // Named flags: `--key value` or `--key=value`. Anything else is positional.
    let mut positional: Vec<String> = Vec::new();
    let mut flags: HashMap<String, String> = HashMap::new();
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        if let Some(rest) = a.strip_prefix("--") {
            let (key, val) = match rest.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => match it.next() {
                    Some(v) => (rest.to_string(), v),
                    None => {
                        eprintln!("missing value for --{rest}\n{USAGE}");
                        process::exit(1);
                    }
                },
            };
            if !KNOWN.contains(&key.as_str()) {
                eprintln!("unknown option --{key}\n{USAGE}");
                process::exit(1);
            }
            flags.insert(key, val);
        } else {
            positional.push(a);
        }
    }
    if positional.len() != 1 {
        eprintln!("{USAGE}");
        process::exit(1);
    }
    let path = &positional[0];

    fn flag<T: std::str::FromStr>(flags: &HashMap<String, String>, key: &str, default: T) -> T {
        match flags.get(key) {
            Some(s) => s.parse().unwrap_or_else(|_| {
                eprintln!("bad value for --{key}: {s}");
                process::exit(1)
            }),
            None => default,
        }
    }

    let hdu_index: usize = flag(&flags, "hdu", 0);
    let conn = match flags.get("conn").map(String::as_str) {
        Some("8") => Connectivity::Eight,
        Some("4") | None => Connectivity::Four,
        Some(other) => {
            eprintln!("bad value for --conn: {other} (expected 4 or 8)");
            process::exit(1);
        }
    };
    let s_sigma: f64 = flag(&flags, "s-sigma", 2.0);
    let alpha: f64 = flag(&flags, "alpha", 1e-6);
    let rejection_rate: f64 = flag(&flags, "rejection-rate", 0.05);
    let move_factor: f64 = flag(&flags, "move-factor", 0.0);
    let area_ratio: f64 = flag(&flags, "area-ratio", 0.90);
    println!(
        "params: hdu={hdu_index} conn={conn:?} s_sigma={s_sigma} alpha={alpha:e} \
         rejection_rate={rejection_rate} move_factor={move_factor} area_ratio={area_ratio}"
    );

    if let Err(e) = run(
        path,
        hdu_index,
        conn,
        s_sigma,
        alpha,
        rejection_rate,
        move_factor,
        area_ratio,
    ) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    path: &str,
    hdu_index: usize,
    conn: Connectivity,
    s_sigma: f64,
    alpha: f64,
    rejection_rate: f64,
    move_factor: f64,
    area_ratio: f64,
) -> Result<(), Box<dyn Error>> {
    let t0 = Instant::now();
    let (mut img, width, height) = read_fits(path, hdu_index)?;
    println!("read {width} x {height} image in {:.2?}", t0.elapsed());

    let replaced = replace_nan(&mut img);
    if replaced > 0 {
        println!("replaced {replaced} non-finite pixels with image minimum");
    }

    // Print raw image stats so we can sanity-check the pipeline.
    {
        let (mn, mx) = img
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                (a.min(v), b.max(v))
            });
        let mean = img.iter().map(|&v| v as f64).sum::<f64>() / img.len() as f64;
        let var = img
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / img.len() as f64;
        println!(
            "raw image stats: min={mn:.3e} max={mx:.3e} mean={mean:.3e} sigma={:.3e}",
            var.sqrt()
        );
    }

    // MTO2 preprocessing step: Gaussian smoothing controlled by s_sigma.
    if s_sigma > 0.0 {
        let t_blur = Instant::now();
        img = gaussian_blur(&img, width, height, s_sigma);
        println!(
            "gaussian blur (s_sigma={s_sigma:.2}) in {:.2?}",
            t_blur.elapsed()
        );
    } else {
        println!("gaussian blur skipped (s_sigma={s_sigma:.2})");
    }

    // Print smoothed image stats.
    {
        let (mn, mx) = img
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                (a.min(v), b.max(v))
            });
        let mean = img.iter().map(|&v| v as f64).sum::<f64>() / img.len() as f64;
        let var = img
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / img.len() as f64;
        println!(
            "smoothed image stats: min={mn:.3e} max={mx:.3e} mean={mean:.3e} sigma={:.3e}",
            var.sqrt()
        );
    }

    // ---- background estimation from the (smoothed) image ----
    let t_bg = Instant::now();
    let bg = estimate_background(&img, width, height, rejection_rate, alpha);
    println!(
        "background estimated in {:.2?}: mean={:.3e} sigma={:.3e} gain={:.3e}",
        t_bg.elapsed(),
        bg.mean,
        bg.var.sqrt(),
        bg.gain
    );
    if bg.var <= 0.0 || bg.gain == 0.0 {
        eprintln!(
            "WARNING: degenerate background (var={:.3e}, gain={:.3e}); \
             significance test may be meaningless",
            bg.var, bg.gain
        );
    }

    let t1 = Instant::now();
    let tree = build_max_tree(&img, width, height, conn);
    println!("built max-tree ({conn:?}) in {:.2?}", t1.elapsed());

    let s = tree.stats();
    let root = tree.root();
    println!(
        "pixels={} nodes={} leaves={} max_depth={}",
        width * height,
        s.nodes,
        s.leaves,
        s.max_depth
    );
    if s.nodes < 100 {
        eprintln!(
            "WARNING: only {} nodes — the image is nearly constant. \
             Did you forget to skip `quantize`? Or is the image flat?",
            s.nodes
        );
    }
    println!(
        "root: pixel {} (value {})",
        tree.root_pixel(),
        tree.level[root]
    );

    let t2 = Instant::now();
    let area = tree.attribute_area();
    let volume = tree.attribute_volume(&area);
    println!("area + volume in {:.2?}", t2.elapsed());

    // -------- diagnostic plot --------
    let level_root = tree.level[root];
    let points: Vec<Point> = (1..tree.num_nodes())
        .filter(|&nd| area[nd] > 0 && volume[nd] > 0.0)
        .map(|nd| Point {
            area: area[nd] as f64,
            volume: volume[nd],
            color: (tree.level[nd] - level_root) as f64,
        })
        .collect();
    let svg_path = format!("volume_vs_area_hdu{hdu_index}_{conn:?}.svg");
    let title = format!(
        "Volume vs Area ({conn:?}, hdu {hdu_index}, {} non-root nodes)",
        points.len()
    );
    LogLogPlot {
        title: &title,
        xlabel: "area (pixels)",
        ylabel: "volume",
        points,
        slope1_refs: vec![("V = A".to_string(), 1.0)],
    }
    .write_to_path(&svg_path)?;
    println!("wrote {svg_path}");

    // -------- significance + segmentation --------
    let t3 = Instant::now();
    let sel = detect(&tree, &area, &volume, &bg);
    let n_sig = sel.significant.iter().filter(|b| **b).count();
    let n_obj = sel.objects.iter().filter(|b| **b).count();
    println!(
        "detect: {} significant nodes, {} objects (alpha={:.0e}) in {:.2?}",
        n_sig,
        n_obj,
        bg.alpha,
        t3.elapsed()
    );

    let (max_area_sig, max_vol_sig) = (0..tree.num_nodes())
        .filter(|&nd| sel.significant[nd])
        .fold((0u32, 0.0f64), |(ma, mv), nd| {
            (ma.max(area[nd]), mv.max(volume[nd]))
        });
    println!(
        "  debug: max_sig_area={} max_sig_vol={:.3e} root_sig={} root_obj={}",
        max_area_sig, max_vol_sig, sel.significant[root], sel.objects[root]
    );

    // -------- isophote refinement (MTO2 move_up) --------
    let t_mv = Instant::now();
    let corrected = move_up(&tree, &area, &sel, &bg, move_factor, area_ratio);
    println!(
        "move_up (move_factor={move_factor}, area_ratio={area_ratio}): {} corrected segments in {:.2?}",
        corrected.iter().filter(|b| **b).count(),
        t_mv.elapsed()
    );

    let t4 = Instant::now();
    let seg = build_segmentation_refined(&tree, &corrected);
    println!(
        "segmentation: {} pixels covered, {} objects, largest object {} px, in {:.2?}",
        seg.segment_of_pixel
            .iter()
            .filter(|&&s| s != u32::MAX)
            .count(),
        seg.segment_seed.len(),
        seg.segment_area.iter().copied().max().unwrap_or(0),
        t4.elapsed()
    );

    let seg_svg = format!(
        "segmentation_hdu{hdu_index}_{conn:?}_s{s_sigma}_mf{move_factor}_ar{area_ratio}.svg"
    );
    let seg_title = format!(
        "Segmentation ({conn:?}, hdu {hdu_index}, mf={move_factor}, ar={area_ratio}, {} objects, alpha={:.0e}, s_sigma={:.2})",
        seg.segment_seed.len(),
        bg.alpha,
        s_sigma,
    );
    {
        use std::io::Write;
        let f = std::fs::File::create(&seg_svg)?;
        let mut w = std::io::BufWriter::new(f);
        write_segmentation_svg(&mut w, &seg, width, height, &seg_title)?;
        w.flush()?;
    }
    println!("wrote {seg_svg}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_sums_to_one() {
        for sigma in [0.5, 1.0, 2.0, 3.5] {
            let k = gaussian_kernel(sigma);
            let s: f32 = k.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "sigma={sigma} sum={s}");
        }
    }

    #[test]
    fn kernel_zero_sigma_is_identity() {
        assert_eq!(gaussian_kernel(0.0), vec![1.0]);
        let img = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(gaussian_blur(&img, 2, 2, 0.0), img);
    }

    #[test]
    fn blur_preserves_constant() {
        let img = vec![7.0f32; 25];
        let out = gaussian_blur(&img, 5, 5, 1.5);
        for v in out {
            assert!((v - 7.0).abs() < 1e-4);
        }
    }

    #[test]
    fn reflect_matches_scipy() {
        // n = 4: indices -3..=7 -> d c b a | a b c d | d c b a
        let got: Vec<usize> = (-4..8).map(|i| reflect(i, 4)).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0]);
    }

    #[test]
    fn kernel_radius_matches_scipy() {
        // sigma=2 -> int(4*2+0.5)=8 -> 17 taps
        assert_eq!(gaussian_kernel(2.0).len(), 17);
    }

    #[test]
    fn blur_preserves_constant_at_edges() {
        // Reflection must keep border pixels at the constant value (zero-padding would not).
        let img = vec![7.0f32; 12 * 9];
        let out = gaussian_blur(&img, 12, 9, 2.0);
        assert!((out[0] - 7.0).abs() < 1e-4);
        assert!((out[12 * 9 - 1] - 7.0).abs() < 1e-4);
    }

    #[test]
    fn blur_smooths_impulse() {
        // 5x5 image with a single bright pixel in the middle.
        let mut img = vec![0.0f32; 25];
        img[12] = 100.0;
        let out = gaussian_blur(&img, 5, 5, 1.0);
        // Center must drop, neighbours must rise.
        assert!(out[12] < 100.0);
        assert!(out[7] > 0.0);
        assert!(out[11] > 0.0);
        assert!(out[13] > 0.0);
        assert!(out[17] > 0.0);
        // Total flux must be roughly conserved.
        let sum: f32 = out.iter().sum();
        assert!((sum - 100.0).abs() < 1.0, "sum={sum}");
    }
}
