mod maxtree;
mod plot;
mod segment;
mod statistical_tests;

use fitsio::{hdu::HduInfo, FitsFile};
use maxtree::{build_max_tree, replace_nan, Connectivity};
use plot::{LogLogPlot, Point};
use segment::{build_segmentation, write_segmentation_svg};
use statistical_tests::{detect, Background};
use std::{env, error::Error, process, time::Instant};

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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <image.fits> [hdu=0] [4|8] [bg_mean=0] [bg_sigma=1] [bg_gain=1] [alpha=1e-6]",
            args[0]
        );
        process::exit(1);
    }
    let path = &args[1];
    let hdu_index: usize = args.get(2).map_or(0, |s| s.parse().expect("bad hdu"));
    let conn = match args.get(3).map(String::as_str) {
        Some("8") => Connectivity::Eight,
        _ => Connectivity::Four,
    };
    let bg_mean: f64 = args.get(4).map_or(0.0, |s| s.parse().expect("bad bg_mean"));
    let bg_sigma: f64 = args.get(5).map_or(1.0, |s| s.parse().expect("bad bg_sigma"));
    let bg_gain: f64 = args.get(6).map_or(1.0, |s| s.parse().expect("bad bg_gain"));
    let alpha: f64 = args.get(7).map_or(1e-6, |s| s.parse().expect("bad alpha"));
    let bg = Background {
        mean: bg_mean,
        var: bg_sigma * bg_sigma,
        gain: bg_gain,
        alpha,
    };

    if let Err(e) = run(path, hdu_index, conn, &bg) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn run(
    path: &str,
    hdu_index: usize,
    conn: Connectivity,
    bg: &Background,
) -> Result<(), Box<dyn Error>> {
    let t0 = Instant::now();
    let (mut img, width, height) = read_fits(path, hdu_index)?;
    println!("read {width} x {height} image in {:.2?}", t0.elapsed());

    let replaced = replace_nan(&mut img);
    if replaced > 0 {
        println!("replaced {replaced} non-finite pixels with image minimum");
    }

    // Print image stats so we can sanity-check the bg args.
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
            "image stats: min={mn:.3e} max={mx:.3e} mean={mean:.3e} sigma={:.3e}",
            var.sqrt()
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
    let sel = detect(&tree, &area, &volume, bg);
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

    let t4 = Instant::now();
    let seg = build_segmentation(&tree, &sel);
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

    let seg_svg = format!("segmentation_hdu{hdu_index}_{conn:?}.svg");
    let seg_title = format!(
        "Segmentation ({conn:?}, hdu {hdu_index}, {} objects, alpha={:.0e})",
        seg.segment_seed.len(),
        bg.alpha
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