mod maxtree;
mod plot;

use fitsio::{hdu::HduInfo, FitsFile};
use maxtree::{build_max_tree, replace_nan, Connectivity};
use plot::{LogLogPlot, Point};
use std::{env, error::Error, process, time::Instant};

fn read_fits(path: &str, hdu_index: usize) -> Result<(Vec<f32>, usize, usize), Box<dyn Error>> {
    let mut file = FitsFile::open(path)?;
    let hdu = file.hdu(hdu_index)?;

    let shape = match &hdu.info {
        HduInfo::ImageInfo { shape, .. } => shape.clone(),
        _ => return Err(format!("HDU {hdu_index} is not an image").into()),
    };
    if shape.len() != 2 {
        return Err(format!("expected a 2D image, got {} axes: {:?}", shape.len(), shape).into());
    }
    let (height, width) = (shape[0], shape[1]);
    let data: Vec<f32> = hdu.read_image(&mut file)?;
    Ok((data, width, height))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <image.fits> [hdu_index=0] [4|8]", args[0]);
        process::exit(1);
    }
    let path = &args[1];
    let hdu_index: usize = args.get(2).map_or(0, |s| s.parse().expect("bad hdu index"));
    let conn = match args.get(3).map(String::as_str) {
        Some("8") => Connectivity::Eight,
        _ => Connectivity::Four,
    };

    if let Err(e) = run(path, hdu_index, conn) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn run(path: &str, hdu_index: usize, conn: Connectivity) -> Result<(), Box<dyn Error>> {
    let t0 = Instant::now();
    let (mut img, width, height) = read_fits(path, hdu_index)?;
    println!("read {width} x {height} image in {:.2?}", t0.elapsed());

    let replaced = replace_nan(&mut img);
    if replaced > 0 {
        println!("replaced {replaced} non-finite pixels with the image minimum");
    }

    let t1 = Instant::now();
    let tree = build_max_tree(&img, width, height, conn);
    println!("built max-tree ({conn:?}) in {:.2?}", t1.elapsed());

    let s = tree.stats();
    let root = tree.root();
    println!("pixels:    {}", width * height);
    println!("nodes:     {}", s.nodes);
    println!("leaves:    {}", s.leaves);
    println!("max depth: {}", s.max_depth);
    println!("root:      pixel {} (value {})", tree.root_pixel(), tree.level[root]);

    let t2 = Instant::now();
    let area = tree.attribute_area();
    let volume = tree.attribute_volume(&area);
    println!("computed area and volume in {:.2?}", t2.elapsed());

    // Largest non-root node by area and volume.
    let mut max_area = (0u32, root);
    let mut max_volume = (0.0f64, root);
    for nd in 1..tree.num_nodes() {
        if area[nd] > max_area.0 { max_area = (area[nd], nd); }
        if volume[nd] > max_volume.0 { max_volume = (volume[nd], nd); }
    }
    let (ax, ay) = tree.node_xy(max_area.1);
    let (vx, vy) = tree.node_xy(max_volume.1);
    println!("root area:   {} (should equal pixels)", area[root]);
    println!("root volume: {:.6e}", volume[root]);
    println!("largest non-root area:   {} at (x={ax}, y={ay})", max_area.0);
    println!("largest non-root volume: {:.6e} at (x={vx}, y={vy})", max_volume.0);

    // Telescoping sanity check.
    let min = tree.level[root] as f64;
    let expected: f64 = img.iter().map(|&v| v as f64 - min).sum();
    println!(
        "check: sum(pixel - min) = {:.6e} (relative diff {:.2e})",
        expected,
        ((volume[root] - expected) / expected.abs().max(f64::MIN_POSITIVE)).abs()
    );

    // Optional CSV dump (kept from earlier).
    if std::env::var_os("MTO2_DUMP").is_some() {
        use std::io::Write;
        let csv_path = format!("volume_vs_area_hdu{hdu_index}_{conn:?}.csv");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&csv_path)?);
        writeln!(f, "node,area,volume,level")?;
        for nd in 1..tree.num_nodes() {
            writeln!(f, "{nd},{},{:.9e},{:.9e}",
                     area[nd], volume[nd], tree.level[nd])?;
        }
        f.flush()?;
        println!("wrote {csv_path}");
    }

    // --- Volume vs Area log-log plot ---
    // Reference line: V = A (slope-1 baseline; source nodes rise above it).
    let level_root = tree.level[root];
    let points: Vec<Point> = (1..tree.num_nodes())
        .filter(|&nd| area[nd] > 0 && volume[nd] > 0.0)
        .map(|nd| Point {
            area: area[nd] as f64,
            volume: volume[nd],
            // Color by level.  Shift by root level so background is the low end.
            color: (tree.level[nd] - level_root) as f64,
        })
        .collect();

    let svg_path = format!("volume_vs_area_hdu{hdu_index}_{conn:?}.svg");
    let title = format!(
        "Volume vs Area ({conn:?}, hdu {hdu_index}, {} non-root nodes)",
        points.len()
    );
    let plot = LogLogPlot {
        title: &title,
        xlabel: "area (pixels)",
        ylabel: "volume",
        points,
        slope1_refs: vec![("V = A".to_string(), 1.0)],
    };
    plot.write_to_path(&svg_path)?;
    println!("wrote {svg_path}");

    Ok(())
}