mod maxtree;

use fitsio::{hdu::HduInfo, FitsFile};
use maxtree::{build_max_tree, replace_nan, Connectivity};
use std::{env, error::Error, process, time::Instant};

/// Read a 2D image HDU as (data, width, height).
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

    // fitsio reports shape as [NAXIS2, NAXIS1] = [height, width].
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

    let s = tree.stats(&img);
    println!("pixels:    {}", width * height);
    println!("nodes:     {}", s.nodes);
    println!("leaves:    {}", s.leaves);
    println!("max depth: {}", s.max_depth);
    println!("root:      pixel {} (value {})", tree.root(), img[tree.root()]);

    let t2 = Instant::now();
    let area = tree.attribute_area();
    let volume = tree.attribute_volume(&img, &area);
    println!("computed area and volume in {:.2?}", t2.elapsed());

    // Largest non-root node by area and by volume.
    let root = tree.root();
    let mut max_area = (0u32, root);
    let mut max_volume = (0.0f64, root);
    for p in 0..width * height {
        if p == root || !tree.is_node(&img, p) {
            continue;
        }
        if area[p] > max_area.0 {
            max_area = (area[p], p);
        }
        if volume[p] > max_volume.0 {
            max_volume = (volume[p], p);
        }
    }
    let (ax, ay) = (max_area.1 % width, max_area.1 / width);
    let (vx, vy) = (max_volume.1 % width, max_volume.1 / width);
    println!("root area:   {} (should equal pixels)", area[root]);
    println!("root volume: {:.6e}", volume[root]);
    println!("largest non-root area:   {} at (x={ax}, y={ay})", max_area.0);
    println!("largest non-root volume: {:.6e} at (x={vx}, y={vy})", max_volume.0);

    // Sanity check: V(root) telescopes to sum(pixel - min).
    let min = img[root] as f64;
    let expected: f64 = img.iter().map(|&v| v as f64 - min).sum();
    println!(
        "check: sum(pixel - min) = {:.6e} (relative diff {:.2e})",
        expected,
        ((volume[root] - expected) / expected.abs().max(f64::MIN_POSITIVE)).abs()
    );
    Ok(())
}

