//! Morphological background estimation (fallback for the constant estimator).
//!
//! Consumes the *already computed* attributes on `MaxTree` (filled by
//! `maxtree::compute_attributes`) so this module does no tree traversal at all.
//! All it does is feature building → standardize → PCA(2) → FCM(2) → keep
//! large background nodes → derive `mean`/`var`/`gain`.

use crate::maxtree::MaxTree;
use crate::statistical_tests::Background;

// ---------------------------------------------------------------------------
// Standardize (columns -> zero mean / unit variance)
// ---------------------------------------------------------------------------

fn standardize(feats: &mut [f32], n: usize, d: usize) {
    let inv_n = 1.0f32 / n as f32;
    for j in 0..d {
        let mut mean = 0.0f32;
        for i in 0..n { mean += feats[i * d + j]; }
        mean *= inv_n;
        let mut var = 0.0f32;
        for i in 0..n {
            let x = feats[i * d + j] - mean;
            var += x * x;
        }
        var *= inv_n;
        let std = var.sqrt().max(1e-12);
        for i in 0..n {
            feats[i * d + j] = (feats[i * d + j] - mean) / std;
        }
    }
}

// ---------------------------------------------------------------------------
// PCA via deflation + power iteration.
// ---------------------------------------------------------------------------

fn pca(feats: &[f32], n: usize, d: usize, k: usize) -> (Vec<f32>, usize) {
    let k = k.min(d);

    let mut cov = vec![0.0f32; d * d];
    for i in 0..n {
        for a in 0..d {
            let xa = feats[i * d + a];
            for b in a..d {
                cov[a * d + b] += xa * feats[i * d + b];
            }
        }
    }
    let inv_n = 1.0f32 / n as f32;
    for a in 0..d {
        for b in a..d {
            let v = cov[a * d + b] * inv_n;
            cov[a * d + b] = v;
            cov[b * d + a] = v;
        }
    }

    let mut state: u64 = 42;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as f32 / (1u64 << 31) as f32) - 1.0
    };

    let mut cov_work = cov.clone();
    let mut components: Vec<Vec<f32>> = Vec::with_capacity(k);

    for _ in 0..k {
        let mut v: Vec<f32> = (0..d).map(|_| next()).collect();
        let mut norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() { *x /= norm; }
        }
        for _ in 0..100 {
            let mut w = vec![0.0f32; d];
            for a in 0..d {
                for b in 0..d { w[a] += cov_work[a * d + b] * v[b]; }
            }
            for c in &components {
                let proj: f32 = w.iter().zip(c).map(|(x, y)| x * y).sum();
                for a in 0..d { w[a] -= c[a] * proj; }
            }
            norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-8 { break; }
            for x in w.iter_mut() { *x /= norm; }
            let delta: f32 = w.iter().zip(&v).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
            v = w;
            if delta < 1e-6 { break; }
        }
        let mut cv = vec![0.0f32; d];
        for a in 0..d {
            for b in 0..d { cv[a] += cov_work[a * d + b] * v[b]; }
        }
        let lambda: f32 = v.iter().zip(&cv).map(|(x, y)| x * y).sum();
        for a in 0..d {
            for b in 0..d { cov_work[a * d + b] -= lambda * v[a] * v[b]; }
        }
        components.push(v);
    }

    let kk = components.len();
    let mut out = vec![0.0f32; n * kk];
    for i in 0..n {
        for (j, c) in components.iter().enumerate() {
            out[i * kk + j] = (0..d).map(|a| feats[i * d + a] * c[a]).sum();
        }
    }
    (out, kk)
}

// ---------------------------------------------------------------------------
// Fuzzy C-Means (k = 2). Returns hard labels (argmax of membership).
// ---------------------------------------------------------------------------

fn fuzzy_c_means(
    feats: &[f32],
    n: usize,
    d: usize,
    m: f32,
    max_iters: usize,
    tol: f32,
) -> Vec<usize> {
    let k = 2usize;

    let mut state: u64 = 42;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as f32 / (1u64 << 32) as f32
    };

    let mut u = vec![0.0f32; n * k];
    for i in 0..n {
        let mut s = 0.0f32;
        for j in 0..k {
            let r = next();
            u[i * k + j] = r;
            s += r;
        }
        if s > 0.0 {
            for j in 0..k { u[i * k + j] /= s; }
        }
    }

    let power = 2.0 / (m - 1.0);
    let mut centroids = vec![0.0f32; k * d];

    for _ in 0..max_iters {
        let mut weights = vec![0.0f32; k];
        for v in centroids.iter_mut() { *v = 0.0; }
        for i in 0..n {
            for j in 0..k {
                let w = u[i * k + j].powf(m);
                weights[j] += w;
                for a in 0..d { centroids[j * d + a] += w * feats[i * d + a]; }
            }
        }
        for j in 0..k {
            let wj = weights[j].max(1e-12);
            for a in 0..d { centroids[j * d + a] /= wj; }
        }

        let mut shift = 0.0f32;
        for i in 0..n {
            let mut dists = [0.0f32; 2];
            for j in 0..k {
                let mut s = 0.0f32;
                for a in 0..d {
                    let diff = feats[i * d + a] - centroids[j * d + a];
                    s += diff * diff;
                }
                dists[j] = s.sqrt().max(1e-12);
            }
            let mut row = [0.0f32; 2];
            for j in 0..k { row[j] = 1.0 / dists[j].powf(power); }
            let sum = row[0] + row[1];
            for j in 0..k {
                let new_v = row[j] / sum;
                shift += (new_v - u[i * k + j]).powi(2);
                u[i * k + j] = new_v;
            }
        }
        if shift.sqrt() < tol { break; }
    }

    (0..n)
        .map(|i| if u[i * k] >= u[i * k + 1] { 0 } else { 1 })
        .collect()
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Estimate a `Background` from the morphological flat-region segmentation,
/// using the attributes already present on `tree`.
pub fn estimate_morphological_background(
    img: &[f32],
    tree: &MaxTree,
    alpha: f64,
) -> Background {
    let n = tree.num_nodes();

    // Degenerate tree: constant image.
    if n < 2 {
        let v = tree.level[tree.root()] as f64;
        let soft_bias = image_soft_bias(img);
        let gain = if v > soft_bias { 1.0 } else { 0.0 };
        return Background { mean: v, var: 1.0, gain, alpha };
    }

    let root = tree.root();
    let (root_cx, root_cy) = (tree.cx[root], tree.cy[root]);

    // --- 5-D features per node (uses the pre-computed attributes) ---
    const D: usize = 5;
    let mut feats = vec![0.0f32; n * D];
    for id in 0..n {
        let area = tree.area[id].max(1) as f32;
        let vol = tree.volume[id].max(1e-12) as f32;
        let alt = tree.level[id].max(1e-12) as f32;
        let dx = tree.cx[id] - root_cx;
        let dy = tree.cy[id] - root_cy;
        feats[id * D + 0] = tree.gamma[id] as f32;
        feats[id * D + 1] = area.log10();
        feats[id * D + 2] = vol.log10();
        feats[id * D + 3] = alt.log10();
        feats[id * D + 4] = (dx * dx + dy * dy).sqrt();
    }

    standardize(&mut feats, n, D);
    let (reduced, rd) = pca(&feats, n, D, 2);
    let labels = fuzzy_c_means(&reduced, n, rd, 2.0, 100, 1e-6);

    // --- pick the background cluster ---
    let mut sum_area = [0.0f64; 2];
    let mut sum_alt = [0.0f64; 2];
    let mut cnt = [0usize; 2];
    for id in 0..n {
        let l = labels[id];
        sum_area[l] += tree.area[id] as f64;
        sum_alt[l] += tree.level[id] as f64;
        cnt[l] += 1;
    }
    let mean_area = [
        sum_area[0] / cnt[0].max(1) as f64,
        sum_area[1] / cnt[1].max(1) as f64,
    ];
    let mean_alt = [
        sum_alt[0] / cnt[0].max(1) as f64,
        sum_alt[1] / cnt[1].max(1) as f64,
    ];
    let keep_label = if (mean_area[0] > mean_area[1])
        || (mean_area[0] == mean_area[1] && mean_alt[0] < mean_alt[1])
    {
        0
    } else {
        1
    };
    let mean_area_keep = mean_area[keep_label];

    // --- keep large background nodes; others inherit from parent ---
    let mut keep_node = vec![false; n];
    for id in 0..n {
        if labels[id] == keep_label && (tree.area[id] as f64) >= mean_area_keep {
            keep_node[id] = true;
        }
        if (tree.area[id] as f64) >= mean_area_keep {
            keep_node[id] = true;
        }
    }

    let mut node_value = vec![f32::NAN; n];
    for id in 0..n {
        if keep_node[id] { node_value[id] = tree.level[id]; }
    }
    for &nd in tree.order.iter().rev() {
        let id = nd as usize;
        if id == root { continue; }
        if node_value[id].is_nan() {
            let p = tree.parent_node[id] as usize;
            if !node_value[p].is_nan() { node_value[id] = node_value[p]; }
        }
    }
    if node_value[root].is_nan() {
        node_value[root] = tree.level[root];
    }
    for id in 0..n {
        if node_value[id].is_nan() { node_value[id] = node_value[root]; }
    }

    // --- aggregate the per-pixel background into mean/var ---
    let n_pix = tree.num_pixels();
    let mut s = 0.0f64;
    let mut ss = 0.0f64;
    for p in 0..n_pix {
        let node = tree.node_of_pixel[p] as usize;
        let v = node_value[node] as f64;
        if v.is_finite() {
            s += v;
            ss += v * v;
        }
    }
    let np = n_pix.max(1) as f64;
    let mean = s / np;
    let var = (ss / np - mean * mean).max(0.0);

    let soft_bias = image_soft_bias(img);
    let gain = if var > 0.0 { (mean - soft_bias) / var } else { 0.0 };

    eprintln!(
        "morph bg: kept {}/{} nodes, mean={:.4e}, sigma={:.4e}, gain={:.4e}",
        keep_node.iter().filter(|&&b| b).count(),
        n,
        mean,
        var.sqrt(),
        gain,
    );

    Background { mean, var, gain, alpha }
}

#[inline]
fn image_soft_bias(img: &[f32]) -> f64 {
    let mn = img
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, |a, b| a.min(b as f64));
    if mn.is_finite() && mn < 0.0 { mn } else { 0.0 }
}