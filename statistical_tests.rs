//! Port of MTO2's `statistical_tests.py` to node-oriented representation.

use crate::maxtree::MaxTree;

const EPS: f64 = f32::EPSILON as f64;

pub struct Background {
    #[allow(dead_code)]
    pub mean: f64,
    pub var: f64,
    pub gain: f64,
    pub alpha: f64,
}

/// Standard-normal quantile via Acklam's approximation (|err| < 1.15e-9).
fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Wilson–Hilferty approximation of χ² quantile at probability p, dof k.
fn chi2_ppf(p: f64, k: f64) -> f64 {
    if k <= 0.0 { return 0.0; }
    let z = norm_ppf(p);
    let t = 1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt();
    if t <= 0.0 { 0.0 } else { k * t * t * t }
}

pub fn significance(
    tree: &MaxTree,
    area: &[u32],
    volume: &[f64],
    bg: &Background,
) -> Vec<bool> {
    let n = tree.num_nodes();
    let mut sig = vec![false; n];
    let p = 1.0 - bg.alpha;
    let gain = bg.gain + EPS;

    for nd in 0..n {
        if nd == tree.root() { continue; }
        let par = tree.parent_node[nd] as usize;
        let denom = bg.var + tree.level[par] as f64 / gain;
        let denom = if denom == 0.0 { EPS } else { denom };
        let vn = volume[nd] / denom;
        let thresh = chi2_ppf(p, area[nd] as f64);
        sig[nd] = vn > thresh;
    }
    for nd in 0..n {
        if area[nd] <= 1 { sig[nd] = false; }
    }
    sig
}

/// Higra-compatible object selection.
///
/// 1. Collapse non-significant nodes to their nearest significant ancestor.
/// 2. At each significant parent, the largest significant child is the
///    "main branch"; all *other* significant children are objects.
/// 3. If nothing survives (degenerate tree, single linear chain), keep the
///    deepest significant node as one object.
///
/// Returns (is_object, is_main_branch).
pub fn select_objects(
    tree: &MaxTree,
    area: &[u32],
    sig: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let n = tree.num_nodes();
    let root = tree.root();

    // Parent in the simplified tree = nearest significant ancestor.
    let mut parent_in_simplified: Vec<Option<usize>> = vec![None; n];
    {
        let mut nearest_sig: Vec<Option<usize>> = vec![None; n];
        nearest_sig[root] = if sig[root] { Some(root) } else { None };
        for &nd in &tree.order {
            let nd = nd as usize;
            if nd == root { continue; }
            let par = tree.parent_node[nd] as usize;
            nearest_sig[nd] = if sig[nd] { Some(nd) } else { nearest_sig[par] };
        }
        for &nd in &tree.order {
            let nd = nd as usize;
            if !sig[nd] { continue; }
            if nd == root { parent_in_simplified[nd] = None; continue; }
            let par = tree.parent_node[nd] as usize;
            parent_in_simplified[nd] = nearest_sig[par];
        }
    }

    // Largest child in the simplified tree per parent.
    let mut largest_child: Vec<usize> = vec![usize::MAX; n];
    let mut largest_area: Vec<u32> = vec![0; n];
    for &nd in &tree.order {
        let nd = nd as usize;
        if !sig[nd] { continue; }
        let Some(par_sig) = parent_in_simplified[nd] else { continue };
        if area[nd] > largest_area[par_sig] {
            largest_area[par_sig] = area[nd];
            largest_child[par_sig] = nd;
        }
    }

    let mut is_main = vec![false; n];
    for nd in 0..n {
        if largest_child[nd] != usize::MAX {
            is_main[largest_child[nd]] = true;
        }
    }

    // Object = significant, not root, not the largest child of its parent.
    let mut is_obj = vec![false; n];
    for &nd in &tree.order {
        let nd = nd as usize;
        if !sig[nd] || nd == root { continue; }
        if !is_main[nd] {
            is_obj[nd] = true;
        }
    }

    // Fallback for degenerate trees: keep the deepest significant node.
    if !is_obj.iter().any(|b| *b) {
        let mut best: Option<usize> = None;
        for &nd in &tree.order {
            let nd = nd as usize;
            if !sig[nd] || nd == root { continue; }
            best = Some(match best {
                None => nd,
                Some(b) if area[nd] < area[b] => nd, // prefer smallest significant
                Some(b) => b,
            });
        }
        if let Some(nd) = best {
            is_obj[nd] = true;
        }
    }

    (is_obj, is_main)
}

pub fn detect(
    tree: &MaxTree,
    area: &[u32],
    volume: &[f64],
    bg: &Background,
) -> (Vec<bool>, Vec<bool>) {
    let sig = significance(tree, area, volume, bg);
    let (obj, _main) = select_objects(tree, area, &sig);
    (sig, obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_ppf_sanity() {
        assert!((norm_ppf(0.5)).abs() < 1e-12);
        assert!((norm_ppf(0.975) - 1.959963984540054).abs() < 1e-6);
    }

    #[test]
    fn chi2_ppf_sanity() {
        let k = 100.0;
        let got = chi2_ppf(1.0 - 1e-6, k);
        let want = 189.884913393596;
        assert!((got - want).abs() / want < 0.02);
    }
}