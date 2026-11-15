//! Statistical significance + object selection, ported from MTO2.

use crate::maxtree::MaxTree;

pub struct Background {
    pub mean: f64,
    pub var: f64,
    pub gain: f64,
    pub alpha: f64,
}

/// Wilson–Hilferty approximation of the chi-square quantile.
/// Accurate to ~1e-3 for df > 10, adequate here.
fn chi2_ppf(p: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 0.0;
    }
    let z = normal_quantile(p);
    let a = 2.0 / (9.0 * df);
    let inner = (1.0 - a + z * a.sqrt()).max(0.0);
    df * inner.powi(3)
}

/// Beasley–Springer–Moro normal quantile approximation.
fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);

    // Coefficients (Acklam's approximation)
    const A: [f64; 6] = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
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

/// Returns (significant, objects).
///
/// Significant-node test (ported from MTO2):
///     V_norm = V / (var + parent_altitude / gain)
///     significant = V_norm > chi2.ppf(alpha, area)
///     (leaves are excluded)
///
/// Object selection (ported from MTO2 `select_objects`):
///     - collapse the tree to its significant nodes
///     - find the "main branch" in the collapsed tree (largest-area child)
///     - objects = significant nodes that are NOT main-branch children
pub fn detect(
    tree: &MaxTree,
    area: &[u32],
    volume: &[f64],
    bg: &Background,
) -> (Vec<bool>, Vec<bool>) {
    let n = tree.num_nodes();
    let root = tree.root();
    let is_leaf = tree.is_leaf();
    let eps = f64::EPSILON;

    let mut sig = vec![false; n];
    for nd in 0..n {
        if nd == root {
            // Test the root too (MTO2 tests it; it's usually not significant).
        }
        if is_leaf[nd] || area[nd] <= 1 {
            continue;
        }
        let par = tree.parent_node[nd] as usize;
        let parent_level = tree.level[par] as f64;

        let denom = bg.var + parent_level / (bg.gain + eps);
        if denom <= 0.0 {
            continue;
        }
        let v_norm = volume[nd] / denom;
        let thr = chi2_ppf(bg.alpha, area[nd] as f64);
        sig[nd] = v_norm > thr;
    }

    // ---- select_objects ----
    let obj = select_objects(tree, area, &sig);
    (sig, obj)
}

/// Port of MTO2 `select_objects`, without higra.
///
/// Steps:
/// 1. Keep only significant nodes; contract non-significant ones into
///    their nearest significant ancestor (or into the root if none).
/// 2. In the contracted tree, for every node record which child has
///    the largest area (ties -> smallest node id, like argmax).
/// 3. A significant node is an "object" if it is NOT a main-branch
///    child of its contracted parent. Additionally, if the root is not
///    significant, its contracted children are treated as non-main-branch.
fn select_objects(tree: &MaxTree, area: &[u32], sig: &[bool]) -> Vec<bool> {
    let n = tree.num_nodes();
    let root = tree.root();

    // 1. nearest significant ancestor (inclusive) for each node.
    //    We process nodes in tree order (parents before children, since
    //    `order` here is the BFS/DFS build order which has parents first).
    let mut nsa = vec![root; n];
    for &nd in &tree.order {
        let nd = nd as usize;
        let par = tree.parent_node[nd] as usize;
        let from_parent = if par == nd { root } else { nsa[par] };
        nsa[nd] = if sig[nd] { nd } else { from_parent };
    }

    // 2. contracted parent for each significant node.
    let mut contracted_parent: Vec<Option<usize>> = vec![None; n];
    for nd in 0..n {
        if !sig[nd] || nd == root {
            continue;
        }
        let par = tree.parent_node[nd] as usize;
        let cp = nsa[par]; // nearest significant ancestor of the parent
        if cp != nd {
            contracted_parent[nd] = Some(cp);
        }
    }

    // 3. largest child of each node in the contracted tree.
    //    For each significant node, look at its contracted children.
    let mut largest_child: Vec<Option<usize>> = vec![None; n];
    let mut largest_area: Vec<u32> = vec![0; n];
    for nd in 0..n {
        if let Some(cp) = contracted_parent[nd] {
            let a = area[nd];
            let replace = match largest_child[cp] {
                None => true,
                Some(lc) => a > largest_area[cp] || (a == largest_area[cp] && nd < lc),
            };
            if replace {
                largest_child[cp] = Some(nd);
                largest_area[cp] = a;
            }
        }
    }

    // 4. main-branch flags in the contracted tree.
    let mut main_branch = vec![false; n];
    for cp in 0..n {
        if let Some(lc) = largest_child[cp] {
            main_branch[lc] = true;
        }
    }

    // 5. If the root is not significant, its contracted children are
    //    demoted from main-branch (they become objects).
    if !sig[root] {
        for nd in 0..n {
            if let Some(cp) = contracted_parent[nd] {
                if cp == root {
                    main_branch[nd] = false;
                }
            }
        }
    }

    // 6. objects = significant AND not main branch.
    let mut obj = vec![false; n];
    for nd in 0..n {
        obj[nd] = sig[nd] && !main_branch[nd];
    }
    obj[root] = false;
    obj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_quantile_sanity() {
        assert!((normal_quantile(0.5)).abs() < 1e-6);
        assert!((normal_quantile(0.975) - 1.959964).abs() < 1e-3);
        assert!((normal_quantile(0.025) + 1.959964).abs() < 1e-3);
    }

    #[test]
    fn chi2_sanity() {
        // chi2.ppf(0.95, 10) ≈ 18.307
        assert!((chi2_ppf(0.95, 10.0) - 18.307).abs() < 0.5);
        // chi2.ppf(0.5, 1) ≈ 0.455
        assert!((chi2_ppf(0.5, 1.0) - 0.455).abs() < 0.05);
    }
}