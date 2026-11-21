//! Statistical significance + object selection, ported from MTO2.

use crate::maxtree::MaxTree;

pub struct Background {
    pub mean: f64,
    pub var: f64,
    pub gain: f64,
    pub alpha: f64,
}

/// Wilson–Hilferty approximation of the chi-square quantile.
fn chi2_ppf(p: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 0.0;
    }
    let z = normal_quantile(p);
    let a = 2.0 / (9.0 * df);
    let inner = (1.0 - a + z * a.sqrt()).max(0.0);
    df * inner.powi(3)
}

/// Acklam's normal quantile approximation.
fn normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);

    const A: [f64; 6] = [
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
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

/// Result of object selection: everything `build_segmentation` needs.
pub struct Selection {
    /// Per original node: nearest significant ancestor (inclusive).
    pub nearest_sig_ancestor: Vec<u32>,
    /// Per significant node: contracted parent (nearest significant ancestor of its max-tree parent).
    /// `None` for root.
    pub contracted_parent: Vec<Option<u32>>,
    /// Significant-node test result.
    pub significant: Vec<bool>,
    /// Object flag (on the original node ids).
    pub objects: Vec<bool>,
    /// Main-branch flag (on the original node ids).
    pub main_branch: Vec<bool>,
}

pub fn detect(
    tree: &MaxTree,
    area: &[u32],
    volume: &[f64],
    bg: &Background,
) -> Selection {
    let n = tree.num_nodes();
    let is_leaf = tree.is_leaf();
    let eps = f64::EPSILON;

    // ---- significance test (unchanged) ----
    let mut sig = vec![false; n];
    for nd in 0..n {
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

    // ---- object selection on the contracted tree ----
    select_objects(tree, area, &sig)
}

/// Port of MTO2 `select_objects` (with `area_ratio` deblending).
///
/// 1. Contract non-significant nodes into their nearest significant ancestor.
/// 2. Apply area-ratio deblending: if a significant child's area is
///    >= area_ratio * parent's area, merge it into the parent branch
///    (i.e., treat it as part of the parent, not an independent object).
/// 3. Mark the largest-area child of each node as the "main branch".
/// 4. Objects = significant nodes that are NOT on any main branch,
///    after demoting root's contracted children when the root is not
///    significant.
fn select_objects(tree: &MaxTree, area: &[u32], sig: &[bool]) -> Selection {
    let n = tree.num_nodes();
    let root = tree.root();
    let area_ratio = 0.90_f64; // MTO2 default

    // 1. nearest significant ancestor (inclusive).
    let mut nsa = vec![root as u32; n];
    for &nd in &tree.order {
        let nd = nd as usize;
        let par = tree.parent_node[nd] as usize;
        let from_parent = if par == nd { root as u32 } else { nsa[par] };
        nsa[nd] = if sig[nd] { nd as u32 } else { from_parent };
    }

    // 2. contracted parent + area-ratio merging.
    let mut contracted_parent: Vec<Option<u32>> = vec![None; n];
    let mut merged_into_parent = vec![false; n];

    for nd in 0..n {
        if !sig[nd] || nd == root {
            continue;
        }
        let par = tree.parent_node[nd] as usize;
        let cp = nsa[par] as usize;
        if cp == nd {
            continue;
        }
        contracted_parent[nd] = Some(cp as u32);

        // area_ratio deblending: large children collapse into parent.
        if (area[nd] as f64) >= area_ratio * area[cp] as f64 {
            merged_into_parent[nd] = true;
        }
    }

    // 3. largest contracted child per node.
    let mut largest_child: Vec<Option<usize>> = vec![None; n];
    let mut largest_area: Vec<u32> = vec![0; n];
    for nd in 0..n {
        if let Some(cp) = contracted_parent[nd] {
            let cp = cp as usize;
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

    // 4. main-branch flags.
    let mut main_branch = vec![false; n];
    for cp in 0..n {
        if let Some(lc) = largest_child[cp] {
            main_branch[lc] = true;
        }
    }
    // area_ratio-merges also count as main branch (absorbed into parent).
    for nd in 0..n {
        if merged_into_parent[nd] {
            main_branch[nd] = true;
        }
    }

    // 5. demote root's contracted children if root is not significant.
    if !sig[root] {
        for nd in 0..n {
            if let Some(cp) = contracted_parent[nd] {
                if cp as usize == root {
                    main_branch[nd] = false;
                }
            }
        }
    }

    // 6. objects = significant AND not main branch.
    let mut objects = vec![false; n];
    for nd in 0..n {
        objects[nd] = sig[nd] && !main_branch[nd];
    }
    objects[root] = false;

    Selection {
        nearest_sig_ancestor: nsa,
        contracted_parent,
        significant: sig.to_vec(),
        objects,
        main_branch,
    }
}

/// Port of MTO2 `MaxTree.move_up` (isophote refinement).
///
/// Every detected object node is moved up its main branch (towards brighter
/// levels) to the first node whose level reaches
/// `level[obj] + |move_factor * local_noise|`, where
/// `local_noise = sqrt((bg.var + level[parent(obj)]) / bg.gain)`.
/// A move is only valid while the node's `area / parent_area >= area_ratio`.
///
/// Returns the `corrected_segments` flags (per original node id). With
/// `move_factor = 0` the thresholds equal the object levels themselves.
pub fn move_up(
    tree: &MaxTree,
    area: &[u32],
    sel: &Selection,
    bg: &Background,
    move_factor: f64,
    area_ratio: f64,
) -> Vec<bool> {
    let n = tree.num_nodes();
    let root = tree.root();

    // Main branch on the FULL tree: largest-area child of every node.
    let mut largest_child: Vec<Option<usize>> = vec![None; n];
    for nd in 0..n {
        if nd == root {
            continue;
        }
        let par = tree.parent_node[nd] as usize;
        let replace = match largest_child[par] {
            None => true,
            Some(lc) => area[nd] > area[lc], // ascending nd => ties keep the lowest id
        };
        if replace {
            largest_child[par] = Some(nd);
        }
    }
    let is_main = |nd: usize| -> bool {
        nd != root && largest_child[tree.parent_node[nd] as usize] == Some(nd)
    };

    // closest_object_ancestor: climb while the node is main-branch AND not an
    // object; stop at the first object or non-main-branch node (or the root).
    let mut cao = vec![root as u32; n];
    for &nd in &tree.order {
        let nd = nd as usize;
        let par = tree.parent_node[nd] as usize;
        cao[nd] = if nd != root && is_main(nd) && !sel.objects[nd] {
            cao[par]
        } else {
            nd as u32
        };
    }

    // Target level of an object after the move.
    let target = |o: usize| -> f64 {
        let par = tree.parent_node[o] as usize;
        let above_bg = (tree.level[par] as f64 - bg.mean).max(0.0);
        let noise = if bg.gain != 0.0 {
            ((bg.var + above_bg) / bg.gain).max(0.0).sqrt()
        } else {
            0.0
        };
        tree.level[o] as f64 + (move_factor * noise).abs()
    };

    // valid_moves
    let mut valid = vec![false; n];
    for nd in 0..n {
        let c = cao[nd] as usize;
        if !sel.objects[c] {
            continue;
        }
        let par = tree.parent_node[nd] as usize; // root: par == nd => ratio 1.0
        let ratio = area[nd] as f64 / area[par] as f64;
        valid[nd] = tree.level[nd] as f64 >= target(c) && ratio >= area_ratio;
    }

    // corrected_segments: first valid node along each moved branch.
    let mut corrected = vec![false; n];
    for nd in 0..n {
        if !valid[nd] {
            continue;
        }
        let par = tree.parent_node[nd] as usize;
        corrected[nd] = !valid[par] || cao[par] != cao[nd];
    }
    corrected
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
        assert!((chi2_ppf(0.95, 10.0) - 18.307).abs() < 0.5);
        assert!((chi2_ppf(0.5, 1.0) - 0.455).abs() < 0.05);
    }
}