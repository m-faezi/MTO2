//! Node-oriented max-tree for a 2D f32 image, plus region attributes.
//!
//! The attributes are computed in a single bottom-up pass over `order` (which
//! is sorted by ascending level, so reverse iteration visits children first).
//! Everything a downstream stage needs -- area, volume, gamma, per-node
//! mean/variance, centroid -- is materialized here so no other module has to
//! re-traverse the tree.

const UNDEF: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    Four,
    Eight,
}

impl Connectivity {
    #[inline]
    fn offsets(self) -> &'static [(isize, isize)] {
        match self {
            Connectivity::Four => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            Connectivity::Eight => &[
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, -1), (-1, 1), (1, 1),
            ],
        }
    }
}

/// Max-tree topology. Attribute vectors are filled by `compute_attributes`.
pub struct MaxTree {
    pub width: usize,
    pub height: usize,
    pub node_of_pixel: Vec<u32>,
    pub parent_node: Vec<u32>,
    pub level: Vec<f32>,
    pub seed_pixel: Vec<u32>,
    pub order: Vec<u32>,

    // ---- region attributes (filled by `compute_attributes`) ----
    pub area: Vec<u32>,
    pub volume: Vec<f64>,
    pub gamma: Vec<u32>,
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
    pub cx: Vec<f32>,
    pub cy: Vec<f32>,
}

pub struct TreeStats {
    pub nodes: usize,
    pub leaves: usize,
    pub max_depth: usize,
}

impl MaxTree {
    #[inline]
    pub fn num_nodes(&self) -> usize { self.parent_node.len() }

    #[inline]
    pub fn num_pixels(&self) -> usize { self.node_of_pixel.len() }

    #[inline]
    pub fn root(&self) -> usize { 0 }

    #[inline]
    pub fn root_pixel(&self) -> usize { self.seed_pixel[0] as usize }

    #[allow(dead_code)]
    #[inline]
    pub fn node_xy(&self, node: usize) -> (usize, usize) {
        let p = self.seed_pixel[node] as usize;
        (p % self.width, p / self.width)
    }

    pub fn stats(&self) -> TreeStats {
        let n = self.num_nodes();
        let mut has_child = vec![false; n];
        let mut depth = vec![0u32; n];
        let mut max_depth = 0usize;
        for &nd in &self.order {
            let nd = nd as usize;
            if nd == self.root() { continue; }
            let par = self.parent_node[nd] as usize;
            has_child[par] = true;
            depth[nd] = depth[par] + 1;
            max_depth = max_depth.max(depth[nd] as usize);
        }
        let leaves = (0..n).filter(|&k| !has_child[k]).count();
        TreeStats { nodes: n, leaves, max_depth }
    }

    /// Returns true for nodes that have no children (i.e., leaves).
    pub fn is_leaf(&self) -> Vec<bool> {
        let n = self.num_nodes();
        let mut has_child = vec![false; n];
        for &nd in &self.order {
            let nd = nd as usize;
            if nd == self.root() { continue; }
            has_child[self.parent_node[nd] as usize] = true;
        }
        (0..n).map(|k| !has_child[k]).collect()
    }
}

// ---------------------------------------------------------------------------
// Tree construction
// ---------------------------------------------------------------------------

#[inline]
fn find_root(zpar: &mut [u32], mut x: u32) -> u32 {
    let mut r = x;
    while zpar[r as usize] != r { r = zpar[r as usize]; }
    while zpar[x as usize] != r {
        let next = zpar[x as usize];
        zpar[x as usize] = r;
        x = next;
    }
    r
}

/// Build the max-tree. Attributes are left zero-initialized; call
/// `compute_attributes` (usually right after) to fill them.
pub fn build_max_tree(
    img: &[f32],
    width: usize,
    height: usize,
    conn: Connectivity,
) -> MaxTree {
    let n = width * height;
    assert_eq!(img.len(), n);
    assert!(n > 0 && n <= u32::MAX as usize, "image dimension out of range");

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| img[a as usize].total_cmp(&img[b as usize]));

    let mut zpar: Vec<u32> = vec![UNDEF; n];
    let mut parent_pix: Vec<u32> = vec![UNDEF; n];
    let offsets = conn.offsets();
    let w = width as isize;
    let h = height as isize;

    for &pi in order.iter().rev() {
        let p = pi as usize;
        parent_pix[p] = p as u32;
        zpar[p] = p as u32;
        let x = (p % width) as isize;
        let y = (p / width) as isize;
        for &(dx, dy) in offsets {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= w || ny >= h { continue; }
            let q = (ny * w + nx) as usize;
            if parent_pix[q] == UNDEF { continue; }
            let r = find_root(&mut zpar, q as u32);
            if r != p as u32 {
                parent_pix[r as usize] = p as u32;
                zpar[r as usize] = p as u32;
            }
        }
    }
    drop(zpar);

    for &pi in &order {
        let p = pi as usize;
        let q = parent_pix[p] as usize;
        let pq = parent_pix[q] as usize;
        if img[pq] == img[q] {
            parent_pix[p] = pq as u32;
        }
    }

    let mut node_of_pixel: Vec<u32> = vec![UNDEF; n];
    let mut parent_node: Vec<u32> = Vec::new();
    let mut level: Vec<f32> = Vec::new();
    let mut seed_pixel: Vec<u32> = Vec::new();
    let mut node_order: Vec<u32> = Vec::new();

    for &pi in &order {
        let p = pi as usize;
        let par = parent_pix[p] as usize;
        let is_node = p == par || img[par] != img[p];
        if is_node {
            let id = parent_node.len() as u32;
            node_of_pixel[p] = id;
            let parent_id = if p == par { id } else { node_of_pixel[par] };
            parent_node.push(parent_id);
            level.push(img[p]);
            seed_pixel.push(p as u32);
            node_order.push(id);
        }
    }
    for &pi in &order {
        let p = pi as usize;
        if node_of_pixel[p] == UNDEF {
            let par = parent_pix[p] as usize;
            node_of_pixel[p] = node_of_pixel[par];
        }
    }

    let n_nodes = parent_node.len();
    MaxTree {
        width,
        height,
        node_of_pixel,
        parent_node,
        level,
        seed_pixel,
        order: node_order,

        area: vec![0; n_nodes],
        volume: vec![0.0; n_nodes],
        gamma: vec![0; n_nodes],
        mean: vec![0.0; n_nodes],
        variance: vec![0.0; n_nodes],
        cx: vec![0.0; n_nodes],
        cy: vec![0.0; n_nodes],
    }
}

// ---------------------------------------------------------------------------
// Attribute computation -- one pass, in-place on `tree`.
// ---------------------------------------------------------------------------

/// Fill `area`, `volume`, `gamma`, `mean`, `variance`, `cx`, `cy` on `tree`.
///
/// All accumulations are bottom-up over `tree.order.iter().rev()` (children
/// before parents). Formulas match the original `attributes.rs` from the
/// morph-bg project exactly:
///   * `area[n]`     = number of pixels in node `n`'s subtree
///   * `mean[n]`     = mean of `img` over that subtree
///   * `variance[n]` = variance of `img` over that subtree
///   * `cx, cy`      = centroid of that subtree
///   * `volume[n]`   = sum over descendants d of `level[d] * area[d]`
///   * `gamma[n]`    = topological height (max distance to a leaf)
pub fn compute_attributes(tree: &mut MaxTree, img: &[f32]) {
    let n = tree.num_nodes();
    let w = tree.width;

    for v in tree.area.iter_mut() { *v = 0; }
    for v in tree.volume.iter_mut() { *v = 0.0; }
    for v in tree.gamma.iter_mut() { *v = 0; }
    for v in tree.mean.iter_mut() { *v = 0.0; }
    for v in tree.variance.iter_mut() { *v = 0.0; }
    for v in tree.cx.iter_mut() { *v = 0.0; }
    for v in tree.cy.iter_mut() { *v = 0.0; }

    let mut sum = vec![0.0f32; n];
    let mut sum_sq = vec![0.0f32; n];
    let mut sx = vec![0.0f32; n];
    let mut sy = vec![0.0f32; n];

    for p in 0..tree.num_pixels() {
        let node = tree.node_of_pixel[p] as usize;
        let (py, px) = (p / w, p % w);
        let v = img[p];
        tree.area[node] += 1;
        sum[node] += v;
        sum_sq[node] += v * v;
        sx[node] += px as f32;
        sy[node] += py as f32;
    }

    for &nd in tree.order.iter().rev() {
        let node = nd as usize;
        if node == tree.root() { continue; }
        let p = tree.parent_node[node] as usize;
        tree.area[p] += tree.area[node];
        sum[p] += sum[node];
        sum_sq[p] += sum_sq[node];
        sx[p] += sx[node];
        sy[p] += sy[node];
    }

    for node in 0..n {
        let a = tree.area[node].max(1) as f64;
        let s = sum[node] as f64;
        let ss = sum_sq[node] as f64;
        let m = (s / a) as f32;
        tree.mean[node] = m;
        let var = (ss / a) - (m as f64).powi(2);
        tree.variance[node] = var.max(0.0) as f32;
        tree.cx[node] = sx[node] / tree.area[node].max(1) as f32;
        tree.cy[node] = sy[node] / tree.area[node].max(1) as f32;
    }

    for &nd in tree.order.iter().rev() {
        let node = nd as usize;
        if node == tree.root() {
            tree.volume[node] += (tree.level[node] as f64) * (tree.area[node] as f64);
            continue;
        }
        let p = tree.parent_node[node] as usize;
        tree.volume[p] +=
            tree.volume[node] + (tree.level[node] as f64) * (tree.area[node] as f64);
        tree.gamma[p] = tree.gamma[p].max(tree.gamma[node] + 1);
    }
}

pub fn replace_nan(img: &mut [f32]) -> usize {
    let min = img.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
    let fill = if min.is_finite() { min } else { 0.0 };
    let mut count = 0;
    for v in img.iter_mut() {
        if !v.is_finite() { *v = fill; count += 1; }
    }
    count
}

#[allow(dead_code)]
pub fn quantize(img: &mut [f32]) {
    for v in img.iter_mut() {
        *v = v.round();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<f32> {
        vec![1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 7.0]
    }

    #[test]
    fn four_connectivity_gives_separate_peaks() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Four);
        let s = t.stats();
        assert_eq!(s.nodes, 3);
        assert_eq!(s.leaves, 2);
    }

    #[test]
    fn eight_connectivity_nests_peaks() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Eight);
        let s = t.stats();
        assert_eq!(s.nodes, 3);
        assert_eq!(s.leaves, 1);
        assert_eq!(s.max_depth, 2);
    }

    #[test]
    fn constant_image_is_single_node() {
        let img = vec![3.0; 12];
        let t = build_max_tree(&img, 4, 3, Connectivity::Four);
        let s = t.stats();
        assert_eq!(s.nodes, 1);
        assert_eq!(s.leaves, 1);
    }

    #[test]
    fn attributes_area_root_is_all_pixels() {
        let img = vec![
            0.0, 2.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 4.0,
        ];
        for conn in [Connectivity::Four, Connectivity::Eight] {
            let mut t = build_max_tree(&img, 4, 3, conn);
            compute_attributes(&mut t, &img);
            assert_eq!(t.area[t.root()] as usize, img.len());
        }
    }

    #[test]
    fn attributes_volume_root_sums_level_times_area() {
        let img = vec![
            0.0, 2.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 4.0,
        ];
        for conn in [Connectivity::Four, Connectivity::Eight] {
            let mut t = build_max_tree(&img, 4, 3, conn);
            compute_attributes(&mut t, &img);
            let expected: f64 = (0..t.num_nodes())
                .map(|n| t.level[n] as f64 * t.area[n] as f64)
                .sum();
            assert!((t.volume[t.root()] - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn attributes_mean_variance_match_pixels() {
        let img = vec![
            0.0, 2.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 4.0,
        ];
        let mut t = build_max_tree(&img, 4, 3, Connectivity::Four);
        compute_attributes(&mut t, &img);
        let n = img.len() as f64;
        let mean: f64 = img.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var: f64 = img.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let root = t.root();
        assert!((t.mean[root] as f64 - mean).abs() < 1e-5);
        assert!((t.variance[root] as f64 - var).abs() < 1e-5);
    }

    #[test]
    fn attributes_centroid_root_is_image_center() {
        let img = vec![1.0f32; 4 * 3];
        let mut t = build_max_tree(&img, 4, 3, Connectivity::Four);
        compute_attributes(&mut t, &img);
        let root = t.root();
        assert!((t.cx[root] - 1.5).abs() < 1e-5);
        assert!((t.cy[root] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn nan_replacement() {
        let mut img = vec![f32::NAN, 2.0, 1.0, f32::INFINITY];
        assert_eq!(replace_nan(&mut img), 2);
        assert_eq!(img, vec![1.0, 2.0, 1.0, 1.0]);
    }
}