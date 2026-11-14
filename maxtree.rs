//! Node-oriented max-tree for a 2D f32 image.
//!
//! Two-index scheme (same as Higra / MTO2):
//!   * `node_of_pixel[p]` : node id containing pixel p
//!   * `parent_node[n]`   : parent node id (root -> itself)
//!   * `level[n]`         : altitude of node n
//!   * `seed_pixel[n]`    : a representative pixel of node n
//!   * `order[n]`         : topological order (parents before children)
//!
//! Attribute arrays have length `num_nodes()`, not `width*height`, which is
//! the main memory win over a pixel-oriented representation.

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

pub struct MaxTree {
    pub width: usize,
    pub height: usize,
    /// Length n = width*height.
    pub node_of_pixel: Vec<u32>,
    /// Length num_nodes. Root is its own parent.
    pub parent_node: Vec<u32>,
    /// Length num_nodes. Gray level of each node.
    pub level: Vec<f32>,
    /// Length num_nodes. Representative pixel (canonical element) of each node.
    pub seed_pixel: Vec<u32>,
    /// Length num_nodes. Topological order: parent precedes children.
    pub order: Vec<u32>,
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

    /// Root node index (always 0 by construction).
    #[inline]
    pub fn root(&self) -> usize { 0 }

    /// Representative pixel of the root node.
    #[inline]
    pub fn root_pixel(&self) -> usize { self.seed_pixel[0] as usize }

    /// Convert a node id to its (x, y) coordinates (from its seed pixel).
    #[inline]
    pub fn node_xy(&self, node: usize) -> (usize, usize) {
        let p = self.seed_pixel[node] as usize;
        (p % self.width, p / self.width)
    }

    /// Whether a pixel is the seed (canonical) of its node.
    #[inline]
    pub fn is_node_seed(&self, p: usize) -> bool {
        self.seed_pixel[self.node_of_pixel[p] as usize] as usize == p
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

    /// Area attribute (Higra `attribute_area`): sum of subtree leaf counts.
    pub fn attribute_area(&self) -> Vec<u32> {
        let n = self.num_nodes();
        let mut area = vec![1u32; n];
        for &nd in self.order.iter().rev() {
            let nd = nd as usize;
            if nd == self.root() { continue; }
            let par = self.parent_node[nd] as usize;
            area[par] += area[nd];
        }
        area
    }

    /// Volume attribute (Higra `attribute_volume`):
    ///   V(n) = area(n) * |level(n) - level(parent(n))| + sum_{c in children} V(c)
    pub fn attribute_volume(&self, area: &[u32]) -> Vec<f64> {
        let n = self.num_nodes();
        let mut vol = vec![0.0f64; n];
        for &nd in self.order.iter().rev() {
            let nd = nd as usize;
            if nd == self.root() { continue; }
            let par = self.parent_node[nd] as usize;
            let h = (self.level[nd] as f64 - self.level[par] as f64).abs();
            vol[nd] += area[nd] as f64 * h;
            vol[par] += vol[nd];
        }
        vol
    }

    /// Topological height (Higra `attribute_topological_height`): longest
    /// downward distance from the root, in edges.
    pub fn attribute_gamma(&self) -> Vec<u32> {
        let n = self.num_nodes();
        let mut g = vec![0u32; n];
        for &nd in &self.order {
            let nd = nd as usize;
            if nd == self.root() { continue; }
            g[nd] = g[self.parent_node[nd] as usize] + 1;
        }
        g
    }

    /// Generic bottom-up accumulator: propagate per-node values from children
    /// to parents. `values` must be length `num_nodes()`. Returns in-place.
    ///
    /// Clones each child value *before* reducing into the parent, so borrow
    /// checker stays happy without needing `split_at_mut`.
    pub fn accumulate<T, F>(&self, mut values: Vec<T>, mut reduce: F) -> Vec<T>
    where
        T: Clone,
        F: FnMut(&mut T, &T),
    {
        assert_eq!(values.len(), self.num_nodes());
        for &nd in self.order.iter().rev() {
            let nd = nd as usize;
            if nd == self.root() { continue; }
            let par = self.parent_node[nd] as usize;
            let v = values[nd].clone();
            reduce(&mut values[par], &v);
        }
        values
    }
}

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

/// Build the max-tree of `img` (row-major, width*height).
/// NaNs must be replaced first (see `replace_nan`).
pub fn build_max_tree(
    img: &[f32],
    width: usize,
    height: usize,
    conn: Connectivity,
) -> MaxTree {
    let n = width * height;
    assert_eq!(img.len(), n);
    assert!(n > 0 && n <= u32::MAX as usize, "image dimension out of range");

    // 1. Stable sort by value. order[0] is the darkest pixel.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by(|&a, &b| img[a as usize].total_cmp(&img[b as usize]));

    // 2. Union-find construction (Berger 2007), process brightest -> darkest.
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

    // 3. Canonicalize: parent pointer lands on a canonical element.
    for &pi in &order {
        let p = pi as usize;
        let q = parent_pix[p] as usize;
        let pq = parent_pix[q] as usize;
        if img[pq] == img[q] {
            parent_pix[p] = pq as u32;
        }
    }

    // 4. Assign node ids in ascending-value order.
    //    A pixel is a node iff it is its own parent OR its parent has a
    //    strictly different value.
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
            debug_assert!(parent_id <= id, "parent must have smaller id");
            parent_node.push(parent_id);
            level.push(img[p]);
            seed_pixel.push(p as u32);
            node_order.push(id);
        }
    }
    // Fill node_of_pixel for non-seed pixels.
    for &pi in &order {
        let p = pi as usize;
        if node_of_pixel[p] == UNDEF {
            let par = parent_pix[p] as usize;
            node_of_pixel[p] = node_of_pixel[par];
        }
    }

    MaxTree {
        width,
        height,
        node_of_pixel,
        parent_node,
        level,
        seed_pixel,
        order: node_order,
    }
}

/// Replace non-finite values with the minimum finite value.
pub fn replace_nan(img: &mut [f32]) -> usize {
    let min = img.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
    let fill = if min.is_finite() { min } else { 0.0 };
    let mut count = 0;
    for v in img.iter_mut() {
        if !v.is_finite() { *v = fill; count += 1; }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<f32> {
        vec![
            1.0, 1.0, 1.0,
            1.0, 5.0, 1.0,
            1.0, 1.0, 7.0,
        ]
    }

    fn node_of(t: &MaxTree, pixel: usize) -> usize {
        t.node_of_pixel[pixel] as usize
    }

    #[test]
    fn four_connectivity_gives_separate_peaks() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Four);
        let s = t.stats();
        // root(1), {5}, {7}
        assert_eq!(s.nodes, 3);
        assert_eq!(s.leaves, 2);
        assert_eq!(s.max_depth, 1);
    }

    #[test]
    fn eight_connectivity_nests_peaks() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Eight);
        let s = t.stats();
        // root(1) -> {5,7} at level 5 -> {7}
        assert_eq!(s.nodes, 3);
        assert_eq!(s.leaves, 1);
        assert_eq!(s.max_depth, 2);
    }

    #[test]
    fn constant_image_is_a_single_node() {
        let img = vec![3.0; 12];
        let t = build_max_tree(&img, 4, 3, Connectivity::Four);
        let s = t.stats();
        assert_eq!(s.nodes, 1);
        assert_eq!(s.leaves, 1);
        assert_eq!(t.root(), 0);
    }

    #[test]
    fn parents_are_valid_and_ordered() {
        let img = vec![
            0.0, 2.0, 2.0, 0.0,
            1.0, 3.0, 2.0, 1.0,
            0.0, 1.0, 1.0, 4.0,
        ];
        let t = build_max_tree(&img, 4, 3, Connectivity::Four);
        let n = t.num_nodes();
        // Parent id strictly smaller (topological order is ascending id).
        for nd in 1..n {
            let par = t.parent_node[nd] as usize;
            assert!(par < nd, "node {nd}: parent {par} not < {nd}");
            assert!(t.level[par] <= t.level[nd]);
        }
        assert_eq!(t.parent_node[0] as usize, 0);
        // node_of_pixel maps to valid ids and seeds are canonical.
        for p in 0..img.len() {
            let nd = node_of(t, p);
            assert!(nd < n);
        }
        for nd in 0..n {
            assert!(t.is_node_seed(t.seed_pixel[nd] as usize));
        }
    }

    #[test]
    fn area_and_volume_four_connectivity() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Four);
        let area = t.attribute_area();
        let vol = t.attribute_volume(&area);
        let n5 = node_of(&t, 4);
        let n7 = node_of(&t, 8);
        assert_eq!(area[t.root()], 9);
        assert_eq!(area[n5], 1);
        assert_eq!(area[n7], 1);
        assert_eq!(vol[n5], 4.0); // 1 * |5 - 1|
        assert_eq!(vol[n7], 6.0); // 1 * |7 - 1|
        assert_eq!(vol[t.root()], 10.0);
    }

    #[test]
    fn area_and_volume_eight_connectivity() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Eight);
        let area = t.attribute_area();
        let vol = t.attribute_volume(&area);
        let n5 = node_of(&t, 4);
        let n7 = node_of(&t, 8);
        assert_eq!(area[t.root()], 9);
        assert_eq!(area[n5], 2);
        assert_eq!(area[n7], 1);
        assert_eq!(vol[n7], 2.0);           // 1 * |7 - 5|
        assert_eq!(vol[n5], 2.0 * 4.0 + 2.0); // 2 * |5 - 1| + V(n7)
        assert_eq!(vol[t.root()], 10.0);
    }

    #[test]
    fn root_volume_telescopes() {
        let img = vec![
            0.0, 2.0, 2.0, 0.0,
            1.0, 3.0, 2.0, 1.0,
            0.0, 1.0, 1.0, 4.0,
        ];
        for conn in [Connectivity::Four, Connectivity::Eight] {
            let t = build_max_tree(&img, 4, 3, conn);
            let area = t.attribute_area();
            let vol = t.attribute_volume(&area);
            assert_eq!(area[t.root()], 12);
            let min = t.level[t.root()] as f64;
            let expected: f64 = img.iter().map(|&v| v as f64 - min).sum();
            assert!((vol[t.root()] - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn gamma_matches_depth() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Eight);
        let g = t.attribute_gamma();
        assert_eq!(g[t.root()], 0);
        assert_eq!(g[node_of(&t, 4)], 1);
        assert_eq!(g[node_of(&t, 8)], 2);
    }

    #[test]
    fn accumulate_sums_area() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Four);
        let leaves: Vec<u32> = (0..t.num_nodes())
            .map(|nd| if t.seed_pixel[nd] as usize == t.seed_pixel[nd] as usize
                        && !(0..t.num_nodes()).any(|m| t.parent_node[m] as usize == nd)
                      { 1 } else { 0 })
            .collect();
        let acc = t.accumulate(leaves, |parent, child| *parent += *child);
        // Accumulated leaf count equals area (each node contributes its leaf set).
        let area = t.attribute_area();
        for nd in 0..t.num_nodes() {
            assert_eq!(acc[nd], area[nd]);
        }
    }

    #[test]
    fn nan_replacement() {
        let mut img = vec![f32::NAN, 2.0, 1.0, f32::INFINITY];
        assert_eq!(replace_nan(&mut img), 2);
        assert_eq!(img, vec![1.0, 2.0, 1.0, 1.0]);
    }
}

