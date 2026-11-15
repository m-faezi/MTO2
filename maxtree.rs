//! Node-oriented max-tree for a 2D f32 image.

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
    pub node_of_pixel: Vec<u32>,
    pub parent_node: Vec<u32>,
    pub level: Vec<f32>,
    pub seed_pixel: Vec<u32>,
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

pub fn replace_nan(img: &mut [f32]) -> usize {
    let min = img.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
    let fill = if min.is_finite() { min } else { 0.0 };
    let mut count = 0;
    for v in img.iter_mut() {
        if !v.is_finite() { *v = fill; count += 1; }
    }
    count
}

/// Round pixel values to integer ADU levels.
///
/// WARNING: only use this if your image is already integer-quantized ADU.
/// For continuous-valued (calibrated) images this collapses the tree to a
/// handful of nodes and destroys the statistical test. MTO2 does *not*
/// quantize reduced/calibrated images — it only works on the spatially
/// reduced image (different concept). We keep the function but no longer
/// call it from `main`.
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
    fn root_volume_telescopes() {
        let img = vec![
            0.0, 2.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 1.0, 4.0,
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
    fn nan_replacement() {
        let mut img = vec![f32::NAN, 2.0, 1.0, f32::INFINITY];
        assert_eq!(replace_nan(&mut img), 2);
        assert_eq!(img, vec![1.0, 2.0, 1.0, 1.0]);
    }
}