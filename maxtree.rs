//! Max-tree construction for a 2D f32 image.
//!
//! Algorithm: union-find construction (Berger et al., 2007) followed by
//! canonicalization, so every node of the tree is represented by exactly one
//! pixel (the "canonical element" of its connected component at that level).
//!
//! Conventions:
//!   * The image is row-major, pixel index p = y * width + x.
//!   * The root is the pixel with the lowest value (first in `order`).
//!   * `parent[p]` is the parent pixel of p. After canonicalization,
//!     `parent[p]` always points at a canonical element (or the root).
//!   * A pixel p is a canonical element (i.e. a real node) iff
//!     p is the root, or img[parent[p]] != img[p].

const UNDEF: usize = usize::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    Four,
    Eight,
}

impl Connectivity {
    fn offsets(self) -> &'static [(isize, isize)] {
        match self {
            Connectivity::Four => &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            Connectivity::Eight => &[
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
                (1, 1),
            ],
        }
    }
}

pub struct MaxTree {
    pub width: usize,
    pub height: usize,
    /// Parent pixel of every pixel (root is its own parent).
    pub parent: Vec<usize>,
    /// Pixels sorted by increasing value. `order[0]` is the root, and every
    /// parent appears before its children, so iterating forward is a
    /// top-down (root-to-leaves) traversal and in reverse is bottom-up.
    pub order: Vec<usize>,
}

pub struct TreeStats {
    pub nodes: usize,
    pub leaves: usize,
    pub max_depth: usize,
}

impl MaxTree {
    pub fn root(&self) -> usize {
        self.order[0]
    }

    /// True if pixel p is the canonical element of a node.
    pub fn is_node(&self, img: &[f32], p: usize) -> bool {
        p == self.root() || img[self.parent[p]] != img[p]
    }

    /// Node count, leaf count and depth (root has depth 0).
    pub fn stats(&self, img: &[f32]) -> TreeStats {
        let n = self.parent.len();
        let mut has_child = vec![false; n];
        let mut depth = vec![0usize; n];
        let mut nodes = 0;
        let mut max_depth = 0;

        // Top-down: parents are visited before children.
        for &p in &self.order {
            if p == self.root() {
                nodes += 1;
                depth[p] = 0;
            } else if self.is_node(img, p) {
                nodes += 1;
                has_child[self.parent[p]] = true;
                depth[p] = depth[self.parent[p]] + 1;
                max_depth = max_depth.max(depth[p]);
            } else {
                // Non-canonical pixel: inherits depth of its node.
                depth[p] = depth[self.parent[p]];
            }
        }

        let leaves = (0..n)
            .filter(|&p| self.is_node(img, p) && !has_child[p])
            .count();

        TreeStats {
            nodes,
            leaves,
            max_depth,
        }
    }

    /// Area attribute (Higra's `attribute_area`): the number of pixels in the
    /// connected component represented by each node, i.e. the sum of the
    /// areas of the leaves of its subtree (every pixel has area 1).
    ///
    /// The result is indexed by pixel but is only meaningful at node pixels
    /// (`is_node`); the root's area is the total number of pixels.
    pub fn attribute_area(&self) -> Vec<u32> {
        let n = self.parent.len();
        assert!(n <= u32::MAX as usize, "image too large for u32 areas");
        let mut area = vec![1u32; n];
        // Bottom-up: children come after their parent in `order`.
        for &p in self.order.iter().rev() {
            if p != self.root() {
                area[self.parent[p]] += area[p];
            }
        }
        area
    }

    /// Volume attribute (Higra's `attribute_volume`), with altitudes taken
    /// as the pixel values:
    ///
    ///   V(n) = area(n) * |alt(n) - alt(parent(n))| + sum_{c in children(n)} V(c)
    ///
    /// The root is its own parent, so its own term is 0. Only meaningful at
    /// node pixels, like `attribute_area`. Pixels that share a node's level
    /// are leaves in Higra's tree and contribute 0 (same altitude as their
    /// parent), so this matches Higra's values on the compact tree.
    pub fn attribute_volume(&self, img: &[f32], area: &[u32]) -> Vec<f64> {
        let n = self.parent.len();
        let mut volume = vec![0.0f64; n];
        for &p in self.order.iter().rev() {
            if p == self.root() {
                continue;
            }
            let q = self.parent[p];
            if self.is_node(img, p) {
                let height = (img[p] as f64 - img[q] as f64).abs();
                volume[p] += area[p] as f64 * height;
            }
            // Non-node pixels have volume 0, so adding is harmless.
            volume[q] += volume[p];
        }
        volume
    }
}

fn find_root(zpar: &mut [usize], mut x: usize) -> usize {
    let mut root = x;
    while zpar[root] != root {
        root = zpar[root];
    }
    // Path compression.
    while zpar[x] != root {
        let next = zpar[x];
        zpar[x] = root;
        x = next;
    }
    root
}

/// Build the max-tree of `img` (row-major, `width * height` values).
///
/// NaN values must be removed beforehand (see `replace_nan`).
pub fn build_max_tree(
    img: &[f32],
    width: usize,
    height: usize,
    conn: Connectivity,
) -> MaxTree {
    let n = width * height;
    assert_eq!(img.len(), n, "image size does not match width * height");
    assert!(n > 0, "empty image");

    // Stable sort by increasing value.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| img[a].total_cmp(&img[b]));

    let mut parent = vec![UNDEF; n];
    let mut zpar = vec![UNDEF; n];
    let offsets = conn.offsets();

    // Process from the brightest pixel to the darkest.
    for &p in order.iter().rev() {
        parent[p] = p;
        zpar[p] = p;
        let (x, y) = ((p % width) as isize, (p / width) as isize);

        for &(dx, dy) in offsets {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let q = ny as usize * width + nx as usize;
            if parent[q] == UNDEF {
                continue; // neighbour not processed yet (darker)
            }
            let r = find_root(&mut zpar, q);
            if r != p {
                parent[r] = p;
                zpar[r] = p;
            }
        }
    }

    // Canonicalize: make every parent pointer land on a canonical element.
    for &p in order.iter() {
        let q = parent[p];
        let pq = parent[q];
        if img[pq] == img[q] {
            parent[p] = pq;
        }
    }

    MaxTree {
        width,
        height,
        parent,
        order,
    }
}

/// Replace non-finite values with the minimum finite value in the image,
/// so they end up in the background. Returns how many were replaced.
pub fn replace_nan(img: &mut [f32]) -> usize {
    let min = img
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let fill = if min.is_finite() { min } else { 0.0 };
    let mut count = 0;
    for v in img.iter_mut() {
        if !v.is_finite() {
            *v = fill;
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    // 3x3 image: background 1, a peak of 5 in the centre, a peak of 7 in the
    // bottom-right corner (diagonal to the centre).
    fn sample() -> Vec<f32> {
        vec![
            1.0, 1.0, 1.0, //
            1.0, 5.0, 1.0, //
            1.0, 1.0, 7.0,
        ]
    }

    #[test]
    fn four_connectivity_gives_separate_peaks() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Four);
        let s = t.stats(&img);
        // root(1), {5}, {7}
        assert_eq!(s.nodes, 3);
        assert_eq!(s.leaves, 2);
        assert_eq!(s.max_depth, 1);
    }

    #[test]
    fn eight_connectivity_nests_peaks() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Eight);
        let s = t.stats(&img);
        // root(1) -> {5,7} at level 5 -> {7}
        assert_eq!(s.nodes, 3);
        assert_eq!(s.leaves, 1);
        assert_eq!(s.max_depth, 2);
    }

    #[test]
    fn constant_image_is_a_single_node() {
        let img = vec![3.0; 12];
        let t = build_max_tree(&img, 4, 3, Connectivity::Four);
        let s = t.stats(&img);
        assert_eq!(s.nodes, 1);
        assert_eq!(s.leaves, 1);
    }

    #[test]
    fn parents_are_canonical_and_ordered() {
        let img = vec![
            0.0, 2.0, 2.0, 0.0, //
            1.0, 3.0, 2.0, 1.0, //
            0.0, 1.0, 1.0, 4.0,
        ];
        let t = build_max_tree(&img, 4, 3, Connectivity::Four);
        let mut pos = vec![0usize; img.len()];
        for (i, &p) in t.order.iter().enumerate() {
            pos[p] = i;
        }
        for p in 0..img.len() {
            if p == t.root() {
                assert_eq!(t.parent[p], p);
                continue;
            }
            let q = t.parent[p];
            assert!(img[q] <= img[p]);
            assert!(pos[q] < pos[p], "parent must precede child in order");
            // Parent must itself be a node.
            assert!(t.is_node(&img, q));
        }
    }

    #[test]
    fn area_and_volume_four_connectivity() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Four);
        let area = t.attribute_area();
        let vol = t.attribute_volume(&img, &area);
        let peak5 = 4; // centre pixel
        let peak7 = 8; // bottom-right pixel
        assert_eq!(area[t.root()], 9);
        assert_eq!(area[peak5], 1);
        assert_eq!(area[peak7], 1);
        assert_eq!(vol[peak5], 4.0); // 1 * |5 - 1|
        assert_eq!(vol[peak7], 6.0); // 1 * |7 - 1|
        assert_eq!(vol[t.root()], 10.0);
    }

    #[test]
    fn area_and_volume_eight_connectivity() {
        let img = sample();
        let t = build_max_tree(&img, 3, 3, Connectivity::Eight);
        let area = t.attribute_area();
        let vol = t.attribute_volume(&img, &area);
        let n5 = 4; // node at level 5 contains {5, 7}
        let n7 = 8;
        assert_eq!(area[t.root()], 9);
        assert_eq!(area[n5], 2);
        assert_eq!(area[n7], 1);
        assert_eq!(vol[n7], 2.0); // 1 * |7 - 5|
        assert_eq!(vol[n5], 2.0 * 4.0 + 2.0); // 2 * |5 - 1| + V(n7)
        assert_eq!(vol[t.root()], 10.0);
    }

    #[test]
    fn root_volume_telescopes() {
        // V(root) must equal sum over pixels of (value - minimum).
        let img = vec![
            0.0, 2.0, 2.0, 0.0, //
            1.0, 3.0, 2.0, 1.0, //
            0.0, 1.0, 1.0, 4.0,
        ];
        for conn in [Connectivity::Four, Connectivity::Eight] {
            let t = build_max_tree(&img, 4, 3, conn);
            let area = t.attribute_area();
            let vol = t.attribute_volume(&img, &area);
            assert_eq!(area[t.root()], 12);
            let expected: f64 = img.iter().map(|&v| v as f64).sum();
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

