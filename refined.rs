//! Segmentation from the refined (`move_up`) segments, as in MTO2.
//!
//! `corrected[nd]` marks the nodes returned by `statistical_tests::move_up`.
//! Each pixel is labelled by the nearest corrected node on the path from its
//! max-tree node to the root (inclusive). Pixels with no corrected ancestor
//! are background. This works on the full max-tree, like Python's
//! `propagate_sequential`, not on the contracted tree.

use crate::maxtree::MaxTree;
use crate::segment::Segmentation;

pub fn build_segmentation_refined(tree: &MaxTree, corrected: &[bool]) -> Segmentation {
    let n_pix = tree.num_pixels();
    let n_nodes = tree.num_nodes();
    let root = tree.root();

    // One segment id per corrected node (root is never a segment).
    let mut node_segment = vec![u32::MAX; n_nodes];
    let mut segment_seed: Vec<u32> = Vec::new();
    for &nd in &tree.order {
        let nd = nd as usize;
        if corrected[nd] && nd != root {
            node_segment[nd] = segment_seed.len() as u32;
            segment_seed.push(nd as u32);
        }
    }

    // Propagate labels root -> leaves (`order` lists parents before children).
    let mut label = vec![u32::MAX; n_nodes];
    for &nd in &tree.order {
        let nd = nd as usize;
        label[nd] = if node_segment[nd] != u32::MAX {
            node_segment[nd]
        } else if nd == root {
            u32::MAX
        } else {
            label[tree.parent_node[nd] as usize]
        };
    }

    let mut pixel_segment = vec![u32::MAX; n_pix];
    let mut segment_area = vec![0u32; segment_seed.len()];
    for p in 0..n_pix {
        let s = label[tree.node_of_pixel[p] as usize];
        pixel_segment[p] = s;
        if s != u32::MAX {
            segment_area[s as usize] += 1;
        }
    }

    Segmentation {
        segment_of_pixel: pixel_segment,
        segment_seed,
        segment_area,
    }
}