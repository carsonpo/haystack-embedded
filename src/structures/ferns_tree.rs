pub mod k_modes;
pub mod node;
pub mod storage;

use node::Node;
use storage::StorageManager;

use crate::errors::HaystackError;

use self::k_modes::find_modes;
use self::node::Vector;
use crate::math::hamming_distance;

use super::filters::{Filter, Filters};
use super::metadata_index::KVPair;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet, VecDeque};

pub struct Tree {
    pub storage_manager: StorageManager,
    pub max_depth: usize,
}

impl Tree {
    pub fn new() -> Result<Self, HaystackError> {
        let storage_manager = StorageManager::new();
        Ok(Tree {
            storage_manager,
            max_depth: 0,
        })
    }

    pub fn insert(&mut self, vector: Vector, id: u128, metadata: Vec<KVPair>) {
        if self.storage_manager.root_offset() == -1 {
            let mut root = Node::new(vector, id, metadata);
            root.is_root = true;
            root.is_red = false;
            self.storage_manager.store_node(&mut root);
            self.storage_manager.set_root_offset(root.offset);

            return;
        }

        let mut new_node = Node::new(vector, id, metadata);
        self.storage_manager.store_node(&mut new_node);
        let mut root = self
            .storage_manager
            .load_node(self.storage_manager.root_offset())
            .unwrap();

        let _ = self.inner_insert(&mut new_node, &mut root);

        // self.storage_manager.store_node(&mut out);
        self.storage_manager.store_node(&mut root);
    }

    fn inner_insert(&mut self, new_node: &mut Node, node: &mut Node) {
        if node.left.is_none() {
            node.left = Some(new_node.offset);
            new_node.parent = Some(node.offset);
            new_node.depth = node.depth + 1;
            self.storage_manager.store_node(node);
            self.storage_manager.store_node(new_node);
            self.max_depth = self.max_depth.max(new_node.depth);

            return;
        } else if node.right.is_none() {
            node.right = Some(new_node.offset);

            new_node.parent = Some(node.offset);
            new_node.depth = node.depth + 1;
            self.storage_manager.store_node(node);
            self.storage_manager.store_node(new_node);
            self.max_depth = self.max_depth.max(new_node.depth);

            return;
        }

        let left_node = self.storage_manager.load_node(node.left.unwrap()).unwrap();
        let right_node = self.storage_manager.load_node(node.right.unwrap()).unwrap();

        let (choose_left, _) =
            self.compare(&new_node.vector, &left_node.vector, &right_node.vector);

        if choose_left {
            return self.inner_insert(new_node, &mut left_node.clone());
        } else {
            return self.inner_insert(new_node, &mut right_node.clone());
        }
    }

    pub fn search(&mut self, query: Vector, k: usize, filters: Filter) -> Vec<(u128, u16)> {
        let mut best_distance = u16::MAX;

        let mut queue = VecDeque::new();

        let root = self
            .storage_manager
            .load_node(self.storage_manager.root_offset())
            .unwrap();

        queue.push_back((root, u16::MAX));

        let mut results = Vec::new();

        let mut num_visited = 0;

        while !queue.is_empty() {
            let (node, distance) = queue.pop_front().unwrap();

            // println!("Visiting node: {:?}", num_visited);

            num_visited += 1;

            // if !node.is_root {
            let distance = hamming_distance(&node.vector, &query);
            let worst_best_distance = results.last().map(|(_, d)| *d).unwrap_or(u16::MAX);
            if distance < worst_best_distance {
                results.push((node.id, distance));
                results.sort_by_key(|(_, d)| *d);
                results.truncate(k);
            }
            // }

            if !node.left.is_none() && !node.right.is_none() {
                let left_node = self.storage_manager.load_node(node.left.unwrap()).unwrap();
                let right_node = self.storage_manager.load_node(node.right.unwrap()).unwrap();

                let (choose_left, can_prune) =
                    self.compare(&query, &left_node.vector, &right_node.vector);

                if choose_left {
                    queue.push_back((left_node, distance));
                    if !can_prune {
                        queue.push_back((right_node, distance));
                    }
                } else {
                    queue.push_back((right_node, distance));
                    if !can_prune {
                        queue.push_back((left_node, distance));
                    }
                }
            } else {
                if !node.left.is_none() {
                    let left_node = self.storage_manager.load_node(node.left.unwrap()).unwrap();
                    queue.push_back((left_node, distance));
                } else if !node.right.is_none() {
                    let right_node = self.storage_manager.load_node(node.right.unwrap()).unwrap();
                    queue.push_back((right_node, distance));
                }
            }
        }

        results
    }

    fn compare(&self, query: &Vector, v1: &Vector, v2: &Vector) -> (bool, bool) {
        let v1v2_dist = hamming_distance(v1, v2);

        if v1v2_dist == 0 {
            return (false, true);
        }

        let v1_query_dist = hamming_distance(v1, query);
        let v2_query_dist = hamming_distance(v2, query);

        let choose_v1 = v1_query_dist < v2_query_dist;
        let can_prune =
            v1v2_dist + v1_query_dist < v2_query_dist || v1v2_dist + v2_query_dist < v1_query_dist;

        (choose_v1, false)
    }
}
