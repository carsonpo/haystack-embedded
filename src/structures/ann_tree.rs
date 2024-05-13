pub mod k_modes;
pub mod node;
pub mod storage;

use node::{Node, NodeType};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use storage::StorageManager;

use crate::errors::HaystackError;

use self::k_modes::find_modes;
use self::node::Vector;
use crate::math::hamming_distance;

use super::filters::{Filter, Filters};
use super::metadata_index::KVPair;

use rayon::prelude::*;

use std::collections::{BinaryHeap, HashSet};

pub struct Tree {
    pub k: usize,
    pub storage_manager: storage::StorageManager,
}

#[derive(Eq, PartialEq)]
struct PathNode {
    distance: u16,
    offset: i64,
}

// Implement `Ord` and `PartialOrd` for `PathNode` to use it in a min-heap
impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.cmp(&self.distance) // Reverse order for min-heap
    }
}

impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Tree {
    pub fn new() -> Result<Self, HaystackError> {
        let mut storage_manager = StorageManager::new();

        // println!("INIT Used space: {}", storage_manager.used_space);

        let mut root = Node::new_leaf();
        root.is_root = true;

        storage_manager.store_node(&mut root)?;
        storage_manager.set_root_offset(root.offset);

        Ok(Tree {
            storage_manager,
            k: crate::constants::K,
        })
    }

    pub fn batch_insert(
        &mut self,
        vectors: Vec<Vector>,
        ids: Vec<u128>,
        metadata: Vec<Vec<KVPair>>,
    ) {
        for ((vector, id), metadata) in vectors.iter().zip(ids.iter()).zip(metadata.iter()) {
            self.insert(vector.clone(), *id, metadata.clone());
        }
    }

    pub fn insert(&mut self, vector: Vector, id: u128, metadata: Vec<KVPair>) {
        let entrypoint = self.find_entrypoint(vector);
        let mut node = self.storage_manager.load_node(entrypoint).unwrap();

        if node.is_full() {
            let mut siblings = node.split().expect("Failed to split node");
            let sibling_offsets: Vec<i64> = siblings
                .iter_mut()
                .map(|sibling| {
                    sibling.parent_offset = node.parent_offset; // Set parent offset before storing
                    self.storage_manager.store_node(sibling).unwrap()
                })
                .collect();

            for sibling in siblings.clone() {
                if sibling.node_type == NodeType::Internal
                    && sibling.children.len() != sibling.vectors.len()
                {
                    panic!("Internal node has different number of children and vectors");
                }
            }

            if node.is_root {
                let mut new_root = Node::new_internal();
                new_root.is_root = true;
                new_root.children.push(node.offset);
                new_root.vectors.push(find_modes(node.vectors.clone()));
                for sibling_offset in &sibling_offsets {
                    let sibling = self.storage_manager.load_node(*sibling_offset).unwrap();
                    new_root.vectors.push(find_modes(sibling.vectors));
                    new_root.children.push(*sibling_offset);
                }
                self.storage_manager.store_node(&mut new_root).unwrap();
                self.storage_manager.set_root_offset(new_root.offset);
                node.is_root = false;
                node.parent_offset = Some(new_root.offset);
                siblings
                    .iter_mut()
                    .for_each(|sibling| sibling.parent_offset = Some(new_root.offset));
                self.storage_manager.store_node(&mut node).unwrap();
                siblings.iter_mut().for_each(|sibling| {
                    if sibling.node_type == NodeType::Internal
                        && sibling.children.len() != sibling.vectors.len()
                    {
                        panic!("Internal node has different number of children and vectors v3");
                    }
                    self.storage_manager.store_node(sibling).unwrap();
                });
            } else {
                let parent_offset = node.parent_offset.unwrap();
                let mut parent = self.storage_manager.load_node(parent_offset).unwrap();
                parent.children.push(node.offset);
                parent.vectors.push(find_modes(node.vectors.clone()));
                sibling_offsets
                    .iter()
                    .for_each(|&offset| parent.children.push(offset));
                siblings
                    .iter()
                    .for_each(|sibling| parent.vectors.push(find_modes(sibling.vectors.clone())));
                if parent.node_type == NodeType::Internal
                    && parent.children.len() != parent.vectors.len()
                {
                    println!("Parent vectors: {:?}", parent.vectors.len());
                    println!("Parent children: {:?}", parent.children);
                    println!("Sibling offsets: {:?}", sibling_offsets.len());

                    panic!("parent node has different number of children and vectors");
                }
                self.storage_manager.store_node(&mut parent).unwrap();
                node.parent_offset = Some(parent_offset);
                self.storage_manager.store_node(&mut node).unwrap();
                siblings.into_iter().for_each(|mut sibling| {
                    if sibling.node_type == NodeType::Internal
                        && sibling.children.len() != sibling.vectors.len()
                    {
                        panic!("Internal node has different number of children and vectors v3");
                    }
                    sibling.parent_offset = Some(parent_offset);
                    self.storage_manager.store_node(&mut sibling).unwrap();
                });

                let mut current_node = parent;
                while current_node.is_full() {
                    let mut siblings = current_node.split().expect("Failed to split node");
                    let sibling_offsets: Vec<i64> = siblings
                        .iter_mut()
                        .map(|sibling| {
                            sibling.parent_offset = Some(current_node.parent_offset.unwrap());
                            self.storage_manager.store_node(sibling).unwrap()
                        })
                        .collect();

                    for sibling in siblings.clone() {
                        if sibling.node_type == NodeType::Internal
                            && sibling.children.len() != sibling.vectors.len()
                        {
                            panic!("Internal node has different number of children and vectors v2");
                        }
                    }

                    if current_node.is_root {
                        let mut new_root = Node::new_internal();
                        new_root.is_root = true;
                        new_root.children.push(current_node.offset);
                        new_root.children.extend(sibling_offsets.clone());
                        new_root
                            .vectors
                            .push(find_modes(current_node.vectors.clone()));
                        siblings.iter().for_each(|sibling| {
                            new_root.vectors.push(find_modes(sibling.vectors.clone()))
                        });
                        self.storage_manager.store_node(&mut new_root).unwrap();
                        self.storage_manager.set_root_offset(new_root.offset);
                        current_node.is_root = false;
                        current_node.parent_offset = Some(new_root.offset);
                        siblings
                            .iter_mut()
                            .for_each(|sibling| sibling.parent_offset = Some(new_root.offset));
                        self.storage_manager.store_node(&mut current_node).unwrap();
                        siblings.into_iter().for_each(|mut sibling| {
                            if sibling.node_type == NodeType::Internal
                                && sibling.children.len() != sibling.vectors.len()
                            {
                                panic!(
                                    "Internal node has different number of children and vectors v4"
                                );
                            }
                            self.storage_manager.store_node(&mut sibling).unwrap();
                        });
                    } else {
                        let parent_offset = current_node.parent_offset.unwrap();
                        let mut parent = self.storage_manager.load_node(parent_offset).unwrap();
                        parent.children.push(current_node.offset);
                        sibling_offsets
                            .iter()
                            .for_each(|&offset| parent.children.push(offset));
                        parent
                            .vectors
                            .push(find_modes(current_node.vectors.clone()));
                        siblings.iter().for_each(|sibling| {
                            if sibling.node_type == NodeType::Internal
                                && sibling.children.len() != sibling.vectors.len()
                            {
                                panic!(
                                    "Internal node has different number of children and vectors v5"
                                );
                            }
                            parent.vectors.push(find_modes(sibling.vectors.clone()))
                        });
                        self.storage_manager.store_node(&mut parent).unwrap();
                        current_node.parent_offset = Some(parent_offset);
                        self.storage_manager.store_node(&mut current_node).unwrap();
                        siblings.into_iter().for_each(|mut sibling| {
                            sibling.parent_offset = Some(parent_offset);
                            self.storage_manager.store_node(&mut sibling).unwrap();
                        });
                        current_node = parent;
                    }
                }
            }
        } else {
            if node.node_type != NodeType::Leaf {
                panic!("Entrypoint is not a leaf node");
            }
            node.vectors.push(vector);
            node.ids.push(id);
            node.metadata.push(metadata);
            self.storage_manager.store_node(&mut node).unwrap();
        }
    }

    fn find_entrypoint(&mut self, vector: Vector) -> i64 {
        const C: usize = crate::constants::C; // Adjust C as per your constant definition

        let mut node = self
            .storage_manager
            .load_node(self.storage_manager.root_offset())
            .unwrap();

        while node.node_type == NodeType::Internal {
            let mut distances: Vec<(usize, u16)> = node
                .vectors
                .par_iter()
                .map(|key| hamming_distance(&vector, key))
                .enumerate()
                .collect();

            if node.vectors.len() != node.children.len() {
                println!("Node vectors: {:?}", node.vectors.len());
                println!("Node children: {:?}", node.children);
                panic!("Internal node has different number of children and vectors");
            }

            // Sort distances and pick the top C candidates
            distances.sort_by_key(|&(_, dist)| dist);
            let closest = distances.iter().take(C).cloned().collect::<Vec<_>>();

            let mut best_child_offset = None;
            let mut best_distance = u16::MAX;

            for (idx, (i, _)) in closest.iter().enumerate() {
                let child_offset = node.children[*i];
                let child_node = self.storage_manager.load_node(child_offset).unwrap();

                let mut child_distances: Vec<(usize, u16)> = child_node
                    .vectors
                    .iter()
                    .map(|key| hamming_distance(&vector, key))
                    .enumerate()
                    .collect();

                // Sort distances and pick the top C candidates in the child node
                child_distances.sort_by_key(|&(_, dist)| dist);
                let child_closest = child_distances.iter().take(C).cloned().collect::<Vec<_>>();

                for (_, distance) in child_closest {
                    if distance < best_distance {
                        best_distance = distance;
                        best_child_offset = Some(child_offset);
                    }
                }
            }

            if let Some(offset) = best_child_offset {
                node = self.storage_manager.load_node(offset).unwrap();
            } else {
                panic!("Failed to find a suitable child node");
            }
        }

        // Now node is a leaf node
        node.offset
    }

    // pub fn search(&mut self, vector: Vector, top_k: usize, filters: Filter) -> Vec<u128> {
    //     const C: usize = crate::constants::C; // Adjust C as per your constant definition
    //     let mut results = Vec::new();
    //     let mut path_heap = Vec::new();
    //     let mut visited = HashSet::new();

    //     // Start by pushing the root node into the path heap
    //     path_heap.push(PathNode {
    //         distance: 0,
    //         offset: self.storage_manager.root_offset(),
    //     });

    //     while let Some(PathNode {
    //         offset: current_offset,
    //         ..
    //     }) = path_heap.pop()
    //     {
    //         if visited.contains(&current_offset) {
    //             continue;
    //         }
    //         visited.insert(current_offset);

    //         let node = self.storage_manager.load_node(current_offset).unwrap();

    //         if node.node_type == NodeType::Internal {
    //             let mut distances: Vec<(usize, u16)> = node
    //                 .vectors
    //                 .iter()
    //                 .map(|key| hamming_distance(&vector, key))
    //                 .enumerate()
    //                 .collect();

    //             if node.vectors.len() != node.children.len() {
    //                 println!("Node vectors: {:?}", node.vectors.len());
    //                 println!("Node children: {:?}", node.children);
    //                 panic!("Internal node has different number of children and vectors");
    //             }

    //             // Sort distances and pick the top C candidates
    //             distances.sort_by_key(|&(_, dist)| dist);
    //             let closest = distances.iter().take(C).cloned().collect::<Vec<_>>();

    //             let mut best_child_nodes = Vec::new();

    //             for (_, (i, _)) in closest.iter().enumerate() {
    //                 let child_offset = node.children[*i];
    //                 let child_node = self.storage_manager.load_node(child_offset).unwrap();

    //                 if child_node.node_type == NodeType::Internal {
    //                     let mut child_distances: Vec<(usize, u16)> = child_node
    //                         .vectors
    //                         .iter()
    //                         .map(|key| hamming_distance(&vector, key))
    //                         .enumerate()
    //                         .collect();

    //                     // Sort distances and pick the top C candidates in the child node
    //                     child_distances.sort_by_key(|&(_, dist)| dist);
    //                     let child_closest =
    //                         child_distances.iter().take(C).cloned().collect::<Vec<_>>();

    //                     if child_node.children.len() != child_node.vectors.len() {
    //                         println!("Child node vectors: {:?}", child_node.vectors.len());
    //                         println!("Child node children: {:?}", child_node.children);
    //                         panic!("Child node has different number of children and vectors");
    //                     }

    //                     for (child_idx, (child_vector_idx, child_distance)) in
    //                         child_closest.iter().enumerate()
    //                     {
    //                         if *child_vector_idx < child_node.children.len() {
    //                             best_child_nodes.push(PathNode {
    //                                 distance: *child_distance,
    //                                 offset: child_node.children[*child_vector_idx],
    //                             });
    //                         } else {
    //                             println!("Child vector idx: {:?}", child_vector_idx);
    //                             println!("Child node children: {:?}", child_node.children.len());
    //                             panic!("Index out of bounds when accessing child_node.children");
    //                         }
    //                     }
    //                 } else {
    //                     // Leaf node: collect candidates
    //                     for (idx, vec) in child_node.vectors.iter().enumerate() {
    //                         let distance = hamming_distance(&vector, vec);
    //                         let id = child_node.ids[idx];
    //                         let metadata = &child_node.metadata[idx];

    //                         if Filters::matches(&filters, metadata) || true {
    //                             results.push((id, distance));
    //                         }
    //                     }

    //                     // Sort and truncate to top_k results
    //                     results.sort_by_key(|&(_, distance)| distance);
    //                     results.truncate(top_k);

    //                     // If we have enough results, stop
    //                     if results.len() >= top_k {
    //                         break;
    //                     }
    //                 }
    //             }

    //             // Add the best child nodes to the heap
    //             for path_node in best_child_nodes {
    //                 path_heap.push(path_node);
    //             }
    //         } else {
    //             // Leaf node: collect candidates
    //             for (idx, vec) in node.vectors.iter().enumerate() {
    //                 let distance = hamming_distance(&vector, vec);
    //                 let id = node.ids[idx];
    //                 let metadata = &node.metadata[idx];

    //                 if Filters::matches(&filters, metadata) || true {
    //                     results.push((id, distance));
    //                 }
    //             }

    //             // Sort and truncate to top_k results
    //             results.sort_by_key(|&(_, distance)| distance);
    //             results.truncate(top_k);

    //             // If we have enough results, stop
    //             if results.len() >= top_k {
    //                 break;
    //             }
    //         }
    //     }

    //     // Extract only the ids from results
    //     results.iter().map(|&(id, _)| id).collect()
    // }

    pub fn search(&mut self, vector: Vector, top_k: usize, filters: Filter) -> Vec<u128> {
        let node = self
            .storage_manager
            .load_node(self.storage_manager.root_offset())
            .unwrap();

        let candidates = self.traverse(&vector, node, top_k, 0);

        // Apply filters to the collected candidates
        let mut filtered_candidates: Vec<(u128, u16)> = candidates
            .into_iter()
            .map(|(id, distance, _)| (id, distance))
            .collect();

        // Sort by distance and truncate to top_k results
        filtered_candidates.sort_by_key(|&(_, distance)| distance);
        filtered_candidates.truncate(top_k);

        filtered_candidates.into_iter().map(|(id, _)| id).collect()
    }

    fn traverse(
        &self,
        vector: &Vector,
        node: Node,
        k: usize,
        depth: usize,
    ) -> Vec<(u128, u16, Vec<KVPair>)> {
        if node.node_type == NodeType::Leaf {
            let top_k_values: Vec<(u128, u16, Vec<KVPair>)> = self
                .collect_top_k(vector, node.vectors, k)
                .par_iter()
                .map(|(idx, distance)| {
                    let id = node.ids[*idx];
                    let metadata = node.metadata[*idx].clone();
                    (id, *distance, metadata)
                })
                .collect();

            return top_k_values;
        }

        let mut alpha = crate::constants::ALPHA >> depth;

        if alpha <= 1 {
            alpha = 1;
        }

        if depth > 2 {
            println!("Depth: {}", depth);
            alpha = 4;
        }

        let best_children: Vec<(usize, Node)> = self
            .collect_top_k(vector, node.vectors, alpha)
            .into_par_iter()
            .map(|(idx, _)| {
                let child_offset = node.children[idx];
                let child_node = self.storage_manager.load_node(child_offset).unwrap();

                (idx, child_node)
            })
            .collect();

        let level_results: Vec<(u128, u16, Vec<KVPair>)> = best_children
            .into_par_iter()
            .map(|(child_idx, child_node)| {
                let child_offset = node.children[child_idx];
                let child_node = self.storage_manager.load_node(child_offset).unwrap();

                self.traverse(vector, child_node, k, depth + 1)
            })
            .flatten()
            .collect();

        // let mut top_k_values = Vec::with_capacity(k);

        // for (id, distance, metadata) in level_results {
        //     if top_k_values.len() < k {
        //         top_k_values.push((id, distance, metadata));
        //     } else {
        //         let worst_best_distance = top_k_values.get(k - 1).unwrap().1;
        //         if distance < worst_best_distance {
        //             top_k_values.pop();
        //             top_k_values.push((id, distance, metadata));
        //             top_k_values.sort_by_key(|(_, distance, _)| *distance);
        //         }
        //     }
        // }

        // top_k_values
        level_results
    }

    fn collect_top_k(&self, query: &Vector, items: Vec<Vector>, k: usize) -> Vec<(usize, u16)> {
        let mut top_k_values = Vec::with_capacity(k);

        for (idx, vec) in items.iter().enumerate() {
            let distance = hamming_distance(vec, query);
            if top_k_values.len() < k {
                top_k_values.push((idx, distance));
            } else {
                let worst_best_distance = top_k_values.get(k - 1).unwrap().1;
                if distance < worst_best_distance {
                    top_k_values.pop();
                    top_k_values.push((idx, distance));
                    top_k_values.sort_by_key(|(_, distance)| *distance);
                }
            }
        }

        top_k_values
    }

    pub fn to_binary(&mut self) -> Vec<u8> {
        let mut serialized = Vec::new();

        serialized.extend((self.k as u64).to_le_bytes().as_ref());
        serialized.extend(self.storage_manager.root_offset().to_le_bytes().as_ref());

        for item in self.storage_manager.data.iter() {
            let serialized_node = item.serialize();
            serialized.extend(serialized_node.len().to_le_bytes().as_ref());
            serialized.extend(serialized_node.as_slice());
        }

        serialized
    }

    pub fn from_binary(data: Vec<u8>) -> Result<Tree, HaystackError> {
        let mut offset = 0;

        let mut tree = Tree::new()?;

        tree.k = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        tree.storage_manager.set_root_offset(i64::from_le_bytes(
            data[offset..offset + 8].try_into().unwrap(),
        ));

        offset += 8;

        while offset < data.len() {
            let node_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;

            let node_data = &data[offset..offset + node_len];
            let node = Node::deserialize(node_data);
            tree.storage_manager
                .data
                .insert(node.offset as usize, node.clone());
            offset += node_len;
        }

        Ok(tree)
    }

    pub fn calibrate(&mut self) -> Result<(), HaystackError> {
        // Step 1: Get all leaf nodes
        let mut leaf_nodes = Vec::new();
        self.collect_leaf_nodes(self.storage_manager.root_offset(), &mut leaf_nodes)?;

        println!("Leaf nodes: {:?}", leaf_nodes.len());

        // Step 2: Build a balanced tree from the leaf nodes
        let root_offset = self.build_balanced_tree(&mut leaf_nodes, 0)?;

        // Step 3: Set the new root offset
        self.storage_manager.set_root_offset(root_offset);

        Ok(())
    }

    fn build_balanced_tree(
        &mut self,
        nodes: &mut [Node],
        depth: usize,
    ) -> Result<i64, HaystackError> {
        if nodes.len() == 1 {
            return Ok(nodes[0].offset);
        }

        let mut parent_nodes = Vec::new();
        let mut current_parent = Node::new_internal();

        for node in nodes.iter_mut() {
            if current_parent.is_full() {
                self.storage_manager.store_node(&mut current_parent)?;
                parent_nodes.push(current_parent);
                current_parent = Node::new_internal();
            }

            node.parent_offset = Some(current_parent.offset);
            current_parent.children.push(node.offset);
            current_parent
                .vectors
                .push(find_modes(node.vectors.clone()));
            self.storage_manager.store_node(node)?;
        }

        if !current_parent.children.is_empty() {
            self.storage_manager.store_node(&mut current_parent)?;
            parent_nodes.push(current_parent);
        }

        if parent_nodes.len() > 1 {
            self.build_balanced_tree(&mut parent_nodes, depth + 1)
        } else {
            let mut root = parent_nodes.pop().unwrap();
            root.is_root = true;
            self.storage_manager.store_node(&mut root)?;
            Ok(root.offset)
        }
    }

    pub fn bulk_insert_and_calibrate(
        &mut self,
        vectors: Vec<Vector>,
        ids: Vec<u128>,
        metadata: Vec<Vec<KVPair>>,
    ) {
        // insert everything into a new root node and then calibrate. this is for speed testing purposes
        let mut root = Node::new_internal();
        root.is_root = true;

        self.storage_manager.store_node(&mut root).unwrap();
        self.storage_manager.set_root_offset(root.offset);

        let mut i = 0;

        let mut current_leaf = Node::new_leaf();

        for ((vector, id), metadata) in vectors.iter().zip(ids.iter()).zip(metadata.iter()) {
            if current_leaf.is_full() {
                let mut new_leaf = Node::new_leaf();
                new_leaf.parent_offset = Some(root.offset);
                self.storage_manager.store_node(&mut current_leaf).unwrap();
                root.children.push(current_leaf.offset);
                root.vectors.push(find_modes(current_leaf.vectors.clone()));
                current_leaf = new_leaf;
            }

            current_leaf.vectors.push(vector.clone());
            current_leaf.ids.push(*id);
            current_leaf.metadata.push(metadata.clone());
        }

        self.storage_manager.store_node(&mut root).unwrap();
        self.storage_manager.set_root_offset(root.offset);

        self.calibrate().unwrap();
    }

    fn collect_leaf_nodes(
        &mut self,
        offset: i64,
        leaf_nodes: &mut Vec<Node>,
    ) -> Result<(), HaystackError> {
        let node = self.storage_manager.load_node(offset)?;
        if node.node_type == NodeType::Leaf {
            leaf_nodes.push(node);
        } else {
            for &child_offset in &node.children {
                self.collect_leaf_nodes(child_offset, leaf_nodes)?;
            }
        }
        Ok(())
    }

    pub fn true_calibrate(&mut self) -> Result<(), HaystackError> {
        // Step 1: Get all leaf nodes
        let mut leaf_nodes = Vec::new();
        self.collect_leaf_nodes(self.storage_manager.root_offset(), &mut leaf_nodes)?;

        println!("Leaf nodes: {:?}", leaf_nodes.len());

        // Step 2: Make a new root
        let mut new_root = Node::new_internal();
        new_root.is_root = true;

        // Step 3: Store the new root to set its offset
        self.storage_manager.store_node(&mut new_root)?;
        self.storage_manager.set_root_offset(new_root.offset);

        // Step 4: Make all the leaf nodes the new root's children, and set all their parent_offsets to the new root's offset
        for leaf_node in &mut leaf_nodes {
            leaf_node.parent_offset = Some(new_root.offset);
            new_root.children.push(leaf_node.offset);
            new_root.vectors.push(find_modes(leaf_node.vectors.clone()));
            self.storage_manager.store_node(leaf_node)?;
        }

        // Update the root node with its children and vectors
        self.storage_manager.store_node(&mut new_root)?;

        // Step 5: Split the nodes until it is balanced/there are no nodes that are full
        let mut current_nodes = vec![new_root];
        while let Some(mut node) = current_nodes.pop() {
            if node.is_full() {
                let mut siblings = node.split().expect("Failed to split node");
                let sibling_offsets: Vec<i64> = siblings
                    .iter_mut()
                    .map(|sibling| {
                        sibling.parent_offset = node.parent_offset; // Set parent offset before storing
                        self.storage_manager.store_node(sibling).unwrap()
                    })
                    .collect();

                for sibling in siblings.clone() {
                    if sibling.node_type == NodeType::Internal
                        && sibling.children.len() != sibling.vectors.len()
                    {
                        panic!("Internal node has different number of children and vectors");
                    }
                }

                if node.is_root {
                    let mut new_root = Node::new_internal();
                    new_root.is_root = true;
                    new_root.children.push(node.offset);
                    new_root.vectors.push(find_modes(node.vectors.clone()));
                    for sibling_offset in &sibling_offsets {
                        let sibling = self.storage_manager.load_node(*sibling_offset).unwrap();
                        new_root.vectors.push(find_modes(sibling.vectors));
                        new_root.children.push(*sibling_offset);
                    }
                    self.storage_manager.store_node(&mut new_root)?;
                    self.storage_manager.set_root_offset(new_root.offset);
                    node.is_root = false;
                    node.parent_offset = Some(new_root.offset);
                    siblings
                        .iter_mut()
                        .for_each(|sibling| sibling.parent_offset = Some(new_root.offset));
                    self.storage_manager.store_node(&mut node)?;
                    siblings.iter_mut().for_each(|sibling| {
                        if sibling.node_type == NodeType::Internal
                            && sibling.children.len() != sibling.vectors.len()
                        {
                            panic!("Internal node has different number of children and vectors v3");
                        }
                        self.storage_manager.store_node(sibling);
                    });
                } else {
                    let parent_offset = node.parent_offset.unwrap();
                    let mut parent = self.storage_manager.load_node(parent_offset).unwrap();
                    parent.children.push(node.offset);
                    parent.vectors.push(find_modes(node.vectors.clone()));
                    sibling_offsets
                        .iter()
                        .for_each(|&offset| parent.children.push(offset));
                    siblings.iter().for_each(|sibling| {
                        parent.vectors.push(find_modes(sibling.vectors.clone()))
                    });
                    if parent.node_type == NodeType::Internal
                        && parent.children.len() != parent.vectors.len()
                    {
                        panic!("parent node has different number of children and vectors");
                    }
                    self.storage_manager.store_node(&mut parent)?;
                    node.parent_offset = Some(parent_offset);
                    self.storage_manager.store_node(&mut node)?;
                    siblings.into_iter().for_each(|mut sibling| {
                        if sibling.node_type == NodeType::Internal
                            && sibling.children.len() != sibling.vectors.len()
                        {
                            panic!("Internal node has different number of children and vectors v3");
                        }
                        sibling.parent_offset = Some(parent_offset);
                        self.storage_manager.store_node(&mut sibling);
                    });

                    let mut current_node = parent;
                    while current_node.is_full() {
                        let mut siblings = current_node.split().expect("Failed to split node");
                        let sibling_offsets: Vec<i64> = siblings
                            .iter_mut()
                            .map(|sibling| {
                                sibling.parent_offset = Some(current_node.parent_offset.unwrap());
                                self.storage_manager.store_node(sibling).unwrap()
                            })
                            .collect();

                        for sibling in siblings.clone() {
                            if sibling.node_type == NodeType::Internal
                                && sibling.children.len() != sibling.vectors.len()
                            {
                                panic!(
                                    "Internal node has different number of children and vectors v2"
                                );
                            }
                        }

                        if current_node.is_root {
                            let mut new_root = Node::new_internal();
                            new_root.is_root = true;
                            new_root.children.push(current_node.offset);
                            new_root.children.extend(sibling_offsets.clone());
                            new_root
                                .vectors
                                .push(find_modes(current_node.vectors.clone()));
                            siblings.iter().for_each(|sibling| {
                                new_root.vectors.push(find_modes(sibling.vectors.clone()))
                            });
                            self.storage_manager.store_node(&mut new_root)?;
                            self.storage_manager.set_root_offset(new_root.offset);
                            current_node.is_root = false;
                            current_node.parent_offset = Some(new_root.offset);
                            siblings
                                .iter_mut()
                                .for_each(|sibling| sibling.parent_offset = Some(new_root.offset));
                            self.storage_manager.store_node(&mut current_node)?;
                            siblings.into_iter().for_each(|mut sibling| {
                                if sibling.node_type == NodeType::Internal && sibling.children.len() != sibling.vectors.len() {
                                    panic!("Internal node has different number of children and vectors v4");
                                }
                                self.storage_manager.store_node(&mut sibling);
                            });
                        } else {
                            let parent_offset = current_node.parent_offset.unwrap();
                            let mut parent = self.storage_manager.load_node(parent_offset).unwrap();
                            parent.children.push(current_node.offset);
                            sibling_offsets
                                .iter()
                                .for_each(|&offset| parent.children.push(offset));
                            parent
                                .vectors
                                .push(find_modes(current_node.vectors.clone()));
                            siblings.iter().for_each(|sibling| {
                                if sibling.node_type == NodeType::Internal && sibling.children.len() != sibling.vectors.len() {
                                    panic!("Internal node has different number of children and vectors v5");
                                }
                                parent.vectors.push(find_modes(sibling.vectors.clone()))
                            });
                            self.storage_manager.store_node(&mut parent)?;
                            current_node.parent_offset = Some(parent_offset);
                            self.storage_manager.store_node(&mut current_node)?;
                            siblings.into_iter().for_each(|mut sibling| {
                                sibling.parent_offset = Some(parent_offset);
                                self.storage_manager.store_node(&mut sibling);
                            });
                            current_node = parent;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn summarize_tree(&self) {
        let mut queue = vec![self.storage_manager.root_offset()];
        let mut depth = 0;

        while !queue.is_empty() {
            let mut next_queue = Vec::new();

            for offset in queue {
                let node = self.storage_manager.load_node(offset).unwrap();
                println!(
                    "Depth: {}, Node type: {:?}, Offset: {}, Children: {}, Vectors: {}",
                    depth,
                    node.node_type,
                    node.offset,
                    node.children.len(),
                    node.vectors.len()
                );

                if node.node_type == NodeType::Internal {
                    next_queue.extend(node.children);
                }
            }

            queue = next_queue;
            depth += 1;
        }

        println!("Tree depth: {}", depth);
    }

    // pub fn insert(&mut self, key: K, value: V) -> Result<(), HaystackError> {
    //     // println!("Inserting key: {}, value: {}", key, value);
    //     let mut root = self
    //         .storage_manager
    //         .load_node(self.storage_manager.root_offset())?;

    //     // println!("Root offset: {}, {}", self.root_offset, root.offset);

    //     if root.is_full() {
    //         // println!("Root is full, needs splitting");
    //         let mut new_root = Node::new_internal(-1);
    //         new_root.is_root = true;
    //         let (median, mut sibling) = root.split(self.b)?;
    //         // println!("Root split: median = {}, new sibling created", median);
    //         // println!("Root split: median = {}, new sibling created", median);
    //         root.is_root = false;
    //         self.storage_manager.store_node(&mut root)?;
    //         // println!("Root stored");
    //         let sibling_offset = self.storage_manager.store_node(&mut sibling)?;
    //         new_root.keys.push(median);
    //         new_root.children.push(self.storage_manager.root_offset()); // old root offset
    //         new_root.children.push(sibling_offset); // new sibling offset
    //         new_root.is_root = true;
    //         self.storage_manager.store_node(&mut new_root)?;
    //         self.storage_manager.set_root_offset(new_root.offset);

    //         root.parent_offset = Some(new_root.offset);
    //         sibling.parent_offset = Some(new_root.offset);
    //         self.storage_manager.store_node(&mut root)?;
    //         self.storage_manager.store_node(&mut sibling)?;
    //     }
    //     // println!("Inserting into non-full root");
    //     self.insert_non_full(self.storage_manager.root_offset(), key, value, 0)?;

    //     // println!("Inserted key, root offset: {}", self.root_offset);

    //     Ok(())
    // }

    // fn insert_non_full(
    //     &mut self,
    //     node_offset: i64,
    //     key: K,
    //     value: V,
    //     depth: usize,
    // ) -> Result<(), HaystackError> {
    //     // if depth > 100 {
    //     //     // Set a reasonable limit based on your observations
    //     //     println!("Recursion depth limit reached: {}", depth);
    //     //     return Ok(());
    //     // }

    //     let mut node = self.storage_manager.load_node(node_offset)?;
    //     // println!(
    //     //     "Depth: {}, Node type: {:?}, Keys: {:?}, is_full: {}",
    //     //     depth,
    //     //     node.node_type,
    //     //     node.keys,
    //     //     node.is_full()
    //     // );

    //     if node.node_type == NodeType::Leaf {
    //         let idx = node.keys.binary_search(&key).unwrap_or_else(|x| x);
    //         // println!(
    //         //     "Inserting into leaf node: key: {}, len: {}",
    //         //     key,
    //         //     node.keys.len()
    //         // );
    //         // println!(
    //         //     "Inserting into leaf node: key: {}, idx: {}, node_offset: {}",
    //         //     key, idx, node_offset
    //         // );

    //         if node.keys.get(idx) == Some(&key) {
    //             node.values[idx] = Some(value);

    //             // println!(
    //             //     "Storing leaf node with keys: {:?}, offset: {}",
    //             //     node.keys, node.offset
    //             // );
    //             self.storage_manager.store_node(&mut node)?;
    //             if node.is_root {
    //                 // println!("Updating root offset to: {}", node.offset);
    //                 // self.root_offset = node.offset.clone();
    //                 self.storage_manager.set_root_offset(node.offset);
    //             }
    //         } else {
    //             node.keys.insert(idx, key);
    //             node.values.insert(idx, Some(value));

    //             // println!(
    //             //     "Storing leaf node with keys: {:?}, offset: {}",
    //             //     node.keys, node.offset
    //             // );
    //             self.storage_manager.store_node(&mut node)?;
    //             if node.is_root {
    //                 // println!("Updating root offset to: {}", node.offset);
    //                 // self.root_offset = node.offset.clone();
    //                 self.storage_manager.set_root_offset(node.offset);
    //             }
    //         }
    //     } else {
    //         let idx = node.keys.binary_search(&key).unwrap_or_else(|x| x); // Find the child to go to
    //         let child_offset = node.children[idx];
    //         let mut child = self.storage_manager.load_node(child_offset)?;

    //         if child.is_full() {
    //             // println!("Child is full, needs splitting");
    //             let (median, mut sibling) = child.split(self.b)?;
    //             let sibling_offset = self.storage_manager.store_node(&mut sibling)?;

    //             node.keys.insert(idx, median.clone());
    //             node.children.insert(idx + 1, sibling_offset);
    //             self.storage_manager.store_node(&mut node)?;

    //             if key < median {
    //                 self.insert_non_full(child_offset, key, value, depth + 1)?;
    //             } else {
    //                 self.insert_non_full(sibling_offset, key, value, depth + 1)?;
    //             }
    //         } else {
    //             self.insert_non_full(child_offset, key, value, depth + 1)?;
    //         }
    //     }

    //     Ok(())
    // }

    pub fn len(&self) -> usize {
        let mut i = 0;
        for item in self.storage_manager.data.iter() {
            match item.node_type {
                NodeType::Leaf => i += item.ids.len(),
                _ => (),
            }
        }

        i
    }
}
