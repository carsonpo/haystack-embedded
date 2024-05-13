use serde::Serialize;

use crate::structures::ann_tree::k_modes::{balanced_k_modes, balanced_k_modes_4};
use crate::structures::metadata_index::KVPair;
use crate::structures::tree::node::{read_length, serialize_length};
use crate::{constants::QUANTIZED_VECTOR_SIZE, errors::HaystackError};
use std::fmt::Debug;

use super::k_modes::balanced_k_modes_k_clusters;

#[derive(Debug, PartialEq, Clone)]
pub enum NodeType {
    Leaf,
    Internal,
}

pub fn serialize_node_type(node_type: &NodeType) -> [u8; 1] {
    match node_type {
        NodeType::Leaf => [0],
        NodeType::Internal => [1],
    }
}

pub fn deserialize_node_type(data: &[u8]) -> NodeType {
    match data[0] {
        0 => NodeType::Leaf,
        1 => NodeType::Internal,
        _ => panic!("Invalid node type"),
    }
}

const K: usize = crate::constants::K;

pub type Vector = [u8; QUANTIZED_VECTOR_SIZE];

#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    pub vectors: Vec<Vector>,
    pub ids: Vec<u128>,
    pub children: Vec<i64>,
    pub metadata: Vec<Vec<KVPair>>,
    pub k: usize,
    pub node_type: NodeType,
    pub offset: i64,
    pub is_root: bool,
    pub parent_offset: Option<i64>,
}

impl Node {
    pub fn new_leaf() -> Self {
        Node {
            vectors: Vec::new(),
            ids: Vec::new(),
            children: Vec::new(),
            metadata: Vec::new(),
            k: K,
            node_type: NodeType::Leaf,
            offset: -1,
            is_root: false,
            parent_offset: Some(-1),
        }
    }

    pub fn new_internal() -> Self {
        Node {
            vectors: Vec::new(),
            ids: Vec::new(),
            children: Vec::new(),
            metadata: Vec::new(),
            k: K,
            node_type: NodeType::Internal,
            offset: -1,
            is_root: false,
            parent_offset: Some(-1),
        }
    }

    pub fn split(&mut self) -> Result<Vec<Node>, HaystackError> {
        let k = match self.node_type {
            NodeType::Leaf => 2,
            NodeType::Internal => 2,
        };
        if self.vectors.len() < k {
            panic!("Cannot split a node with less than k keys");
        }

        // Assuming a modified balanced_k_modes that returns k clusters of indices
        let clusters_indices = balanced_k_modes_k_clusters(self.vectors.clone(), k);

        let mut clusters_vectors = vec![Vec::new(); k];
        let mut clusters_ids = vec![Vec::new(); k];
        let mut clusters_children = vec![Vec::new(); k];
        let mut clusters_metadata = vec![Vec::new(); k];

        // Distribute vectors, IDs, children, and metadata based on indices from clustering
        for (i, indices) in clusters_indices.iter().enumerate() {
            if indices.is_empty() {
                panic!("Empty cluster found");
            }
            for &index in indices {
                clusters_vectors[i].push(self.vectors[index].clone());
                if self.node_type == NodeType::Leaf {
                    clusters_ids[i].push(self.ids[index].clone());
                    clusters_metadata[i].push(self.metadata[index].clone());
                }
                if self.node_type == NodeType::Internal {
                    clusters_children[i].push(self.children[index].clone());
                }
            }
        }

        let mut siblings = Vec::new();

        // Create sibling nodes for the second, third, ..., k-th clusters
        for i in 1..k {
            let sibling = Node {
                vectors: clusters_vectors[i].clone(),
                ids: clusters_ids[i].clone(),
                children: clusters_children[i].clone(),
                metadata: clusters_metadata[i].clone(),
                k: self.k,
                node_type: self.node_type.clone(),
                offset: -1, // This should be set when the node is stored
                is_root: false,
                parent_offset: self.parent_offset,
            };
            siblings.push(sibling.clone());

            if sibling.node_type == NodeType::Internal
                && (sibling.vectors.len() != sibling.clone().children.len()
                    || sibling.children.is_empty())
            {
                panic!("Internal node vectors and children must be the same length");
            }
        }

        // Update the current node with the first cluster
        self.vectors = clusters_vectors[0].clone();
        self.ids = clusters_ids[0].clone();
        self.children = clusters_children[0].clone();
        self.metadata = clusters_metadata[0].clone();

        Ok(siblings)
    }

    pub fn is_full(&self) -> bool {
        return self.vectors.len() >= self.k;
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut serialized = Vec::new(); // Adjust capacity based on expected size

        // Serialize node_type and is_root
        serialized.extend_from_slice(&serialize_node_type(&self.node_type));
        serialized.push(self.is_root as u8);

        // Serialize parent_offset
        serialized.extend_from_slice(&(self.parent_offset.unwrap_or(-1) as i64).to_le_bytes());

        // Serialize offset
        serialized.extend_from_slice(&self.offset.to_le_bytes());

        // Serialize vectors
        serialize_length(&mut serialized, self.vectors.len() as u32);
        for vector in &self.vectors {
            serialized.extend_from_slice(vector);
        }

        // Serialize ids
        serialize_length(&mut serialized, self.ids.len() as u32);
        for id in &self.ids {
            serialized.extend_from_slice(&id.to_le_bytes());
        }

        // Serialize children
        serialize_length(&mut serialized, self.children.len() as u32);
        for child in &self.children {
            serialized.extend_from_slice(&child.to_le_bytes());
        }

        // Serialize metadata
        serialize_length(&mut serialized, self.metadata.len() as u32);
        for meta in &self.metadata {
            serialize_metadata(&mut serialized, meta); // A function to serialize a Vec<KVPair>
        }

        serialized
    }

    pub fn deserialize(data: &[u8]) -> Self {
        let mut offset = 0;

        // Deserialize node_type and is_root
        let node_type = deserialize_node_type(&data[offset..offset + 1]);
        offset += 1;
        let is_root = data[offset] == 1;
        offset += 1;

        // Deserialize parent_offset
        let parent_offset = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Deserialize offset
        let node_offset = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        offset += 8;

        // Deserialize vectors
        let vectors_len = read_length(&data[offset..offset + 4]);
        offset += 4;
        let mut vectors = Vec::with_capacity(vectors_len);
        for _ in 0..vectors_len {
            vectors.push(
                data[offset..offset + QUANTIZED_VECTOR_SIZE]
                    .try_into()
                    .unwrap(),
            );
            offset += QUANTIZED_VECTOR_SIZE;
        }

        // Deserialize ids
        let ids_len = read_length(&data[offset..offset + 4]);
        offset += 4;
        let mut ids = Vec::with_capacity(ids_len);
        for _ in 0..ids_len {
            let id = u128::from_le_bytes(data[offset..offset + 16].try_into().unwrap());
            offset += 16;
            ids.push(id);
        }

        // Deserialize children
        let children_len = read_length(&data[offset..offset + 4]);
        offset += 4;
        let mut children = Vec::with_capacity(children_len);
        for _ in 0..children_len {
            let child = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            offset += 8;
            children.push(child);
        }

        // Deserialize metadata
        let metadata_len = read_length(&data[offset..offset + 4]);
        offset += 4;
        let mut metadata = Vec::with_capacity(metadata_len);
        for _ in 0..metadata_len {
            metadata.push(deserialize_metadata(&data[offset..]));
            // Increment offset inside deserialize_metadata based on actual size
        }

        Node {
            vectors,
            ids,
            children,
            metadata,
            k: K,
            node_type,
            offset: node_offset,
            is_root,
            parent_offset: if parent_offset != -1 {
                Some(parent_offset)
            } else {
                None
            },
        }
    }
}
impl Default for Node {
    fn default() -> Self {
        Node {
            vectors: Vec::new(),
            ids: Vec::new(),
            children: Vec::new(),
            metadata: Vec::new(),
            k: K,                      // Adjust this as necessary
            node_type: NodeType::Leaf, // Or another appropriate default NodeType
            offset: -1,
            is_root: false,
            parent_offset: None,
        }
    }
}

fn serialize_metadata(serialized: &mut Vec<u8>, metadata: &[KVPair]) {
    // Serialize the length of the metadata vector
    serialize_length(serialized, metadata.len() as u32);

    for kv in metadata {
        let serialized_kv = kv.serialize(); // Assuming KVPair has a serialize method that returns Vec<u8>
                                            // Serialize the length of this KVPair
        serialize_length(serialized, serialized_kv.len() as u32);
        // Append the serialized KVPair
        serialized.extend_from_slice(&serialized_kv);
    }
}

fn deserialize_metadata(data: &[u8]) -> Vec<KVPair> {
    let mut offset = 0;

    // Read the length of the metadata vector
    let metadata_len = read_length(&data[offset..offset + 4]) as usize;
    offset += 4;

    let mut metadata = Vec::with_capacity(metadata_len);
    for _ in 0..metadata_len {
        // Read the length of the next KVPair
        let kv_length = read_length(&data[offset..offset + 4]) as usize;
        offset += 4;

        // Deserialize the KVPair from the next segment
        let kv = KVPair::deserialize(&data[offset..offset + kv_length]);
        offset += kv_length;

        metadata.push(kv);
    }

    metadata
}
