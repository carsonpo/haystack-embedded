use serde::Serialize;

use crate::structures::ann_tree::k_modes::{balanced_k_modes, balanced_k_modes_4};
use crate::structures::metadata_index::KVPair;
use crate::structures::tree::node::{read_length, serialize_length};
use crate::{constants::QUANTIZED_VECTOR_SIZE, errors::HaystackError};
use std::fmt::Debug;

use super::k_modes::balanced_k_modes_k_clusters;

pub type Vector = [u8; QUANTIZED_VECTOR_SIZE];

#[derive(Debug, PartialEq, Clone)]
pub struct Node {
    pub vector: Vector,
    pub id: u128,
    pub metadata: Vec<KVPair>,

    pub offset: i64,
    pub depth: usize,

    pub left: Option<i64>,
    pub right: Option<i64>,
    pub parent: Option<i64>,

    pub is_root: bool,
    pub is_red: bool,
}

impl Node {
    pub fn new(vector: Vector, id: u128, metadata: Vec<KVPair>) -> Self {
        Node {
            vector,
            id,
            metadata,
            offset: -1,
            depth: 0,
            left: None,
            right: None,
            parent: None,
            is_root: false,
            is_red: true,
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut serialized = Vec::new(); // Adjust capacity based on expected size

        // Serialize node_type and is_root
        serialized.push(self.is_root as u8);
        serialized.push(self.is_red as u8);

        // Serialize parent_offset
        serialized.extend_from_slice(&(self.parent.unwrap_or(-1) as i64).to_le_bytes());

        // Serialize offset
        serialized.extend_from_slice(&self.offset.to_le_bytes());

        // Serialize left and right

        serialized.extend_from_slice(&self.left.unwrap_or(-1).to_le_bytes());
        serialized.extend_from_slice(&self.right.unwrap_or(-1).to_le_bytes());

        // serialize depth
        serialized.extend_from_slice(&(self.depth as i64).to_le_bytes());

        // Serialize vector

        serialized.extend_from_slice(&self.vector);

        // Serialize id
        serialized.extend_from_slice(&self.id.to_le_bytes());

        // Serialize metadata

        serialize_metadata(&mut serialized, &self.metadata);

        serialized
    }

    pub fn deserialize(data: &[u8]) -> Self {
        let mut offset = 0;

        // Deserialize node_type and is_root

        let is_root = data[offset] == 1;

        offset += 1;

        let is_red = data[offset] == 1;

        offset += 1;

        // Deserialize parent_offset

        let parent = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());

        offset += 8;

        // Deserialize offset

        let offset_value = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());

        offset += 8;

        let left = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());

        offset += 8;

        let right = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());

        offset += 8;

        // deserialize depth

        let depth = i64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());

        offset += 8;

        // Deserialize vector

        let mut vector = [0; QUANTIZED_VECTOR_SIZE];

        vector.copy_from_slice(&data[offset..offset + QUANTIZED_VECTOR_SIZE]);

        offset += QUANTIZED_VECTOR_SIZE;

        // Deserialize id

        let id = u128::from_le_bytes(data[offset..offset + 16].try_into().unwrap());

        offset += 16;

        // Deserialize metadata

        let metadata = deserialize_metadata(&data[offset..]);

        Node {
            vector,
            id,
            metadata,
            offset: offset_value,
            depth: depth as usize,
            left: Some(left),
            right: Some(left),
            parent: if parent == -1 { None } else { Some(parent) },
            is_root,
            is_red,
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
