use crate::structures::tree::node::Node;

use super::serialization::{TreeDeserialization, TreeSerialization};
use crate::errors::HaystackError;
use std::fmt::Debug;

pub struct StorageManager<K, V> {
    pub data: Vec<Node<K, V>>,
    root_offset: i64,
}

impl<K, V> StorageManager<K, V>
where
    K: Clone + Ord + TreeSerialization + TreeDeserialization + Debug,
    V: Clone + TreeSerialization + TreeDeserialization,
{
    pub fn new() -> Self {
        StorageManager {
            data: Vec::new(),
            root_offset: 0,
        }
    }

    pub fn store_node(&mut self, node: &mut Node<K, V>) -> Result<i64, HaystackError> {
        if node.offset == -1 {
            // println!("Pushing node to storage");
            // println!("node has keys: {:?}", node.keys);
            // println!("{}", self.data.len() as i64);
            node.offset = self.data.len() as i64;
            self.data.push(node.clone());

            return Ok(node.offset);
        }

        let offset = node.offset as usize;
        self.data[offset] = node.clone();
        // self.data.insert(offset, node.clone());
        Ok(offset as i64)
    }

    pub fn load_node(&mut self, offset: i64) -> Result<Node<K, V>, HaystackError> {
        let mut node = self.data[offset as usize].clone();
        node.offset = offset;

        Ok(node)
    }

    pub fn root_offset(&self) -> i64 {
        self.root_offset
    }

    pub fn set_root_offset(&mut self, root_offset: i64) {
        self.root_offset = root_offset;
    }
}
