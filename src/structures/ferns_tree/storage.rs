use crate::structures::ferns_tree::node::Node;

use crate::errors::HaystackError;
use std::fmt::Debug;

pub struct StorageManager {
    pub data: Vec<Node>,
    root_offset: i64,
}

impl StorageManager {
    pub fn new() -> Self {
        StorageManager {
            data: Vec::new(),
            root_offset: -1,
        }
    }

    pub fn store_node(&mut self, node: &mut Node) -> Result<i64, HaystackError> {
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

    pub fn load_node(&mut self, offset: i64) -> Result<Node, HaystackError> {
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
