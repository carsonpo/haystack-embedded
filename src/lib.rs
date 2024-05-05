#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use js_sys::Float32Array;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use serde_json;
use std::collections::HashMap;

use constants::VECTOR_SIZE;
use libflate::gzip::{Decoder, Encoder};
use rayon::prelude::*;
use std::io::{Read, Write};
use structures::{
    filters::{Filter, Filters},
    inverted_index::InvertedIndexItem,
    metadata_index::{KVPair, MetadataIndexItem},
    namespace_state::NamespaceState,
};
use utils::quantize;

pub mod constants;
pub mod errors;
pub mod math;
pub mod structures;
pub mod utils;

use crate::errors::HaystackError;

#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[cfg_attr(feature = "python", pyclass)]
pub struct HaystackEmbedded {
    state: NamespaceState,
}

#[cfg_attr(feature = "python", pymethods)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl HaystackEmbedded {
    #[cfg(feature = "python")]
    #[new]
    pub fn new(namespace_id: String) -> HaystackEmbedded {
        let state = NamespaceState::new(namespace_id);
        HaystackEmbedded { state }
    }

    #[cfg(feature = "wasm")]
    #[wasm_bindgen(constructor)]
    pub fn new(namespace_id: String) -> HaystackEmbedded {
        let state = NamespaceState::new(namespace_id);
        HaystackEmbedded { state }
    }

    fn inner_add_vector(&mut self, vector: [f32; VECTOR_SIZE], metadata: String) {
        let quantized_vector = quantize(&vector);

        let metadata: Vec<KVPair> = serde_json::from_str(&metadata).unwrap();

        let vector_index = self
            .state
            .vectors
            .push(quantized_vector)
            .expect("Failed to add vector");

        let id = uuid::Uuid::new_v4().as_u128();

        for kv in metadata.clone() {
            if kv.key == "text" {
                continue;
            }
            let inverted_index_item = InvertedIndexItem {
                indices: vec![vector_index],
                ids: vec![id],
            };
            self.state
                .inverted_index
                .insert_append(kv, inverted_index_item)
        }

        let metadata_index_item = MetadataIndexItem {
            id,
            kvs: metadata,
            vector_index,
        };

        self.state.metadata_index.insert(id, metadata_index_item);
    }

    fn inner_batch_add_vectors(
        &mut self,
        vectors: Vec<[f32; VECTOR_SIZE]>,
        metadata: String,
    ) -> u8 {
        let quantized_vectors: Vec<_> = vectors.iter().map(|v| quantize(v)).collect();
        let metadata: Vec<Vec<KVPair>> = serde_json::from_str(&metadata).unwrap();

        // Ensure that vectors and metadata are of equal length
        assert_eq!(
            vectors.len(),
            metadata.len(),
            "Vectors and metadata length mismatch"
        );

        let vector_indices = self
            .state
            .vectors
            .batch_push(quantized_vectors)
            .expect("Failed to add vectors");

        let ids: Vec<u128> = (0..vectors.len())
            .map(|_| uuid::Uuid::new_v4().as_u128())
            .collect();

        let mut inverted_index_items: HashMap<KVPair, Vec<(usize, u128)>> = HashMap::new();
        let mut batch_metadata_to_insert = Vec::new();

        for (idx, kvs) in metadata.iter().enumerate() {
            let metadata_index_item = MetadataIndexItem {
                id: ids[idx],
                kvs: kvs.clone(),
                vector_index: vector_indices[idx],
            };

            batch_metadata_to_insert.push((ids[idx], metadata_index_item));

            for kv in kvs {
                inverted_index_items
                    .entry(kv.clone())
                    .or_insert_with(Vec::new)
                    .push((vector_indices[idx], ids[idx]));
            }
        }

        // Insert metadata
        // self.state
        //     .metadata_index
        //     .batch_insert(batch_metadata_to_insert);

        for (id, item) in batch_metadata_to_insert {
            self.state.metadata_index.insert(id, item);
        }

        // Insert inverted index items
        for (kv, items) in inverted_index_items {
            let inverted_index_item = InvertedIndexItem {
                indices: items.iter().map(|(idx, _)| *idx).collect(),
                ids: items.iter().map(|(_, id)| *id).collect(),
            };

            self.state
                .inverted_index
                .insert_append(kv, inverted_index_item);
        }

        0
    }

    fn inner_query(&mut self, vector: [f32; VECTOR_SIZE], filters: String, k: usize) -> String {
        let quantized_vector = quantize(&vector);
        let filters: Filter = serde_json::from_str(&filters).unwrap();
        let (indices, ids) =
            Filters::evaluate(&filters, &mut self.state.inverted_index).get_indices();

        // Create batches
        let mut batch_indices: Vec<Vec<usize>> = Vec::new();
        let mut current_batch = Vec::new();

        for index in indices {
            if current_batch.is_empty() {
                current_batch.push(index);
            } else {
                let last_index = *current_batch.last().unwrap();
                if index == last_index + 1 {
                    current_batch.push(index);
                } else {
                    batch_indices.push(current_batch);
                    current_batch = vec![index];
                }
            }
        }

        if !current_batch.is_empty() {
            batch_indices.push(current_batch);
        }

        let top_k_to_use = k.min(ids.len());
        let mut all_top_k_indices: Vec<(u128, u16)> = Vec::new();

        for batch in batch_indices {
            let vectors = self
                .state
                .vectors
                .get_contiguous(batch[0], batch.len())
                .unwrap();

            let mut batch_top_k: Vec<(u128, u16)> = vectors
                .par_iter()
                .enumerate()
                .map(|(i, vector)| {
                    let global_idx = batch[0] + i;
                    if global_idx >= ids.len() {
                        return (0, u16::MAX);
                    }
                    let distance = math::hamming_distance(&quantized_vector, vector);
                    (ids[global_idx], distance)
                })
                .collect();

            batch_top_k.sort_by_key(|&(_, distance)| distance);
            batch_top_k.truncate(top_k_to_use);

            all_top_k_indices.extend(batch_top_k);
        }

        // Aggregate and sort
        all_top_k_indices.sort_by_key(|&(_, distance)| distance);
        all_top_k_indices.truncate(top_k_to_use);

        let mut results = Vec::new();

        for (idx, _) in all_top_k_indices.clone() {
            if let Some(item) = self.state.metadata_index.get(idx) {
                results.push(item.kvs.clone());
            } else {
                // results.push(vec![])
                // panic!("Index not found: {}", idx);
                // I want to print out as much debug info as possible
                // so I can figure out what's going on

                panic!(
                    "ID not found: {}. top_k: {}, all_top_k_indices: {:?}, metadata index len: {:?}, inverted index len: {:?}",
                    idx, k, all_top_k_indices, self.state.metadata_index.len(), self.state.inverted_index.len()
                )
            }
        }

        serde_json::to_string(&results).unwrap()
    }

    #[cfg(feature = "wasm")]
    pub fn query(&mut self, vector: Float32Array, filters: String, k: usize) -> String {
        let mut vector_array = [0.0; VECTOR_SIZE];
        vector_array.copy_from_slice(&vector.to_vec());

        self.inner_query(vector_array, filters, k)
    }

    #[cfg(feature = "python")]
    pub fn query(&mut self, vector: [f32; VECTOR_SIZE], filters: String, k: usize) -> String {
        self.inner_query(vector, filters, k)
    }

    #[cfg(feature = "wasm")]
    pub fn add_vector(&mut self, vector: Float32Array, metadata: String) {
        let mut vector_array = [0.0; VECTOR_SIZE];
        vector_array.copy_from_slice(&vector.to_vec());

        self.inner_add_vector(vector_array, metadata)
    }

    #[cfg(feature = "python")]
    pub fn add_vector(&mut self, vector: [f32; VECTOR_SIZE], metadata: String) {
        self.inner_add_vector(vector, metadata)
    }

    #[cfg(feature = "wasm")]
    pub fn batch_add_vectors(&mut self, vectors: Vec<Float32Array>, metadata: String) -> u8 {
        let vectors_array: Vec<[f32; VECTOR_SIZE]> = vectors
            .iter()
            .map(|v| {
                let mut vector_array = [0.0; VECTOR_SIZE];
                vector_array.copy_from_slice(&v.to_vec());
                vector_array
            })
            .collect();

        self.inner_batch_add_vectors(vectors_array, metadata)
    }

    #[cfg(feature = "python")]
    pub fn batch_add_vectors(&mut self, vectors: Vec<[f32; VECTOR_SIZE]>, metadata: String) -> u8 {
        self.inner_batch_add_vectors(vectors, metadata)
    }

    pub fn save_state(&mut self) -> Vec<u8> {
        let out = self.state.save_state();
        let mut encoder = Encoder::new(Vec::new()).unwrap();
        encoder.write_all(&out).unwrap();
        let (data, _) = encoder.finish().unwrap();
        data
        // out
    }

    pub fn load_state(&mut self, data: Vec<u8>) {
        // self.state.load_state(data);
        let mut decoder = Decoder::new(&data[..]).unwrap();
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).unwrap();
        self.state.load_state(out).unwrap();
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn haystack_embedded(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HaystackEmbedded>()?;
    Ok(())
}
