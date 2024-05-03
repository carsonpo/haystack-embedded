#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use js_sys::Float32Array;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use serde_json;
use std::collections::HashMap;

use constants::VECTOR_SIZE;
use rayon::prelude::*;
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
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
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

    fn inner_batch_add_vectors(&mut self, vectors: Vec<[f32; VECTOR_SIZE]>, metadata: String) {
        let quantized_vectors = vectors.iter().map(|v| quantize(v)).collect();

        let metadata: Vec<Vec<KVPair>> = serde_json::from_str(&metadata).unwrap();

        let vector_indices = self
            .state
            .vectors
            .batch_push(quantized_vectors)
            .expect("Failed to add vectors");

        let ids = (0..vectors.len())
            .map(|_| uuid::Uuid::new_v4().as_u128())
            .collect::<Vec<u128>>();

        let mut inverted_index_items = HashMap::new();

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

        self.state
            .metadata_index
            .batch_insert(batch_metadata_to_insert);

        for (kv, items) in inverted_index_items {
            let inverted_index_item = InvertedIndexItem {
                indices: items.iter().map(|(idx, _)| *idx).collect(),
                ids: items.iter().map(|(_, id)| *id).collect(),
            };

            self.state
                .inverted_index
                .insert_append(kv, inverted_index_item);
        }
    }

    fn inner_query(&mut self, vector: [f32; VECTOR_SIZE], filters: String, k: usize) -> String {
        let quantized_vector = quantize(&vector);

        let filters: Filter = serde_json::from_str(&filters).unwrap();

        let (indices, ids) =
            Filters::evaluate(&filters, &mut self.state.inverted_index).get_indices();

        let mut batch_indices: Vec<Vec<usize>> = Vec::new();

        let mut current_batch = Vec::new();

        for index in indices {
            if current_batch.len() == 0 {
                current_batch.push(index);
            } else {
                let last_index = current_batch[current_batch.len() - 1];
                if index == last_index + 1 {
                    current_batch.push(index);
                } else {
                    batch_indices.push(current_batch);
                    current_batch = Vec::new();
                    current_batch.push(index);
                }
            }
        }

        current_batch.sort();
        current_batch.dedup();

        if current_batch.len() > 0 {
            batch_indices.push(current_batch);
        }

        let mut top_k_indices = Vec::new();

        let top_k_to_use = k.min(ids.len());

        for batch in batch_indices {
            let vectors = self
                .state
                .vectors
                .get_contiguous(batch[0], batch.len())
                .unwrap();
            top_k_indices.extend(
                vectors
                    .par_iter()
                    .enumerate()
                    .fold(
                        || Vec::new(),
                        |mut acc, (idx, vector)| {
                            let distance = math::hamming_distance(&quantized_vector, vector);

                            if acc.len() < top_k_to_use {
                                acc.push((ids[idx], distance));
                            } else {
                                acc.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                                if distance < acc[0].1 {
                                    acc[0] = (ids[idx], distance);
                                }
                            }

                            acc
                        },
                    )
                    .reduce(
                        || Vec::new(), // Initializer for the reduce step
                        |mut a, mut b| {
                            // How to combine results from different threads
                            a.append(&mut b);
                            a.sort_by_key(|&(_, dist)| dist); // Sort by distance
                            a.truncate(top_k_to_use); // Keep only the top k elements
                            a
                        },
                    ),
            )
        }

        let mut results = Vec::new();

        for (idx, _) in top_k_indices {
            let r = self.state.metadata_index.get(idx);

            match r {
                Some(item) => {
                    results.push(item.kvs.clone());
                }
                None => {}
            }
        }

        // results
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
    pub fn batch_add_vectors(&mut self, vectors: Vec<Float32Array>, metadata: String) {
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
    pub fn batch_add_vectors(&mut self, vectors: Vec<[f32; VECTOR_SIZE]>, metadata: String) {
        self.inner_batch_add_vectors(vectors, metadata)
    }

    pub fn save_state(&mut self) -> Vec<u8> {
        self.state.save_state()
    }

    pub fn load_state(&mut self, data: Vec<u8>) {
        self.state.load_state(data);
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn haystack_embedded(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HaystackEmbedded>()?;
    Ok(())
}
