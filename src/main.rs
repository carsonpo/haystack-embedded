pub mod constants;
pub mod errors;
pub mod math;
pub mod structures;
pub mod utils;

use constants::QUANTIZED_VECTOR_SIZE;
use rand::Rng;
use structures::ann_tree::Tree;
use structures::filters::Filter;
use structures::metadata_index::KVPair;

fn make_random_vector() -> [u8; QUANTIZED_VECTOR_SIZE] {
    let mut vector = [0u8; QUANTIZED_VECTOR_SIZE];
    for i in 0..QUANTIZED_VECTOR_SIZE {
        vector[i] = rand::random();
    }
    vector
}

// slightly perturb the vector by flipping a few bits, NOT bytes
fn perturb_vector(vector: [u8; QUANTIZED_VECTOR_SIZE]) -> [u8; QUANTIZED_VECTOR_SIZE] {
    let mut new_vector = vector.clone();

    for i in 0..QUANTIZED_VECTOR_SIZE * 8 {
        if rand::random::<f32>() < 0.025 {
            new_vector[i / 8] ^= 1 << (i % 8);
        }
    }

    new_vector
}
fn main() {
    let mut tree = Tree::new().unwrap();

    let mut vectors = vec![[1u8; QUANTIZED_VECTOR_SIZE]];

    let mut ids = vec![0];

    let metadata = vec![KVPair::new("key".to_string(), "value".to_string())];

    // tree.insert([1u8; QUANTIZED_VECTOR_SIZE], 0, metadata.clone());
    // tree.insert([1u8; QUANTIZED_VECTOR_SIZE], 0, metadata.clone());
    // tree.insert([1u8; QUANTIZED_VECTOR_SIZE], 0, metadata.clone());
    // tree.insert([1u8; QUANTIZED_VECTOR_SIZE], 0, metadata.clone());
    // tree.insert([1u8; QUANTIZED_VECTOR_SIZE], 0, metadata.clone());
    // tree.insert([1u8; QUANTIZED_VECTOR_SIZE], 0, metadata.clone());

    let mut query_vector = make_random_vector();

    // tree.insert(query_vector, 0, metadata.clone());
    // tree.insert(query_vector, 0, metadata.clone());

    const NUM_VECTORS: usize = 20_000_000;

    const TARGET_FRACTION: usize = 100_000;

    // tree.insert(query_vector, 0, metadata.clone());
    // tree.insert(query_vector, 0, metadata.clone());

    let mut all_vectors = Vec::new();
    let mut all_ids = Vec::new();
    let mut all_metadata = Vec::new();

    for i in 0..(NUM_VECTORS / TARGET_FRACTION) {
        // tree.insert(perturb_vector(query_vector), i as u128, metadata.clone());
        all_vectors.push(perturb_vector(query_vector));
        all_ids.push(i as u128);
        all_metadata.push(metadata.clone());
    }

    // let mut batch_vectors = Vec::new();
    // let mut batch_ids = Vec::new();
    // let mut batch_metadata = Vec::new();

    // for i in 0..NUM_VECTORS / 10 {
    //     // vectors.push(make_random_vector());
    //     // ids.push(rand::random());
    //     tree.insert(make_random_vector(), rand::random(), metadata.clone());

    //     // batch_vectors.push(make_random_vector());
    //     // batch_ids.push(rand::random());
    //     // batch_metadata.push(metadata.clone());
    // }

    let mut has_id_been_set = false;

    for i in 0..NUM_VECTORS {
        // vectors.push(make_random_vector());
        // ids.push(rand::random());
        let vec = make_random_vector();

        // if i % 100 == 0 {
        //     println!("Inserting vector {}", i);
        // }

        // tree.insert(vec, rand::random(), metadata.clone());

        all_vectors.push(vec);
        all_ids.push(rand::random());
        all_metadata.push(metadata.clone());

        // batch_vectors.push(make_random_vector());
        // batch_ids.push(rand::random());
        // batch_metadata.push(metadata.clone());
    }

    tree.bulk_insert_and_calibrate(all_vectors, all_ids, all_metadata);

    tree.true_calibrate();

    println!("Calibrated");

    // tree.summarize_tree();

    // for i in 0..(NUM_VECTORS / TARGET_FRACTION) {
    //     tree.insert(query_vector, 0, metadata.clone());
    // }

    // tree.calibrate();

    // tree.batch_insert(batch_vectors, batch_ids, batch_metadata);

    // for (vector, id) in vectors.iter().zip(ids.iter()) {
    //     tree.insert(*vector, *id);
    // }

    let result = tree.search(
        query_vector,
        (NUM_VECTORS / TARGET_FRACTION) as usize,
        Filter::Eq("key".to_string(), "value".to_string()),
    );

    println!("Result: {:?}", result);

    for _ in 0..1000 {
        let _ = tree.search(
            query_vector,
            1,
            Filter::Eq("key".to_string(), "value".to_string()),
        );
    }

    let start = std::time::Instant::now();

    for _ in 0..1000 {
        let _ = tree.search(
            query_vector,
            1,
            Filter::Eq("key".to_string(), "value".to_string()),
        );
    }

    println!("Time taken: {:?}", start.elapsed().div_f32(1000.0));

    // assert_eq!(result, vec![0]);
    // num_found is the number of items with ids between 0 and 10
    let mut found = result
        .iter()
        .filter(|&&x| x <= (NUM_VECTORS / TARGET_FRACTION) as u128)
        .collect::<Vec<&u128>>();
    found.dedup();

    let num_found = found.len() as f32 / (result.len()) as f32 * 100.0;

    // assert_eq!(num_found, 10);
    println!("Num found: {}%, Out of {}", (num_found), NUM_VECTORS);
}
