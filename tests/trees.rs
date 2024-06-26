extern crate haystack_embedded;

#[cfg(test)]
mod tests {

    use haystack_embedded::errors::HaystackError;
    use haystack_embedded::structures::inverted_index::InvertedIndexItem;
    use haystack_embedded::structures::metadata_index::{KVPair, MetadataIndex, MetadataIndexItem};
    use haystack_embedded::structures::namespace_state::NamespaceState;
    use haystack_embedded::structures::tree::node::{Node, NodeType};
    use haystack_embedded::structures::tree::storage::StorageManager;
    use haystack_embedded::structures::tree::Tree;

    #[test]
    fn test_store_and_load_node() {
        let mut storage_manager: StorageManager<i32, String> = StorageManager::new();

        let mut node = Node::new_leaf(0);
        node.keys.push(1);
        node.values.push(Some("one".to_string()));

        // Store the node
        let offset = storage_manager.store_node(&mut node).unwrap();
        assert_eq!(offset, 0); // Check that the node is stored at the correct offset

        // Load the node
        let loaded_node = storage_manager.load_node(offset).unwrap();
        assert_eq!(loaded_node.keys, vec![1]);
        assert_eq!(loaded_node.values, vec![Some("one".to_string())]);
    }

    #[test]
    fn test_store_multiple_nodes() {
        let mut storage_manager: StorageManager<i32, String> = StorageManager::new();

        let mut node1 = Node::new_leaf(0);
        node1.keys.push(1);
        node1.values.push(Some("one".to_string()));

        let mut node2 = Node::new_leaf(0);
        node2.keys.push(2);
        node2.values.push(Some("two".to_string()));

        // Store the first node
        let offset1 = storage_manager.store_node(&mut node1).unwrap();
        assert_eq!(offset1, 0);

        // Store the second node
        let offset2 = storage_manager.store_node(&mut node2).unwrap();
        assert!(offset2 > offset1); // Ensure that the second node is stored after the first

        // Load the first node
        let loaded_node1 = storage_manager.load_node(offset1).unwrap();
        assert_eq!(loaded_node1.keys, vec![1]);
        assert_eq!(loaded_node1.values, vec![Some("one".to_string())]);

        // Load the second node
        let loaded_node2 = storage_manager.load_node(offset2).unwrap();
        assert_eq!(loaded_node2.keys, vec![2]);
        assert_eq!(loaded_node2.values, vec![Some("two".to_string())]);
    }

    // #[test]
    // fn test_resize_storage() {
    //     let path = PathBuf::from_str("tests/data")
    //         .unwrap()
    //         .join(uuid::Uuid::new_v4().to_string())
    //     let mut storage_manager: StorageManager<i32, String> =
    //         StorageManager::new(path.clone()).unwrap();

    //     let mut large_node = Node::new_leaf(0);
    //     for i in 0..1000 {
    //         large_node.keys.push(i);
    //         large_node.values.push(Some(format!("value_{}", i)));
    //     }

    //     // Store the large node
    //     let offset = storage_manager.store_node(&mut large_node).unwrap();
    //     assert_eq!(offset, HEADER_SIZE);

    //     // Load the large node
    //     let loaded_node = storage_manager.load_node(offset).unwrap();
    //     assert_eq!(loaded_node.keys.len(), 1000);
    //     assert_eq!(loaded_node.values.len(), 1000);
    // }

    #[test]
    fn test_new_leaf() {
        let node: Node<i32, String> = Node::new_leaf(0);
        assert!(node.keys.is_empty());
        assert!(node.values.is_empty());
        assert!(node.children.is_empty());
        assert_eq!(node.node_type, NodeType::Leaf);
    }

    #[test]
    fn test_search_in_leaf() {
        let mut tree = Tree::new().expect("Failed to create tree");
        tree.insert(1, "one".to_string()).unwrap();
        tree.insert(2, "two".to_string()).unwrap();
        assert_eq!(tree.search(1).unwrap(), Some("one".to_string()));
        assert_eq!(tree.search(2).unwrap(), Some("two".to_string()));
        assert_eq!(tree.search(3).unwrap(), None);
    }

    #[test]
    fn test_complex_tree_operations() {
        let mut tree = Tree::new().expect("Failed to create tree");
        for i in 0..10 {
            tree.insert(i, format!("value_{}", i)).unwrap();
        }
        assert_eq!(tree.search(5).unwrap(), Some("value_5".to_string()));
        assert_eq!(tree.search(9).unwrap(), Some("value_9".to_string()));
        assert_eq!(tree.search(10).unwrap(), None);
    }

    #[test]
    fn test_serialization_and_deserialization() {
        let mut node: Node<i32, String> = Node::new_leaf(0);
        node.set_key_value(0, "value_0".to_string());
        node.set_key_value(1, "value_1".to_string());
        let serialized = node.serialize();
        let deserialized: Node<i32, String> = Node::deserialize(&serialized);

        assert_eq!(node.keys, deserialized.keys);
        assert_eq!(node.values, deserialized.values);
        assert_eq!(node.children, deserialized.children);
    }

    #[test]
    fn test_tree_initialization() {
        let tree: Result<Tree<i32, String>, HaystackError> = Tree::new();
        assert!(tree.is_ok());
    }

    #[test]
    fn test_insert_search_leaf() {
        let mut tree = Tree::new().expect("Failed to create tree");

        tree.insert(1, "one".to_string()).unwrap();
        tree.insert(2, "two".to_string()).unwrap();

        assert_eq!(tree.search(1).unwrap(), Some("one".to_string()));
        assert_eq!(tree.search(2).unwrap(), Some("two".to_string()));
        assert_eq!(tree.search(3).unwrap(), None);
    }

    // Edge Cases

    #[test]
    fn test_insert_duplicate_keys() {
        let mut tree = Tree::new().expect("Failed to create tree");

        tree.insert(1, "one".to_string()).unwrap();
        tree.insert(1, "one_duplicate".to_string()).unwrap(); // Assuming overwrite behavior

        assert_eq!(tree.search(1).unwrap(), Some("one_duplicate".to_string()));
    }

    #[test]
    fn test_search_non_existent_key() {
        let mut tree: Tree<i32, String> = Tree::new().expect("Failed to create tree");

        assert_eq!(tree.search(999).unwrap(), None);
    }
    // Complex Operations

    #[test]
    fn test_complex_insertions() {
        let mut tree = Tree::new().expect("Failed to create tree");

        for i in 0..100 {
            tree.insert(i, format!("value_{}", i))
                .expect(format!("Failed to insert {}", i).as_str());
        }

        for i in 0..100 {
            assert_eq!(tree.search(i).unwrap(), Some(format!("value_{}", i)));
        }
    }

    #[test]
    fn test_large_scale_insert_search() {
        let mut tree = Tree::new().unwrap();

        let num_items = 1000;
        for i in 0..num_items {
            tree.insert(i, format!("value_{}", i))
                .expect(format!("Failed to insert {}", i).as_str());
        }

        for i in 0..num_items {
            assert_eq!(tree.search(i).unwrap(), Some(format!("value_{}", i)));
        }
    }

    #[test]
    fn test_repeated_insertions_same_key() {
        let mut tree = Tree::new().unwrap();

        tree.insert(1, "one".to_string()).unwrap();
        tree.insert(1, "still_one".to_string()).unwrap(); // Try inserting the same key

        // Check that the value has not been replaced if replacing isn't supported
        assert_eq!(tree.search(1).unwrap(), Some("still_one".to_string()));
    }

    // #[test]
    // fn test_insertion_order_independence() {
    //     let path = PathBuf::from_str("tests/data")
    //         .unwrap()
    //         .join(uuid::Uuid::new_v4().to_string());
    //     let mut tree = Tree::new(path.clone()).unwrap();
    //     let mut tree_reverse = Tree::new().unwrap();

    //     let keys = vec![3, 1, 4, 1, 5, 9, 2];
    //     let values = vec!["three", "one", "four", "one", "five", "nine", "two"];

    //     for (&k, &v) in keys.iter().zip(values.iter()) {
    //         tree.insert(k, v.to_string()).unwrap();
    //     }

    //     for (&k, &v) in keys.iter().zip(values.iter()).rev() {
    //         tree_reverse.insert(k, v.to_string()).unwrap();
    //     }

    //     for &k in &keys {
    //         assert_eq!(tree.search(k).unwrap(), tree_reverse.search(k).unwrap());
    //     }
    // }

    #[test]
    fn test_search_non_existent_keys() {
        let mut tree: Tree<i32, String> = Tree::new().unwrap();

        assert_eq!(tree.search(999).unwrap(), None);
    }

    #[test]
    fn test_insert_search_edge_integers() {
        let mut tree = Tree::new().unwrap();

        let min_int = i32::MIN;
        let max_int = i32::MAX;

        tree.insert(min_int, "minimum".to_string()).unwrap();
        tree.insert(max_int, "maximum".to_string()).unwrap();

        assert_eq!(tree.search(min_int).unwrap(), Some("minimum".to_string()));
        assert_eq!(tree.search(max_int).unwrap(), Some("maximum".to_string()));
    }

    #[test]
    fn test_batch_insert() {
        let mut tree: Tree<i32, String> = Tree::new().expect("Failed to create tree");

        const NUM_ITEMS: usize = 10_000;

        for i in 0..NUM_ITEMS {
            tree.insert(i as i32, format!("value_{}", i))
                .expect(format!("Failed to insert {}", i).as_str());
        }

        for i in 0..NUM_ITEMS {
            assert_eq!(tree.search(i as i32).unwrap(), Some(format!("value_{}", i)));
        }

        let mut tree: Tree<i32, String> = Tree::new().expect("Failed to create tree");

        let entries: Vec<(i32, String)> = (0..NUM_ITEMS)
            .map(|i| (i as i32, format!("value_{}", i)))
            .collect();

        tree.batch_insert(entries).expect("Failed to batch insert");

        for i in 0..NUM_ITEMS {
            assert_eq!(tree.search(i as i32).unwrap(), Some(format!("value_{}", i)));
        }
    }

    #[test]
    fn test_metadata_index_batch_insert() {
        let mut metadata_index = MetadataIndex::new();

        let mut metadata_index_items: Vec<(u128, MetadataIndexItem)> = Vec::new();

        let mut ids = Vec::new();

        for i in 0..1000 {
            let id = uuid::Uuid::new_v4().as_u128();
            ids.push(id);
            let metadata_index_item = MetadataIndexItem {
                id: id,
                kvs: vec![
                    KVPair::new("key".to_string(), "value".to_string()),
                    KVPair::new("index".to_string(), i.to_string()),
                ],
                vector_index: i as usize,
            };

            metadata_index_items.push((id, metadata_index_item));
        }

        // metadata_index.batch_insert(metadata_index_items);

        // works with the above

        // fails with the below

        for (id, metadata_index_item) in metadata_index_items {
            metadata_index.insert(id, metadata_index_item);
        }

        for i in 0..1000 {
            let metadata_index_item = metadata_index.get(ids[i]).unwrap();
            assert_eq!(metadata_index_item.id, ids[i]);
            assert_eq!(metadata_index_item.kvs[1].value, i.to_string());
        }

        assert_eq!(metadata_index.tree.len(), 1000);
    }

    #[test]
    fn test_to_and_from_bytes() {
        let mut tree = Tree::new().unwrap();

        let num_items = 1000;

        for i in 0..num_items {
            tree.insert(i, format!("value_{}", i))
                .expect(format!("Failed to insert {}", i).as_str());
        }

        let bytes = tree.to_binary();

        let mut tree_from_bytes =
            Tree::<i32, String>::from_binary(bytes).expect("Failed to load tree");

        for i in 0..num_items {
            assert_eq!(
                tree_from_bytes.search(i).unwrap(),
                Some(format!("value_{}", i))
            );
        }
    }

    #[test]
    fn test_metadata_index_to_and_from_binary() {
        let mut metadata_index = MetadataIndex::new();

        let mut metadata_index_items: Vec<(u128, MetadataIndexItem)> = Vec::new();

        for i in 0..10000 {
            let metadata_index_item = MetadataIndexItem {
                id: i,
                kvs: vec![KVPair::new("key".to_string(), "value".to_string())],
                vector_index: i as usize,
            };

            metadata_index_items.push((i, metadata_index_item));
        }

        metadata_index.batch_insert(metadata_index_items);

        let bytes = metadata_index.to_binary();

        let mut metadata_index_from_bytes = MetadataIndex::from_binary(bytes);

        for i in 0..10000 {
            let metadata_index_item = metadata_index_from_bytes.get(i).unwrap();
            assert_eq!(metadata_index_item.id, i);
        }
    }

    #[test]
    fn test_namespace_state_to_and_from_binary() {
        let mut namespace_state = NamespaceState::new("test".to_string());

        // add vectors, metadata, and inverted indices

        let vectors: Vec<[u8; 128]> = (0..1000).map(|_| [0u8; 128]).collect();
        let metadata: Vec<(u128, MetadataIndexItem)> = (0..1000)
            .map(|_| {
                (
                    0,
                    MetadataIndexItem {
                        id: 0,
                        kvs: vec![KVPair::new("key".to_string(), "value".to_string())],
                        vector_index: 0,
                    },
                )
            })
            .collect();
        let inverted_indices = InvertedIndexItem {
            ids: vec![0],
            indices: vec![0],
        };

        namespace_state.vectors.batch_push(vectors).unwrap();
        namespace_state.metadata_index.batch_insert(metadata);
        namespace_state.inverted_index.insert_append(
            KVPair::new("key".to_string(), "value".to_string()),
            inverted_indices,
        );

        let bytes = namespace_state.save_state();

        let mut namespace_state_from_bytes = NamespaceState::new("test".to_string());

        namespace_state_from_bytes.load_state(bytes);

        assert_eq!(
            namespace_state_from_bytes
                .inverted_index
                .get(KVPair::new("key".to_string(), "value".to_string()))
                .unwrap()
                .ids,
            vec![0]
        );

        assert_eq!(
            namespace_state_from_bytes
                .metadata_index
                .get(0)
                .unwrap()
                .kvs
                .get(0)
                .unwrap()
                .value,
            "value"
        );

        assert_eq!(
            namespace_state_from_bytes
                .vectors
                .get(0)
                .unwrap()
                .iter()
                .map(|&x| x)
                .collect::<Vec<u8>>(),
            vec![0u8; 128]
        );
    }

    #[test]
    fn test_complex_namespace_state_to_and_from_binary() {
        let mut namespace_state = NamespaceState::new("test".to_string());

        // add vectors, metadata, and inverted indices

        let vectors: Vec<[u8; 128]> = (0..1000).map(|_| [0u8; 128]).collect();
        let metadata: Vec<(u128, MetadataIndexItem)> = (0..1000)
            .map(|i| {
                (
                    i,
                    MetadataIndexItem {
                        id: 0,
                        kvs: vec![KVPair::new("key".to_string(), "value".to_string())],
                        vector_index: 0,
                    },
                )
            })
            .collect();

        let vector_indices = namespace_state.vectors.batch_push(vectors).unwrap();

        let inverted_indices = InvertedIndexItem {
            ids: vec![0; 1000],
            indices: vector_indices,
        };

        namespace_state.metadata_index.batch_insert(metadata);

        namespace_state.inverted_index.insert_append(
            KVPair::new("key".to_string(), "value".to_string()),
            inverted_indices,
        );

        let bytes = namespace_state.save_state();

        let mut namespace_state_from_bytes = NamespaceState::new("test".to_string());

        namespace_state_from_bytes.load_state(bytes);

        assert_eq!(
            namespace_state_from_bytes
                .inverted_index
                .get(KVPair::new("key".to_string(), "value".to_string()))
                .unwrap()
                .ids,
            vec![0; 1000]
        );

        assert_eq!(
            namespace_state_from_bytes
                .metadata_index
                .get(0)
                .unwrap()
                .kvs
                .get(0)
                .unwrap()
                .value,
            "value"
        );

        assert_eq!(
            namespace_state_from_bytes
                .vectors
                .get(0)
                .unwrap()
                .iter()
                .map(|&x| x)
                .collect::<Vec<u8>>(),
            vec![0u8; 128]
        );

        assert_eq!(
            namespace_state_from_bytes
                .vectors
                .get(999)
                .unwrap()
                .iter()
                .map(|&x| x)
                .collect::<Vec<u8>>(),
            vec![0u8; 128]
        );

        assert_eq!(
            namespace_state_from_bytes
                .metadata_index
                .get(999)
                .unwrap()
                .kvs
                .get(0)
                .unwrap()
                .value,
            "value"
        );

        assert_eq!(
            namespace_state_from_bytes
                .inverted_index
                .get(KVPair::new("key".to_string(), "value".to_string()))
                .unwrap()
                .ids,
            vec![0; 1000]
        );
    }
}
