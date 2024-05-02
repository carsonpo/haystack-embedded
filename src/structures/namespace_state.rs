use crate::structures::dense_vector_list::DenseVectorList;
use crate::structures::inverted_index::InvertedIndex;
use crate::structures::metadata_index::MetadataIndex;

pub struct NamespaceState {
    pub namespace_id: String,
    pub metadata_index: MetadataIndex,
    pub inverted_index: InvertedIndex,
    pub vectors: DenseVectorList,
}

impl NamespaceState {
    pub fn new(namespace_id: String) -> Self {
        let metadata_index = MetadataIndex::new();
        let inverted_index = InvertedIndex::new();
        let vectors = DenseVectorList::new(100_000);

        NamespaceState {
            namespace_id,
            metadata_index,
            inverted_index,
            vectors,
        }
    }

    pub fn save_state(&mut self) -> Vec<u8> {
        let metadata_index_binary = self.metadata_index.to_binary();
        let inverted_index_binary = self.inverted_index.to_binary();
        let vectors_binary = self.vectors.to_binary();

        let mut state = Vec::new();

        state.extend(metadata_index_binary.len().to_le_bytes().as_ref());
        state.extend(metadata_index_binary);

        state.extend(inverted_index_binary.len().to_le_bytes().as_ref());
        state.extend(inverted_index_binary);

        state.extend(vectors_binary.len().to_le_bytes().as_ref());

        state.extend(vectors_binary);

        state
    }

    pub fn load_state(&mut self, data: Vec<u8>) {
        let mut offset = 0;

        let metadata_index_len =
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        self.metadata_index =
            MetadataIndex::from_binary(data[offset..offset + metadata_index_len].to_vec());
        offset += metadata_index_len;

        let inverted_index_len =
            u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        self.inverted_index =
            InvertedIndex::from_binary(data[offset..offset + inverted_index_len].to_vec());
        offset += inverted_index_len;

        let vectors_len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        self.vectors = DenseVectorList::from_binary(&data[offset..offset + vectors_len].to_vec())
            .expect("Failed to load vectors");
    }
}
