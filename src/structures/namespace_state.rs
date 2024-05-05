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
        let vectors = DenseVectorList::new(1000);

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

    pub fn load_state(&mut self, data: Vec<u8>) -> Result<(), String> {
        let mut offset = 0;

        // Helper function to safely extract usize from data
        fn extract_length(data: &[u8], start: usize) -> Result<usize, String> {
            data.get(start..start + 8)
                .ok_or_else(|| "Data slice error".to_string())
                .and_then(|slice| {
                    slice
                        .try_into()
                        .map_err(|_| "Conversion error".to_string())
                        .and_then(|bytes: [u8; 8]| Ok(u64::from_le_bytes(bytes) as usize))
                })
        }

        let metadata_index_len = extract_length(&data, offset)?;
        offset += 8;

        let metadata_index_data = data
            .get(offset..offset + metadata_index_len)
            .ok_or("Metadata index slice error")?;
        self.metadata_index = MetadataIndex::from_binary(metadata_index_data.to_vec());
        offset += metadata_index_len;

        let inverted_index_len = extract_length(&data, offset)?;
        offset += 8;

        let inverted_index_data = data
            .get(offset..offset + inverted_index_len)
            .ok_or("Inverted index slice error")?;
        self.inverted_index = InvertedIndex::from_binary(inverted_index_data.to_vec());
        offset += inverted_index_len;

        let vectors_len = extract_length(&data, offset)?;
        offset += 8;

        let vectors_data = data
            .get(offset..offset + vectors_len)
            .ok_or("Vectors slice error")?;
        self.vectors = DenseVectorList::from_binary(vectors_data)
            .map_err(|_| "Failed to load vectors".to_string())?;

        Ok(())
    }
}
