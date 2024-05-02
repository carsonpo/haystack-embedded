use crate::constants::QUANTIZED_VECTOR_SIZE;
use crate::errors::HaystackError;
use std::io;

const SIZE_OF_U64: usize = std::mem::size_of::<u64>();

pub struct DenseVectorList {
    data: Vec<[u8; QUANTIZED_VECTOR_SIZE]>,
}

impl DenseVectorList {
    pub fn new(elements: u64) -> Self {
        DenseVectorList {
            data: Vec::with_capacity(elements as usize),
        }
    }

    pub fn push(&mut self, vector: [u8; QUANTIZED_VECTOR_SIZE]) -> Result<usize, HaystackError> {
        self.data.push(vector);

        Ok(self.data.len() - 1)
    }

    pub fn batch_push(
        &mut self,
        vectors: Vec<[u8; QUANTIZED_VECTOR_SIZE]>,
    ) -> Result<Vec<usize>, HaystackError> {
        let start = self.data.len();
        self.data.extend(vectors);

        Ok((start..self.data.len()).collect())
    }

    pub fn get(&self, index: usize) -> Result<&[u8; QUANTIZED_VECTOR_SIZE], HaystackError> {
        let offset = index;
        let end = offset + 1;

        if end > self.data.len() {
            return Err(HaystackError::new("Index out of bounds"));
        }

        Ok(&self.data[index])
    }

    pub fn get_contiguous(
        &self,
        index: usize,
        num_elements: usize,
    ) -> Result<&[[u8; QUANTIZED_VECTOR_SIZE]], HaystackError> {
        let start = index;
        let end = start + num_elements;

        if end > self.data.len() {
            return Err(HaystackError::new("Index out of bounds"));
        }

        // the indices are contiguous, so we can just get a slice of the mmap
        let vectors: &[[u8; QUANTIZED_VECTOR_SIZE]] =
            unsafe { std::slice::from_raw_parts(self.data.as_ptr().add(start), num_elements) };

        Ok(vectors)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn insert(
        &mut self,
        index: usize,
        vector: [u8; QUANTIZED_VECTOR_SIZE],
    ) -> Result<(), HaystackError> {
        if index > self.data.len() {
            return Err(HaystackError::new("Index out of bounds"));
        }

        self.data.insert(index, vector);

        Ok(())
    }

    pub fn to_binary(&self) -> Vec<u8> {
        let mut serialized = Vec::new();

        serialized.extend_from_slice((self.data.len() as u64).to_le_bytes().as_ref());

        for vector in &self.data {
            serialized.extend_from_slice(vector);
        }

        serialized
    }

    pub fn from_binary(data: &[u8]) -> Result<Self, HaystackError> {
        let mut offset = 0;

        let num_elements =
            u64::from_le_bytes(data[offset..offset + SIZE_OF_U64].try_into().unwrap()) as usize;
        offset += SIZE_OF_U64;

        let mut dense_vector_list = DenseVectorList::new(num_elements as u64);

        for _ in 0..num_elements {
            let mut vector = [0; QUANTIZED_VECTOR_SIZE];
            vector.copy_from_slice(&data[offset..offset + QUANTIZED_VECTOR_SIZE]);
            offset += QUANTIZED_VECTOR_SIZE;

            dense_vector_list.push(vector)?;
        }

        Ok(dense_vector_list)
    }
}
