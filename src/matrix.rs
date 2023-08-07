use std::slice::{ChunksMut, Chunks};
use std::ops::{Index, IndexMut};


#[derive(Debug)]
pub struct Matrix {
	cols: usize,
	rows: usize,
	data: Vec<f32>,
}

// trait to enable overload of the recycle function for the matrix
pub trait MatrixRecycle<R> {
	fn recycle_with_value(resource: R, value: f32, size: MatrixSizeParam) -> Self;
}

/// A tuple, such that: (collumns, rows)
type MatrixSizeParam = (usize, usize);

// Construction & Destruction
impl Matrix {
	/// Make a matrix with all items initialized to a specific value
	pub fn with_value(value: f32, size: MatrixSizeParam) -> Self {
		Self::recycle_with_value(vec![], value, size)
	}

	/// Make a matrix with all zeros
	pub fn zeros(size: MatrixSizeParam) -> Self {
		Self::with_value(0.0, size)
	}

	/// Make a matrix with all ones
	pub fn ones(size: MatrixSizeParam) -> Self {
		Self::with_value(1.0, size)
	}

	/// Make a matrix with items initialized in a vector
	pub fn from_vec(data: Vec<f32>, size: MatrixSizeParam) -> Self {
		let (cols, rows) = size;
		// assert!(cols != 0 && rows != 0, "Attempted to make a matrix with 0 dimensions: {:?}", (cols, rows));
		assert!(!data.is_empty(), "Attempted to make a matrix from an empty vector");
		assert!(cols * rows == data.len(), "Size of the matrix must equal the vector's length. Cols: {}, Rows: {}, Items: {}", cols, rows, data.len());
		Self {
			cols,
			rows,
			data,
		}
	}

	/// Deconstruct the matrix into a tuple (V, (C, R)) so that:
	/// V is a vector with the data,
	/// C is the number of collumns in the matrix,
	/// R is the number of rows in the matrix.
	pub fn into_vec(self) -> (Vec<f32>, (usize, usize)) {
		(self.data, (self.cols, self.rows))
	}
}

impl MatrixRecycle<Vec<f32>> for Matrix {
	/// Recycle a vector to construct a matrix
	fn recycle_with_value(mut data: Vec<f32>, value: f32, size: MatrixSizeParam) -> Self {
		let (cols, rows) = size;
		assert!(cols != 0 && rows != 0, "Attempted to make a matrix with 0 dimensions: {:?}", (cols, rows));
		data.clear();
		data.resize(cols * rows, value);
		Self {
			cols,
			rows,
			data,
		}
	}
}

impl MatrixRecycle<Matrix> for Matrix {
	/// Recycle a matrix to construct another matrix
	fn recycle_with_value(resource: Matrix, value: f32, size: MatrixSizeParam) -> Self {
		Self::recycle_with_value(resource.data, value, size)
	}
}

// Utility
impl Matrix {
	/// Returns a tuple (C, R) so that:
	/// C is the number of collumns in the matrix
	/// R is the number of rows in the matrix
	pub fn size(&self) -> (usize, usize) {
		(self.cols, self.rows)
	}

	/// Returns an iterator to the collumns (which are continguos in memory), so that the items are immutable
	pub fn collumns(&self) -> Chunks<'_, f32> {
		self.data.chunks(self.rows)
	}

	/// Returns an iterator to the collumns (which are continguos in memory), so that the items are mutable
	pub fn collumns_mut(&mut self) -> ChunksMut<'_, f32> {
		self.data.chunks_mut(self.rows)
	}

	/// Checks if the struct is valid, by checking if rows * cols == data.len()
	pub fn validate(&self) -> bool {
		self.cols * self.rows == self.data.len()
	}
}

// Math
impl Matrix {
	/// Transforms a matrix with the parameter, which is a transformation/function matrix.
	/// Returns a mutable reference to the transformed matrix to enable chaining of transformations
	pub fn transform_with(&mut self, transformation: &Self) -> &mut Self {
		assert!(transformation.cols == self.rows, "Tried transforming a vector with number of rows isn't  equal to the transformation matrix's number of collumns. Matrix: {:?}, Transformation: {:?}", self.size(), transformation.size());

		// Step 1: efficient resource management for cache coherence
		let original_len = self.data.len();
		// use the extra capacity in the matrix as a buffer for work
		self.data.reserve(transformation.data.len());

		// Step 2: matrix multiplication

		// cant use scaler_collumns because of mutabilty conflict.
		// todo: resolve this with interior mutability?
		// let mut scaler_collumns = self.collumns();

		let mut transform_collumns = transformation.collumns();
		// data is continuous, so instead of 2D indexing, 1D indexing is used with an extra counter
		let mut xy = 0;
		for _x in 0..self.cols { // for collumn in scaler_collumns {
			for _y in 0..self.rows { // for scaler in collumn {
				let scaler = self.data[xy]; // self[[_x,_y]];
				xy += 1;
				for t in transform_collumns.next().unwrap() {
					self.data.push(scaler * t);
				}
			}
		}

		// Step 3: sum
		for i in (original_len + self.rows .. original_len + transformation.data.len()).rev() {
			self.data[i-self.rows] += self.data[i];
		}

		// Step 4: copy
		for i in 0..original_len {
			self.data[i] = self.data[i + original_len];
		}

		// Step 5: remove unnesesary data from Step 1
		self.data.truncate(original_len);
		// todo is it safe to use set_len() as such:
		unsafe { self.data.set_len(original_len); }

		// Step 6: Return self for chaning operations
		self
	}
}


// Operators

impl Index<[usize; 2]> for Matrix {
	type Output = f32;
	/// Indexing operator where the index is [Collumn, Row]
	fn index(&self, index: [usize; 2]) -> &Self::Output {
		let [col, row] = index;
		assert!(col < self.cols && row < self.rows, "Out of bounds access to a matrix: {{col:{}, row:{}}}", col, row);
		&self.data[self.rows * col + row]
	}
}

impl IndexMut<[usize; 2]> for Matrix {
	/// Indexing operator where the index is [Collumn, Row]
	fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
		let [col, row] = index;
		assert!(col < self.cols && row < self.rows, "Out of bounds access to a matrix: {{col:{}, row:{}}}", col, row);
		&mut self.data[self.rows * col + row]
	}
}
