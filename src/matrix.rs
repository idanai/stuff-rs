use std::slice::{ChunksMut, Chunks};
use std::ops::{Index, IndexMut, AddAssign, Add, SubAssign, Sub, MulAssign, Mul};


#[derive(Debug, Clone)]
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
	/// Make a matrix from a slice of arrays, where each array is a collumn in the matrix.
	#[must_use]
	pub fn new<const NRows: usize>(values: &[[f32; NRows]]) -> Self {
		assert!(NRows > 0 && values.len() > 0, "Attempted to create a zero dimensional matrix");

		Self {
			cols: values.len(),
			rows: NRows,
			data: values.iter().flatten().copied().collect(),
		}
	}

	/// Make a matrix with all items initialized to a specific value
	#[must_use]
	pub fn with_value(value: f32, size: MatrixSizeParam) -> Self {
		Self::recycle_with_value(vec![], value, size)
	}

	/// Make a matrix with all zeros
	#[must_use]
	pub fn zeros(size: MatrixSizeParam) -> Self {
		Self::with_value(0.0, size)
	}

	/// Make a matrix with all ones
	#[must_use]
	pub fn ones(size: MatrixSizeParam) -> Self {
		Self::with_value(1.0, size)
	}

	/// Make a matrix with items initialized in a vector
	#[must_use]
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
	#[must_use]
	pub fn into_vec(self) -> (Vec<f32>, (usize, usize)) {
		(self.data, (self.cols, self.rows))
	}

	pub fn into_identity(&mut self, len: usize) {
		self.cols = len;
		self.rows = len;
		// self.data.reserve(len * len);
		self.data.clear();
		self.data.resize(len * len, 0.0);
		self.data.iter_mut().step_by(len + 1).for_each(|x| *x = 1.0);
	}
}

impl MatrixRecycle<Vec<f32>> for Matrix {
	/// Recycle a vector to construct a matrix
	#[must_use]
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
	#[must_use]
	fn recycle_with_value(resource: Matrix, value: f32, size: MatrixSizeParam) -> Self {
		Self::recycle_with_value(resource.data, value, size)
	}
}

// Utility
impl Matrix {
	/// Returns a tuple (C, R) so that:
	/// C is the number of collumns in the matrix
	/// R is the number of rows in the matrix
	#[must_use]
	pub fn size(&self) -> (usize, usize) {
		(self.cols, self.rows)
	}

	/// Returns an iterator to the collumns (which are continguos in memory), so that the items are immutable
	#[must_use]
	pub fn collumns(&self) -> Chunks<'_, f32> {
		self.data.chunks(self.rows)
	}

	/// Returns an iterator to the collumns (which are continguos in memory), so that the items are mutable
	#[must_use]
	pub fn collumns_mut(&mut self) -> ChunksMut<'_, f32> {
		self.data.chunks_mut(self.rows)
	}

	/// Checks if the struct is valid, by checking if rows * cols == data.len()
	#[must_use]
	pub fn validate(&self) -> bool {
		self.cols * self.rows == self.data.len()
	}

	#[must_use]
	fn zip<'a, 'b>(&'a self, rhs: &'b Self) -> impl Iterator<Item = (&'a f32, &'b f32)> {
		self.data.iter().zip(rhs.data.iter())
	}

	#[must_use]
	fn zip_mut<'a, 'b>(&'a mut self, rhs: &'b Self) -> impl Iterator<Item = (&'a mut f32, &'b f32)> {
		self.data.iter_mut().zip(rhs.data.iter())
	}

	#[must_use]
	fn zip_mut_2<'a, 'b>(&'a mut self, rhs: &'b mut Self) -> impl Iterator<Item = (&'a mut f32, &'b mut f32)> {
		self.data.iter_mut().zip(rhs.data.iter_mut())
	}
}

// Math
impl Matrix {
	/// Transforms a matrix with the parameter, which is a transformation/function matrix.
	/// 
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

	/// Matrix multiplication.
	///
	/// Uses 'self' as a transformation function applied to the parameter.
	///
	/// Unlike transform_with(), this is more concise with the mathematical notation,
	/// even though they are the same computationally.
	pub fn transform<'a>(&self, m: &'a mut Self) -> &'a mut Self {
		m.transform_with(&self)
	}

	/// Scale a matrix by a number
	pub fn scale(&mut self, scaler: f32) -> &mut Self {
		self.data.iter_mut().for_each(|x| *x *= scaler);
		self
	}

	#[must_use]
	pub fn pow(mut self, power: usize)-> Self {
		assert!(self.cols == self.rows, "Attempted raising a non-square matrix to a power");
		match power {
			0 => self.into_identity(self.cols),
			1 => {}
			_ => {
				// clone the matrix into a work buffer at the end of the matrix
				let original_len = self.data.len();
				for i in 0..original_len {
					self.data.push(self.data[i]);
				}
				// this is quite naive, so...
				// todo improve me!
				let clone = self.clone();
				for _ in 2..=power {
					self *= &clone;
				}
				// remove the work buffer from the end of the matrix
				self.data.truncate(original_len);
			}
		}
		self
	}

	#[must_use]
	pub fn transpose(mut self) -> Self {
		if self.cols != 1 && self.rows != 1 {
			let original_len = self.data.len();
			self.data.reserve(original_len);

			for row in 0..self.rows {
				for col in 0..self.cols {
					self.data.push(self[[col, row]]);
				}
			}

			for i in 0..original_len {
				self.data[i] = self.data[i + original_len];
			}

			self.data.truncate(original_len);
		}
		std::mem::swap(&mut self.cols, &mut self.rows);
		self
	}
}


// Operators

impl Index<[usize; 2]> for Matrix {
	type Output = f32;
	/// Indexing operator where the index is [Collumn, Row]
	#[must_use]
	fn index(&self, index: [usize; 2]) -> &Self::Output {
		let [col, row] = index;
		assert!(col < self.cols && row < self.rows, "Out of bounds access to a matrix: {{col:{}, row:{}}}", col, row);
		&self.data[self.rows * col + row]
	}
}

impl IndexMut<[usize; 2]> for Matrix {
	/// Indexing operator where the index is [Collumn, Row]
	#[must_use]
	fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
		let [col, row] = index;
		assert!(col < self.cols && row < self.rows, "Out of bounds access to a matrix: {{col:{}, row:{}}}", col, row);
		&mut self.data[self.rows * col + row]
	}
}

impl AddAssign<&Matrix> for Matrix {
	fn add_assign(&mut self, rhs: &Matrix) {
		assert!(self.cols == rhs.cols && self.rows == rhs.rows,
			"Attempted adding 2 matrices of different sizes. Lhs: {:?}, Rhs: {:?}",
			self.size(), rhs.size());

		self.zip_mut(&rhs).for_each(|(a, b)| *a += b);
	}
}

impl Add<&Matrix> for Matrix {
	type Output = Matrix;
	#[must_use]
	fn add(mut self, rhs: &Matrix) -> Self::Output {
		self += rhs;
		self
	}
}

impl SubAssign<&Matrix> for Matrix {
	fn sub_assign(&mut self, rhs: &Matrix) {
		assert!(self.cols == rhs.cols && self.rows == rhs.rows,
			"Attempted subtracting 2 matrices of different sizes. Lhs: {:?}, Rhs: {:?}",
			self.size(), rhs.size());
		self.zip_mut(&rhs).for_each(|(a,b)| *a -= b);
	}
}

impl Sub<&Matrix> for Matrix {
	type Output = Matrix;
	#[must_use]
	fn sub(mut self, rhs: &Matrix) -> Self::Output {
		self -= rhs;
		self
	}
}

impl MulAssign<f32> for Matrix {
	/// Scale a matrix by a number
	fn mul_assign(&mut self, scaler: f32) {
		self.scale(scaler);
	}
}

impl Mul<f32> for Matrix {
	type Output = Self;
	/// Returns a scaled matrix
	#[must_use]
	fn mul(mut self, scaler: f32) -> Self::Output {
		self *= scaler;
		self
	}
}

impl MulAssign<&Matrix> for Matrix {
	/// Matrix multiplication. Shorthand for calling self.transform_with(&transformation_matrix)
	fn mul_assign(&mut self, rhs: &Matrix) {
		self.transform_with(&rhs);
	}
}

impl Mul<&Matrix> for Matrix {
	type Output = Self;
	/// Matrix multiplication.
	/// 
	/// Note: consumes the left hand matrix.
	/// 
	/// Note: this is different than mathematical notation of matrix multiplication:
	/// here it is ordered from left to right- so that the matrix on left is transformed by the one next to it,
	/// on the right, returning a matrix that will be transformed by the one on it's right side, and so on...
	#[must_use]
	fn mul(mut self, rhs: &Matrix) -> Self::Output {
		self *= rhs;
		self
	}
}