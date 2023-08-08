mod matrix;

use matrix::Matrix;

fn main() {
	let m = Matrix::new(&[
		[1., 2., 3.],
		[4., 5., 6.],
	]);

	println!("{:?}", m);
}