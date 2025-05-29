/// Polynomial fitting via normal equations and Gaussian elimination
pub struct PolynomialFit<T>
where
    Point<T, T>: Clone,
{
    points: Vec<Point<T, T>>,
    coeff: Option<Vec<T>>,
    degree: usize,
}
impl<
        T: Copy
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Default
            + PartialOrd
            + std::ops::Neg<Output = T>
            + One,
    > PolynomialFit<T>
where
    Point<T, T>: Clone,
{
    pub fn new(input_points: Vec<Point<T, T>>, degree: usize) -> Result<Self, FittingError> {
        if input_points.len() <= degree {
            return Err(FittingError::InsufficientData);
        }
        Ok(PolynomialFit {
            points: input_points,
            coeff: None,
            degree,
        })
    }
    pub fn create_design_matrix(&self) -> Result<Matrix<T>, FittingError> {
        let n = self.points.len();
        let m = self.degree + 1;
        let mut mat = Matrix::new(n, m, T::default());
        for (i, p) in self.points.iter().enumerate() {
            let mut pow = T::one();
            for j in 0..m {
                mat.set(i, j, pow)?;
                pow = pow * p.x;
            }
        }
        Ok(mat)
    }
    pub fn create_augmented_matrix(&self) -> Result<Matrix<T>, FittingError> {
        let x = self.create_design_matrix()?;
        let xt = x.calculate_transposed_matrix()?;
        let xtx = xt.multiply_by_second_matrix(x.clone())?;
        // build y vector as Nx1
        let n = self.points.len();
        let mut ymat = Matrix::new(n, 1, T::default());
        for (i, p) in self.points.iter().enumerate() {
            ymat.set(i, 0, p.y)?;
        }
        let xty = xt.multiply_by_second_matrix(ymat)?;
        let r = self.degree + 1;
        let c = r + 1;
        let mut aug = Matrix::new(r, c, T::default());
        for i in 0..r {
            for j in 0..r {
                aug.set(i, j, xtx.get(i, j).unwrap())?;
            }
            aug.set(i, r, xty.get(i, 0).unwrap())?;
        }
        Ok(aug)
    }
    pub fn gaussian_elimination(&self) -> Result<Matrix<T>, FittingError> {
        let mut aug = self.create_augmented_matrix()?;
        aug.to_reduced_row_echelon_form()?;
        Ok(aug)
    }
    pub fn fit_polynomial(&mut self) -> Result<&Vec<T>, FittingError> {
        let reduced = self.gaussian_elimination()?;
        let coeffs = reduced.back_substitute()?;
        self.coeff = Some(coeffs);
        Ok(self.coeff.as_ref().unwrap())
    }
    pub fn calculate_error(&self) -> Result<T, FittingError> {
        if let Some(ref coeffs) = self.coeff {
            let mut err = T::default();
            for p in &self.points {
                let mut pow = T::one();
                let mut pred = T::default();
                for &c in coeffs.iter() {
                    pred = pred + c * pow;
                    pow = pow * p.x;
                }
                let diff = p.y - pred;
                err = err + diff * diff;
            }
            Ok(err)
        } else {
            Err(FittingError::InvalidInput(
                "Call fit_polynomial first".into(),
            ))
        }
    }
}

use std::{
    ops::{Add, Div, Mul, Sub},
    time::Instant,
};

use num_traits::One;

use crate::{FittingError, Matrix, Point};

fn main() {
    println!("Polynomial fitting with Gaussian elimination");

    // Generate synthetic points from y = 1 + 2x + 3x^2
    let mut points = Vec::new();
    for x in 0..1000 {
        let xf = x as f64;
        let y = 1.0 + 2.0 * xf + 3.0 * xf * xf;
        points.push(Point { x: xf, y });
    }

    let degree = 2;
    let start = Instant::now();

    let mut pf = PolynomialFit::new(points, degree).unwrap();
    let coeffs = pf.fit_polynomial().unwrap();

    let duration = start.elapsed();
    println!("Rust fit time: {:.6} secs", duration.as_secs_f64());

    println!("Fitted coefficients:");
    for (i, c) in coeffs.iter().enumerate() {
        println!("  x^{}: {:.6}", i, c);
    }

    println!();
    println!("(For comparison run py_exp.py, which took .0124 seconds. This is a 141x increase in speed for me!)");
}

#[cfg(test)]
mod tests {
    use num_traits::Float;

    use super::*;

    #[test]
    fn test_matrix_reduce_ops_reduction() {
        // Use a flat Vec<i32> for the matrix data
        // Example from book (linear algebra and its applications, example ch 1, eamxple 1, pg 111)
        let mut matrix = Matrix {
            data: vec![36., 51., 13., 33., 52., 34., 74., 45., 0., 7., 1.1, 3.],
            rows: 3,
            cols: 4,
        };

        let result = matrix.to_reduced_row_echelon_form().unwrap();

        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(0, 1).unwrap(), 0.0);
        assert_eq!(matrix.get(0, 3).unwrap(), 0.27722318361443293);
        assert_eq!(matrix.get(1, 1).unwrap(), 1.0);
    }

    #[test]
    fn test_check_row_echelon_form_true() {
        let matrix = Matrix {
            data: vec![1.0, 2.0, 0.0, 3.0],
            rows: 2,
            cols: 2,
        };
        assert!(matrix.check_row_echelon_form());
    }

    #[test]
    fn test_check_row_echelon_form_false() {
        let matrix = Matrix {
            data: vec![1.0, 0.0, 1.0, 2.0],
            rows: 2,
            cols: 2,
        };
        assert!(!matrix.check_row_echelon_form());
    }

    #[test]
    fn test_matrix_reduce_ops() {
        // Use a flat Vec<i32> for the matrix data
        let mut matrix = Matrix {
            data: vec![4.0, 5.0, 6.0, 5.0, 8.0, 4.0],
            rows: 3,
            cols: 2,
        };

        let result = matrix.find_pivot(0, 0).unwrap();

        assert_eq!(result, 2);

        let _ = matrix.normalize_row(2, 0);

        assert_eq!(matrix.get(2, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(2, 1).unwrap(), 0.5);
    }

    #[test]
    fn test_matrix_multiplication() {
        // Use a flat Vec<i32> for the matrix data
        let mut initial_matrix = Matrix {
            data: vec![1, 2, 3, 3, 2, 1, 1, 2, 3],
            rows: 3,
            cols: 3,
        };

        let mut second_matrix = Matrix {
            data: vec![4, 5, 6, 5, 8, 2],
            rows: 3,
            cols: 2,
        };

        let result = initial_matrix
            .multiply_by_second_matrix(second_matrix)
            .unwrap();

        assert_eq!(result.shape(), (3, 2));

        assert_eq!(result.get(1, 1).unwrap(), 27);
    }

    #[test]
    fn test_basic_matrix_operations() {
        // Use a flat Vec<i32> for the matrix data
        let mut initial_matrix = Matrix {
            data: vec![1, 2, 3, 4, 5, 6],
            rows: 2,
            cols: 3,
        };

        assert_eq!(initial_matrix.get(1, 2).unwrap(), 6);
        let _ = initial_matrix.set(1, 2, 10);
        assert_eq!(initial_matrix.get(1, 2).unwrap(), 10);

        let new_matrix = initial_matrix.swap_rows(1, 0).unwrap();
        assert_eq!(new_matrix.get(0, 1).unwrap(), 5);
    }

    #[test]
    fn test_calculate_transpose() {
        // Use a flat Vec<i32> for the matrix data
        let initial_matrix = Matrix {
            data: vec![1, 2, 3, 4, 5, 6],
            rows: 2,
            cols: 3,
        };
        let result = initial_matrix.calculate_transposed_matrix().unwrap();

        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result.get(0, 1).unwrap(), 4);
    }

    #[test]
    fn test_polynomial_fit_perfect() {
        let points = vec![
            Point { x: 0.0, y: 1.0 },
            Point { x: 1.0, y: 3.0 },
            Point { x: 2.0, y: 7.0 },
        ];
        let mut pf = PolynomialFit::new(points, 2).unwrap();
        let coeffs = pf.fit_polynomial().unwrap();
        assert!((coeffs[0] - 1.0).abs() < 1e-6);
        assert!((coeffs[1] - 1.0).abs() < 1e-6);
        assert!((coeffs[2] - 1.0).abs() < 1e-6);
        assert!((pf.calculate_error().unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_insufficient_data_error() {
        let pts = vec![Point { x: 0.0, y: 1.0 }];
        assert!(PolynomialFit::new(pts, 1).is_err());
    }
}
