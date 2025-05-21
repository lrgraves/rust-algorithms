use num_traits::One;
use std::{
    fmt::{self, Error},
    ops::{Add, Div, Mul, Sub},
    vec,
};

/// Custom error type for polynomial fitting operations
#[derive(Debug)]
enum FittingError {
    InvalidDimensions(String),
    SingularMatrix,
    InsufficientData,
    NumericalInstability,
    InvalidInput(String),
}
impl fmt::Display for FittingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FittingError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            FittingError::SingularMatrix => write!(f, "Matrix is singular and cannot be inverted"),
            FittingError::InsufficientData => write!(
                f,
                "Insufficient data points for the requested polynomial degree"
            ),
            FittingError::NumericalInstability => {
                write!(f, "Numerical instability detected during computation")
            }
            FittingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

/// A data point which is used in the  fitting.
impl std::error::Error for FittingError {}
#[derive(Debug, Clone, PartialEq)]
struct Point<T, U> {
    x: T,
    y: U,
}

/// Matrix implementation with efficient storage and vector flat format (speed + efficiency!)
#[derive(Debug, Clone, PartialEq)]
struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Copy + Default> Matrix<T> {
    /// Creates a new matrix with the specified dimensions
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `default_value` - Default value to fill the matrix with
    fn new(rows: usize, cols: usize, default_value: T) -> Self {
        Matrix {
            data: vec![default_value; rows * cols],
            rows,
            cols,
        }
    }

    /// Gets the value at the specified row and column
    ///
    /// # Arguments
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    fn get(&self, row: usize, col: usize) -> Option<T> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Sets the value at the specified row and column
    ///
    /// # Arguments
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    /// * `value` - The value to set
    fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), FittingError> {
        if row < self.rows && col < self.cols {
            self.data[row * self.cols + col] = value;
            Ok(())
        } else {
            Err(FittingError::InvalidDimensions(format!(
                "Index ({}, {}) out of bounds for matrix {}x{}",
                row, col, self.rows, self.cols
            )))
        }
    }

    /// Returns the shape of the matrix as (rows, cols)
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl<T: Copy + Default> Matrix<T> {
    /// This function calculates the transpose of a matrix A. This involves:
    /// 1) Reflecting A over its main diagonal
    /// 2) Write the rows of A as the columns of A_t
    /// 3) Write the columns of A as the rows of A_t
    /// Checks: The shape should flip, and a direct check of values
    fn calculate_transposed_matrix(&self) -> Result<Matrix<T>, FittingError> {
        let mut result = Matrix::new(self.cols, self.rows, T::default());

        for row in 0..self.rows {
            for col in 0..self.cols {
                let value = self.get(row, col).unwrap();
                result.set(col, row, value)?;
            }
        }
        Ok(result)
    }

    /// Utilize the naive method of O(N^3) complexity, as may have non-square matrices. Can speed up later.
    fn multiply_by_second_matrix(&self, second_matrix: Matrix<T>) -> Result<Matrix<T>, FittingError>
    where
        T: Mul<Output = T> + Add<Output = T> + Copy + Default,
    {
        // Check if dimensions are compatible for multiplication
        if self.cols != second_matrix.rows {
            return Err(FittingError::InvalidDimensions(format!(
                "Cannot multiply {}x{} matrix with {}x{} matrix",
                self.rows, self.cols, second_matrix.rows, second_matrix.cols
            )));
        }
        // Calculate the matrix multiplication. A' = &self * second_matrix
        let mut result = Matrix::new(self.rows, second_matrix.cols, T::default());

        for row in 0..self.rows {
            for col in 0..second_matrix.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self.get(row, k).unwrap() * second_matrix.get(k, col).unwrap();
                }
                result.set(row, col, sum)?;
            }
        }

        Ok(result)
    }
}

impl<
        T: Copy
            + PartialOrd
            + Add<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Default
            + std::ops::Neg<Output = T>,
    > Matrix<T>
{
    /// Finds the pivot (row with maximum absolute value) in a specific column
    ///
    /// # Arguments
    /// * `column` - The column to search in
    /// * `start_row` - The row to start searching from
    ///
    /// # Returns
    /// The index of the row containing the maximum value
    fn find_pivot(&self, column: usize, start_row: usize) -> Result<usize, FittingError> {
        if column >= self.cols || start_row >= self.rows {
            return Err(FittingError::InvalidDimensions(format!(
                "Column {} or start row {} out of bounds",
                column, start_row
            )));
        }

        let mut max_value = self.get(start_row, column);
        let mut index = start_row;

        for row in start_row + 1..self.rows {
            let value = self.get(row, column);
            if value > max_value {
                max_value = value;
                index = row;
            }
        }

        Ok(index)
    }

    /// Swaps two rows in the matrix
    ///
    /// # Arguments
    /// * `row_to_swap_to` - First row index
    /// * `row_to_swap_from` - Second row index
    ///
    /// # Returns
    /// Self reference for method chaining
    fn swap_rows(
        &mut self,
        row_to_swap_to: usize,
        row_to_swap_from: usize,
    ) -> Result<&mut Self, FittingError> {
        if row_to_swap_to >= self.rows || row_to_swap_from >= self.rows {
            return Err(FittingError::InvalidDimensions(format!(
                "Row indices {} and {} out of bounds",
                row_to_swap_to, row_to_swap_from
            )));
        }

        // I think this is a bad solution, as we should leverage rusts slice function, but I need to read this more to figure it out.
        let mut temp_row = Vec::with_capacity(self.cols);

        for i in 0..self.cols {
            temp_row.push(self.get(row_to_swap_from, i).unwrap().clone());
            self.set(row_to_swap_from, i, self.get(row_to_swap_to, i).unwrap())?;
            self.set(row_to_swap_to, i, temp_row[i]);
        }

        Ok(self)
    }

    /// Normalizes a row by dividing all elements by the pivot element
    ///
    /// # Arguments
    /// * `row` - The row to normalize
    /// * `pivot_col` - The column containing the pivot element
    ///
    /// # Returns
    /// Self reference for method chaining
    fn normalize_row(&mut self, row: usize, pivot_col: usize) -> Result<&mut Self, FittingError> {
        // Implementation would normalize the row, again, rust slice method would work better
        // I think this is a bad solution, as we should leverage rusts slice function, but I need to read this more to figure it out.
        let norm_value = self.get(row, pivot_col).unwrap(); // I also dont love that I am unwrapping options all over the place. This is bad rust practice.

        for i in 0..self.cols {
            self.set(row, i, self.get(row, i).unwrap() / norm_value);
        }

        Ok(self)
    }

    /// Eliminates elements below the pivot (forward elimination). Assumes already normalized.
    ///
    /// # Arguments
    /// * `pivot_row` - The row containing the pivot
    /// * `pivot_col` - The column containing the pivot
    ///
    /// # Returns
    /// Self reference for method chaining
    fn eliminate_below(
        &mut self,
        pivot_row: usize,
        pivot_col: usize,
    ) -> Result<&mut Self, FittingError> {
        // Implementation would eliminate elements below pivot

        for i in pivot_row + 1..self.rows {
            let scale_factor =
                self.get(i, pivot_col).unwrap() / self.get(pivot_row, pivot_col).unwrap();

            for j in pivot_col..self.cols {
                let source_value = self.get(pivot_row, j).unwrap();
                self.set(
                    i,
                    j,
                    self.get(i, j).unwrap() + (source_value * scale_factor).neg(),
                );
            }
        }

        Ok(self)
    }

    /// Eliminates elements above the pivot (backward elimination)
    ///
    /// # Arguments
    /// * `pivot_row` - The row containing the pivot
    /// * `pivot_col` - The column containing the pivot
    ///
    /// # Returns
    /// Self reference for method chaining
    fn eliminate_above(
        &mut self,
        pivot_row: usize,
        pivot_col: usize,
    ) -> Result<&mut Self, FittingError> {
        for i in 0..pivot_row {
            let scale_factor =
                self.get(i, pivot_col).unwrap() / self.get(pivot_row, pivot_col).unwrap();

            for j in pivot_col..self.cols {
                let source_value = self.get(pivot_row, j).unwrap();
                self.set(
                    i,
                    j,
                    self.get(i, j).unwrap() + (source_value * scale_factor).neg(),
                )?;
            }
        }

        Ok(self)
    }

    /// Converts matrix to row echelon form (higher-level method)
    ///
    /// # Returns
    /// Self reference for method chaining
    fn to_row_echelon_form(&mut self) -> Result<&mut Self, FittingError> {
        // Loop through each column (for a square matrix)
        let min_dim = std::cmp::min(self.rows, self.cols - 1); // -1 for augmented matrix

        for i in 0..min_dim {
            // Find pivot
            let pivot_row = self.find_pivot(i, i)?;

            // Swap rows if necessary
            if pivot_row != i {
                self.swap_rows(i, pivot_row)?;
            }

            // Normalize the pivot row
            self.normalize_row(i, i)?;

            // Eliminate below
            self.eliminate_below(i, i)?;
        }

        Ok(self)
    }

    /// Converts matrix to reduced row echelon form (higher-level method)
    ///
    /// # Returns
    /// Self reference for method chaining
    fn to_reduced_row_echelon_form(&mut self) -> Result<&mut Self, FittingError> {
        // First get to row echelon form
        self.to_row_echelon_form()?;

        // Then eliminate above
        let min_dim = std::cmp::min(self.rows, self.cols - 1);

        for i in (0..min_dim).rev() {
            self.eliminate_above(i, i)?;
        }

        Ok(self)
    }

    /// Checks if matrix is in row echelon form
    ///
    /// # Returns
    /// true if in row echelon form, false otherwise
    fn check_row_echelon_form(&self) -> bool {
        // Implementation checks if matrix is in row echelon form
        let mut prev_lead: Option<usize> = None;
        let mut seen_zero_row = false;
        for row in 0..self.rows {
            let mut lead_idx = None;
            for col in 0..self.cols {
                if self.get(row, col).unwrap() != T::default() {
                    lead_idx = Some(col);
                    break;
                }
            }
            if let Some(idx) = lead_idx {
                if seen_zero_row {
                    return false;
                }
                if let Some(prev_idx) = prev_lead {
                    if idx <= prev_idx {
                        return false;
                    }
                }
                prev_lead = Some(idx);
            } else {
                seen_zero_row = true;
            }
        }
        true
    }

    /// Performs back substitution to solve the system
    ///
    /// # Returns
    /// Vector of solutions
    fn back_substitute(&self) -> Result<Vec<T>, FittingError> {
        if !self.check_row_echelon_form() {
            return Err(FittingError::InvalidInput("Must be echelon form".into()));
        }
        let mut coeff = Vec::with_capacity(self.cols - 1);
        // last column is RHS, so for cols elements, row i, coeff index i = self.get(i, last)
        let last = self.cols - 1;
        for i in 0..self.rows {
            coeff.push(self.get(i, last).unwrap());
        }
        Ok(coeff)
    }
}

/// Polynomial fitting via normal equations and Gaussian elimination
struct PolynomialFit<T>
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
    fn new(input_points: Vec<Point<T, T>>, degree: usize) -> Result<Self, FittingError> {
        if input_points.len() <= degree {
            return Err(FittingError::InsufficientData);
        }
        Ok(PolynomialFit {
            points: input_points,
            coeff: None,
            degree,
        })
    }
    fn create_design_matrix(&self) -> Result<Matrix<T>, FittingError> {
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
    fn create_augmented_matrix(&self) -> Result<Matrix<T>, FittingError> {
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
    fn gaussian_elimination(&self) -> Result<Matrix<T>, FittingError> {
        let mut aug = self.create_augmented_matrix()?;
        aug.to_reduced_row_echelon_form()?;
        Ok(aug)
    }
    fn fit_polynomial(&mut self) -> Result<&Vec<T>, FittingError> {
        let reduced = self.gaussian_elimination()?;
        let coeffs = reduced.back_substitute()?;
        self.coeff = Some(coeffs);
        Ok(self.coeff.as_ref().unwrap())
    }
    fn calculate_error(&self) -> Result<T, FittingError> {
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

use std::time::Instant;

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
