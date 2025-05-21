use std::{
    fmt::{self, Error},
    ops::{Add, Div, Mul},
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
        // Implementation would check if matrix is in row echelon form
        todo!()
    }

    /// Performs back substitution to solve the system
    ///
    /// # Returns
    /// Vector of solutions
    fn back_substitute(&self) -> Result<Vec<T>, FittingError> {
        // Check if matrix is in row echelon form
        if !self.check_row_echelon_form() {
            return Err(FittingError::InvalidInput(
                "Matrix must be in row echelon form for back substitution".to_string(),
            ));
        }
        let mut coeff = Vec::with_capacity(self.cols);
        // Implementation would perform back substitution
        for i in (0..self.cols).rev() {
            coeff[i] = self.get(i, self.cols).unwrap();

            for j in 0..self.cols {
                coeff[i] = coeff[i] + self.get(i, j).unwrap().neg() * coeff[j];
            }
            coeff[i] = coeff[i] / self.get(i, i).unwrap();
        }

        Ok(coeff)
    }
}

/// Structure for polynomial fitting
#[derive(Debug)]
struct PolynomialFit<T: Copy> {
    points: Vec<Point<T, T>>,
    coeff: Option<Vec<T>>,
    degree: usize,
}

impl<
        T: Copy
            + Add<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Default
            + PartialOrd
            + std::ops::Neg<Output = T>,
    > PolynomialFit<T>
{
    /// Creates a new polynomial fit instance
    ///
    /// # Arguments
    /// * `input_points` - Vector of data points
    /// * `degree` - Degree of polynomial to fit
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

    /// Creates the design matrix for polynomial fitting
    fn create_design_matrix(&self) -> Result<Matrix<T>, FittingError> {
        // Implementation would create design matrix X
        todo!()
    }

    /// Creates the augmented matrix for Gaussian elimination
    fn create_augmented_matrix(&self) -> Result<Matrix<T>, FittingError> {
        // Implementation would create [X^T*X | X^T*y]
        todo!()
    }

    /// Performs Gaussian elimination to solve the system
    fn gaussian_elimination(&self) -> Result<Matrix<T>, FittingError> {
        let mut augmented = self.create_augmented_matrix()?;
        augmented.to_reduced_row_echelon_form()?;
        Ok(augmented)
    }

    /// Fits the polynomial to the data points
    fn fit_polynomial(&mut self) -> Result<&Vec<T>, FittingError> {
        let reduced_matrix = self.gaussian_elimination()?;
        let coefficients = reduced_matrix.back_substitute()?;
        self.coeff = Some(coefficients);
        Ok(self.coeff.as_ref().unwrap())
    }

    /// Calculates the error of the fit
    fn calculate_error(&self) -> Result<T, FittingError> {
        if let Some(coeff) = &self.coeff {
            // Implementation would calculate error
            todo!()
        } else {
            Err(FittingError::InvalidInput(
                "Coefficients not available. Call fit_polynomial first.".to_string(),
            ))
        }
    }

    /// Validates the inputs for fitting
    fn validate_inputs(&self) -> Result<(), FittingError> {
        if self.points.len() <= self.degree {
            return Err(FittingError::InsufficientData);
        }

        // More validations as needed
        Ok(())
    }
}

fn main() {
    println!("Polynomial fitting with Gaussian elimination");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_reduce_ops_reduction() {
        // Use a flat Vec<i32> for the matrix data
        let mut matrix = Matrix {
            data: vec![4.0, 5.0, 6.0, 5.0, 8.0, 4.0],
            rows: 3,
            cols: 2,
        };

        let result = matrix.to_row_echelon_form().unwrap();

        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(0, 1).unwrap(), 0.0);
        assert_eq!(matrix.get(1, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(1, 1).unwrap(), 0.0);
        assert_eq!(matrix.get(2, 1).unwrap(), 0.0);
        assert_eq!(matrix.get(2, 0).unwrap(), 0.0);
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
}
