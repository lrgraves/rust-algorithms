use std::ops::{Add, Div, Mul};

use crate::FittingError;

/// Matrix implementation with efficient storage and vector flat format (speed + efficiency!)
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T: Copy + Default> Matrix<T> {
    /// Creates a new matrix with the specified dimensions
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `default_value` - Default value to fill the matrix with
    pub fn new(rows: usize, cols: usize, default_value: T) -> Self {
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
    pub fn get(&self, row: usize, col: usize) -> Option<T> {
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
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), FittingError> {
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
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl<T: Copy + Default> Matrix<T> {
    /// This function calculates the transpose of a matrix A. This involves:
    /// 1) Reflecting A over its main diagonal
    /// 2) Write the rows of A as the columns of A_t
    /// 3) Write the columns of A as the rows of A_t
    /// Checks: The shape should flip, and a direct check of values
    pub fn calculate_transposed_matrix(&self) -> Result<Matrix<T>, FittingError> {
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
    pub fn multiply_by_second_matrix(
        &self,
        second_matrix: Matrix<T>,
    ) -> Result<Matrix<T>, FittingError>
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
    pub fn find_pivot(&self, column: usize, start_row: usize) -> Result<usize, FittingError> {
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
    pub fn swap_rows(
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
    pub fn normalize_row(
        &mut self,
        row: usize,
        pivot_col: usize,
    ) -> Result<&mut Self, FittingError> {
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
    pub fn eliminate_below(
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
    pub fn eliminate_above(
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
    pub fn to_row_echelon_form(&mut self) -> Result<&mut Self, FittingError> {
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
    pub fn to_reduced_row_echelon_form(&mut self) -> Result<&mut Self, FittingError> {
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
    pub fn check_row_echelon_form(&self) -> bool {
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
    pub fn back_substitute(&self) -> Result<Vec<T>, FittingError> {
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
