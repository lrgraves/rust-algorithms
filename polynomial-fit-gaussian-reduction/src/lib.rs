//! # Linear Algebra Learning Library
//!
//! Examples to demonstrate learning from the "Linear Algebra and Its Applications" book.
//!
//! This library demonstrates core examples from each chapter.

// External dependencies
pub use num_traits::{Float, One, Zero};

// Core modules - the foundation of our mathematical enterprise
pub mod errors;
pub mod matrix;
pub mod point;

// Chapter-specific implementations
pub mod matrix_decomposition; // Chapter 2: LU factorization, 2x2 determinants and singular matrices.
pub mod polynomial_fit; // Chapter 1: Gaussian elimination & polynomial fitting

// Re-export commonly used types for convenience
// Think of this as the senior staff having direct bridge access
pub use errors::FittingError;
pub use matrix::Matrix;
pub use matrix_decomposition::LUDecomposition;
pub use point::Point;
pub use polynomial_fit::PolynomialFit;

// Prelude module for convenient imports
// Like having your most essential tools in easy reach
pub mod prelude {
    pub use crate::{FittingError, LUDecomposition, Matrix, Point, PolynomialFit};
    pub use num_traits::{Float, One, Zero};
}

// Utility functions that don't belong to specific structs
pub mod utils {
    use num_traits::Float;

    use crate::FittingError;

    /// Checks if two floating point values are approximately equal
    /// Essential for numerical computing - like checking if coordinates are "close enough"
    pub fn approx_equal<T: Float>(a: T, b: T, epsilon: T) -> bool {
        (a - b).abs() < epsilon
    }

    /// Creates a random test matrix for benchmarking and testing
    /// TODO: Implement random matrix generation
    pub fn random_matrix<T>(_rows: usize, _cols: usize) -> Result<crate::Matrix<T>, FittingError>
    where
        T: Copy + Default,
    {
        todo!("Implement random matrix generation for testing")
    }
}

// Integration tests that demonstrate real-world usage
#[cfg(test)]
mod integration_tests {
    use super::prelude::*;

    #[test]
    fn test_full_workflow_polynomial_fitting() {
        // TODO: Test the complete polynomial fitting pipeline
        // This demonstrates how all modules work together
    }

    #[test]
    fn test_full_workflow_matrix_analysis() {
        // TODO: Test LU decomposition -> determinant -> singularity check
        // Shows the mathematical relationship between concepts
    }
}

// Feature flags for optional functionality
// Like having different ship configurations for different missions
#[cfg(feature = "benchmarks")]
pub mod benchmarks {
    //! Benchmarking utilities for performance analysis
    //! Enable with: cargo build --features benchmarks

    pub fn benchmark_matrix_multiplication() {
        todo!("Implement matrix multiplication benchmarks")
    }

    pub fn benchmark_lu_decomposition() {
        todo!("Implement LU decomposition benchmarks")
    }
}

#[cfg(feature = "visualization")]
pub mod visualization {
    //! Optional visualization utilities
    //! Enable with: cargo build --features visualization

    pub fn plot_polynomial_fit() {
        todo!("Implement polynomial visualization")
    }

    pub fn visualize_matrix_transformation() {
        todo!("Implement matrix transformation visualization")
    }
}
