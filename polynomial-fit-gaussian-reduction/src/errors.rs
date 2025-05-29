use std::fmt;

/// Custom error type for polynomial fitting operations
#[derive(Debug)]
pub enum FittingError {
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
