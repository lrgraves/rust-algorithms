/// A data point which is used in the  fitting.
#[derive(Debug, Clone, PartialEq)]
pub struct Point<T, U> {
    pub x: T,
    pub y: U,
}
