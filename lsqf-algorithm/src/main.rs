struct Point{
    x: f32,
    y: f32
}


struct Matrix{
    data: Vec<Vec<f32>>
}

impl Matrix{
    fn calculate_transposed_matrix(self)->matrix {
        // Calculate the transpose matrix
    }

    fn multiply_by_second_matrix(self, second_matrix: matrix)->matrix {
        // Calculate the matrix multiplication. A' = self * second_matrix
    }

    fn find_pivot(self, column: usize) -> usize{
        // Find row in a column with maximum value
    }

    fn swap_rows(mut self, row_to_swap_to: usize, row_to_swap_from:usize) {
        // swaps rows in the self matrix
    }

    fn normalize_row(mut self, row: usize){
        // normalizes row by dividing by max value? or by first value in columns?
    }

    fn eliminate_below(mut self, pivot_row: usize, column: usize){
        // back elimination for row echelon form
    }

    fn eliminate_above(mut self, pivot_row: usize, column: usize){
        // forward elimination for row echelon form
    }

    fn check_row_echelon_form(self) -> bool{
        // verify in row echelon form
    }

    fn back_substitute(self)->Vec<f32>{
        // solve
    }
}
struct PolynomialFit {
    points: Vec<point>,
    coeff: Vec<f32>,
    design_matrix: matrix,
}

impl polynomial_fit{
    fn new(input_points: Vec<points>)->polynomial_fit{
        // create design matrix
    }
    fn create_augmented_matrix(self)->matrix{
        // calculate augmented matrix by taking transpose of design matrix
    }

    fn gaussian_elimination(self)->Result<matrix, Err>{
        // perform guassian elimination by making the augmented matrix, then getting it to row echelon form, checked at end. outputs new matrix
    }

    fn fit_polynomial(mut self)->Result(Vec<f32>, f32){
        // fits a coefficient vector, updates
    }

    fn calculate_error(self, coeff)->Result<f32, Err>{
        // using the coeff, calculate error
    }
    
    fn validate_inputs(self)->Results<bool, Err>{
        // validates inputs? 
    }

}
fn main() {
    println!("Hello, world!");
}
