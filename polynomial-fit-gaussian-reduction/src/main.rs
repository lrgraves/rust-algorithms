// main.rs - Command Line Interface for Linear Algebra Operations

use std::env;
use std::io::{self, Write};

// Import from your library
use polynomial_fit_gaussian_reduction::prelude::*;
use polynomial_fit_gaussian_reduction::Point;

fn main() {
    println!("Linear Algebra Operations Toolkit");
    println!("=====================================");

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "gaussian" => run_gaussian_elimination_demo(),
        "poly" => run_polynomial_fitting_demo(),
        "lu" => run_lu_decomposition_demo(),
        "det" => run_determinant_demo(),
        "singular" => run_singularity_check_demo(),
        "interactive" => run_interactive_mode(),
        "--help" | "-h" => print_usage(),
        _ => {
            println!("Unknown command: {}", args[1]);
            print_usage();
        }
    }
}

fn print_usage() {
    println!("Usage: cargo run <command>");
    println!();
    println!("Commands:");
    println!("  gaussian    - Demonstrate Gaussian elimination");
    println!("  poly        - Demonstrate polynomial fitting");
    println!("  lu          - Demonstrate LU decomposition");
    println!("  det         - Calculate matrix determinant");
    println!("  singular    - Check if matrix is singular");
    println!("  interactive - Interactive matrix input mode");
    println!("  --help      - Show this help message");
    println!();
    println!("Examples:");
    println!("  cargo run gaussian");
    println!("  cargo run poly");
    println!("  cargo run interactive");
}

fn run_gaussian_elimination_demo() {
    println!("Gaussian Elimination Demo");
    println!("============================");

    // TODO: Get matrix from user input or use predefined example
    let mut matrix = get_example_augmented_matrix();

    println!("Original augmented matrix:");
    print_matrix(&matrix);

    match matrix.to_reduced_row_echelon_form() {
        Ok(_) => {
            println!("\nReduced Row Echelon Form:");
            print_matrix(&matrix);

            match matrix.back_substitute() {
                Ok(solution) => {
                    println!("\nSolution vector:");
                    for (i, val) in solution.iter().enumerate() {
                        println!("  x{} = {:.6}", i, val);
                    }
                }
                Err(e) => println!("❌ Error solving system: {}", e),
            }
        }
        Err(e) => println!("❌ Error during elimination: {}", e),
    }
}

fn run_polynomial_fitting_demo() {
    println!("Polynomial Fitting Demo");
    println!("===========================");

    // Generate sample data: y = 2 + 3x + x²
    let points = vec![
        Point { x: 0.0, y: 2.0 },
        Point { x: 1.0, y: 6.0 },
        Point { x: 2.0, y: 12.0 },
        Point { x: 3.0, y: 20.0 },
        Point { x: 4.0, y: 30.0 },
    ];

    println!("Sample data points:");
    for (i, p) in points.iter().enumerate() {
        println!("  ({:.1}, {:.1})", p.x, p.y);
    }

    let degree = 2;
    let mut poly_fit = match PolynomialFit::new(points, degree) {
        Ok(pf) => pf,
        Err(e) => {
            println!("❌ Error creating polynomial fit: {}", e);
            return;
        }
    };

    match poly_fit.fit_polynomial() {
        Ok(coefficients) => {
            println!("\nFitted polynomial coefficients:");
            for (i, coeff) in coefficients.iter().enumerate() {
                println!("  x^{}: {:.6}", i, coeff);
            }

            match poly_fit.calculate_error() {
                Ok(error) => println!("Sum of squared errors: {:.6}", error),
                Err(e) => println!("❌ Error calculating fit error: {}", e),
            }
        }
        Err(e) => println!("❌ Error fitting polynomial: {}", e),
    }
}

fn run_lu_decomposition_demo() {
    println!("LU Decomposition Demo");
    println!("========================");

    let matrix = get_example_square_matrix();

    println!("Original matrix:");
    print_matrix(&matrix);

    // TODO: Implement once you have LU decomposition
    match matrix.lu_decomposition() {
        Ok(lu) => {
            println!("\nL matrix (Lower triangular):");
            print_matrix(&lu.l_matrix);

            println!("\nU matrix (Upper triangular):");
            print_matrix(&lu.u_matrix);

            println!("\nPermutation vector: {:?}", lu.permutation);

            println!("Determinant from LU: {:.6}", lu.determinant());
        }
        Err(e) => println!("❌ Error during LU decomposition: {}", e),
    }
}

fn run_determinant_demo() {
    println!("Determinant Calculation Demo");
    println!("===============================");

    // Test with 2x2 matrix first
    let matrix_2x2 = Matrix {
        data: vec![3.0, 8.0, 4.0, 6.0],
        rows: 2,
        cols: 2,
    };

    println!("2x2 Matrix:");
    print_matrix(&matrix_2x2);

    // TODO: Implement determinant calculation
    match matrix_2x2.determinant() {
        Ok(det) => println!("Determinant: {:.6}", det),
        Err(e) => println!("❌ Error calculating determinant: {}", e),
    }

    // Test with larger matrix
    let matrix_3x3 = get_example_square_matrix();
    println!("\n3x3 Matrix:");
    print_matrix(&matrix_3x3);

    match matrix_3x3.determinant() {
        Ok(det) => println!("Determinant: {:.6}", det),
        Err(e) => println!("❌ Error calculating determinant: {}", e),
    }
}

fn run_singularity_check_demo() {
    println!(" Singularity Check Demo");
    println!("==========================");

    // Non-singular matrix
    let regular_matrix = get_example_square_matrix();
    println!("Regular matrix:");
    print_matrix(&regular_matrix);

    // TODO: Implement singularity check
    match regular_matrix.is_singular() {
        Ok(is_sing) => println!("Is singular: {}", is_sing),
        Err(e) => println!("❌ Error checking singularity: {}", e),
    }

    // Singular matrix (rows are linearly dependent)
    let singular_matrix = Matrix {
        data: vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 2.0, 3.0],
        rows: 3,
        cols: 3,
    };

    println!("\nSingular matrix (linearly dependent rows):");
    print_matrix(&singular_matrix);

    match singular_matrix.is_singular() {
        Ok(is_sing) => println!("Is singular: {}", is_sing),
        Err(e) => println!("❌ Error checking singularity: {}", e),
    }
}

fn run_interactive_mode() {
    println!("Interactive Matrix Input Mode");
    println!("=================================");

    loop {
        println!("\nOptions:");
        println!("  1. Input matrix for Gaussian elimination");
        println!("  2. Input matrix for LU decomposition");
        println!("  3. Input points for polynomial fitting");
        println!("  4. Exit");

        print!("Choose option (1-4): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => {
                if let Some(matrix) = input_augmented_matrix() {
                    run_gaussian_on_matrix(matrix);
                }
            }
            "2" => {
                if let Some(matrix) = input_square_matrix() {
                    run_lu_on_matrix(matrix);
                }
            }
            "3" => {
                if let Some(points) = input_polynomial_points() {
                    run_polynomial_on_points(points);
                }
            }
            "4" => {
                println!("🖖 Live long and prosper!");
                break;
            }
            _ => println!("❌ Invalid option. Please choose 1-4."),
        }
    }
}

// Helper functions for matrix input and display
fn input_augmented_matrix() -> Option<Matrix<f64>> {
    println!("Enter augmented matrix dimensions:");

    print!("Rows: ");
    io::stdout().flush().unwrap();
    let rows = read_usize()?;

    print!("Columns: ");
    io::stdout().flush().unwrap();
    let cols = read_usize()?;

    println!("Enter matrix elements row by row:");
    let mut data = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        print!("Row {}: ", row + 1);
        io::stdout().flush().unwrap();

        let mut line = String::new();
        io::stdin().read_line(&mut line).unwrap();

        let row_values: Result<Vec<f64>, _> =
            line.trim().split_whitespace().map(|s| s.parse()).collect();

        match row_values {
            Ok(mut values) => {
                if values.len() != cols {
                    println!("❌ Expected {} values, got {}", cols, values.len());
                    return None;
                }
                data.append(&mut values);
            }
            Err(_) => {
                println!("❌ Invalid number format");
                return None;
            }
        }
    }

    Some(Matrix { data, rows, cols })
}

fn input_square_matrix() -> Option<Matrix<f64>> {
    println!("Enter square matrix dimension:");

    print!("Size (n for nxn): ");
    io::stdout().flush().unwrap();
    let size = read_usize()?;

    println!("Enter matrix elements row by row:");
    let mut data: Vec<f64> = Vec::with_capacity(size * size); // Add explicit type annotation

    // TODO: Implement matrix input logic similar to augmented matrix
    // For now, return example matrix
    Some(get_example_square_matrix())
}

fn input_polynomial_points() -> Option<Vec<Point<f64, f64>>> {
    println!("Enter number of data points:");

    print!("Count: ");
    io::stdout().flush().unwrap();
    let count = read_usize()?;

    let mut points = Vec::with_capacity(count);

    for i in 0..count {
        print!("Point {} (x y): ", i + 1);
        io::stdout().flush().unwrap();

        let mut line = String::new();
        io::stdin().read_line(&mut line).unwrap();

        let coords: Vec<&str> = line.trim().split_whitespace().collect();
        if coords.len() != 2 {
            println!("❌ Expected x and y coordinates");
            return None;
        }

        match (coords[0].parse::<f64>(), coords[1].parse::<f64>()) {
            (Ok(x), Ok(y)) => points.push(Point { x, y }),
            _ => {
                println!("❌ Invalid coordinate format");
                return None;
            }
        }
    }

    Some(points)
}

fn read_usize() -> Option<usize> {
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().parse().ok()
}

fn run_gaussian_on_matrix(mut matrix: Matrix<f64>) {
    println!("\nPerforming Gaussian elimination...");
    // TODO: Implement using your existing gaussian elimination code
}

fn run_lu_on_matrix(matrix: Matrix<f64>) {
    println!("\nPerforming LU decomposition...");
    // TODO: Implement using your LU decomposition code
}

fn run_polynomial_on_points(points: Vec<Point<f64, f64>>) {
    println!("\nPerforming polynomial fitting...");
    // TODO: Implement using your polynomial fitting code
}

// Example matrices for demos
fn get_example_augmented_matrix() -> Matrix<f64> {
    // System: x + 2y + 3z = 14
    //         2x + y + z = 9
    //         x + y + z = 6
    Matrix {
        data: vec![1.0, 2.0, 3.0, 14.0, 2.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 6.0],
        rows: 3,
        cols: 4,
    }
}

fn get_example_square_matrix() -> Matrix<f64> {
    Matrix {
        data: vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0],
        rows: 3,
        cols: 3,
    }
}

fn print_matrix<T: std::fmt::Display + Copy + Default>(matrix: &Matrix<T>) {
    let (rows, cols) = matrix.shape();

    for row in 0..rows {
        print!("  [");
        for col in 0..cols {
            if let Some(value) = matrix.get(row, col) {
                if col == cols - 1 {
                    print!("{:8.3}", value);
                } else {
                    print!("{:8.3}, ", value);
                }
            }
        }
        println!("]");
    }
}
