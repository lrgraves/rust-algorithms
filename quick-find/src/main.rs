use std::fmt::Debug;
use rand::prelude::*;

#[derive(Debug)]
pub struct QuickUnionUF {
    // Tracks parent connections
    pub parent: Vec<usize>,
    // Tracks size of each tree for weighting
    pub size: Vec<usize>,
}

impl QuickUnionUF {
    pub fn new(n: usize) -> Self {
        // Initialize each element as its own parent with size 1
        let parent = (0..n).collect();
        let size = vec![1; n];

        QuickUnionUF { parent, size }
    }

    pub fn root(&mut self, mut i: usize) -> usize {
        // Find root with path compression
        while self.parent[i] != i {
            // Path compression - point to grandparent
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }

    pub fn connected(&mut self, p: usize, q: usize) -> bool {
        // Check if p and q have the same root
        self.root(p) == self.root(q)
    }

    pub fn union(&mut self, p: usize, q: usize) {
        // Connect p and q with weighting
        let root_p = self.root(p);
        let root_q = self.root(q);

        // Already connected
        if root_p == root_q {
            return;
        }

        // Weighted union: connect smaller tree to larger one
        if self.size[root_p] < self.size[root_q] {
            self.parent[root_p] = root_q;
            self.size[root_q] += self.size[root_p];
        } else {
            self.parent[root_q] = root_p;
            self.size[root_p] += self.size[root_q];
        }
    }
}
// Helper function to convert 2D coordinates to 1D index
fn get_index(row: usize, col: usize, n: usize) -> usize {
    row * n + col
}

pub fn check_transparency_requirement(n: usize, odds_of_transparent_bit: f32, monte_carlo_trials: usize) -> f32 {
    /// This checks if a crystal or microstructure, which is modeled as an NxN 2D structure, will have a clear line of sight through it.
    /// To check this, a percolation test is run for M monte carlo trials, using the quick find algorithm. 
    /// This reports the percent of trials that had a clear line of sight.
    
    // Track successful percolation trials
    let mut successful_trials = 0;
    
    // Define virtual nodes
    let top_node = n * n;     // Virtual top node
    let bottom_node = n * n + 1;  // Virtual bottom node
    
    let mut rng = rand::rng();
    
    // Start monte carlo trials
    for _ in 0..monte_carlo_trials {
        // Create a new Union-Find structure for each trial
        let mut uf = QuickUnionUF::new(n * n + 2);  // +2 for virtual top and bottom nodes
        
        // Create grid of transparent/opaque cells
        let mut grid = vec![false; n * n];
        
        // Randomly assign transparent/opaque values based on probability
        for i in 0..n*n {
            grid[i] = rng.random_bool(odds_of_transparent_bit as f64);
        }
        
        // Connect transparent cells to virtual nodes and to adjacent transparent cells
        for row in 0..n {
            for col in 0..n {
                let cell_idx = get_index(row, col, n);
                
                // Skip if cell is opaque
                if !grid[cell_idx] {
                    continue;
                }
                
                // Connect top row transparent cells to virtual top node
                if row == 0 {
                    uf.union(cell_idx, top_node);
                }
                
                // Connect bottom row transparent cells to virtual bottom node
                if row == n - 1 {
                    uf.union(cell_idx, bottom_node);
                }
                
                // Connect to adjacent transparent cells (right and down)
                // Check right neighbor
                if col < n - 1 && grid[get_index(row, col + 1, n)] {
                    uf.union(cell_idx, get_index(row, col + 1, n));
                }
                
                // Check down neighbor
                if row < n - 1 && grid[get_index(row + 1, col, n)] {
                    uf.union(cell_idx, get_index(row + 1, col, n));
                }
            }
        }
        
        // Check if percolation occurs (top is connected to bottom)
        if uf.connected(top_node, bottom_node) {
            successful_trials += 1;
        }
    }
    
    // Return probability of percolation
    successful_trials as f32 / monte_carlo_trials as f32
}


#[allow(dead_code)]
fn main() {
    // Parameters
    let grid_size = 20;
    let probability_transparent = 0.59; // Theoretical percolation threshold for 2D square grid is around 0.593
    let trials = 1000;
    
    // Run the percolation test
    let percolation_probability = check_transparency_requirement(grid_size, probability_transparent, trials);
    
    println!("Grid size: {}x{}", grid_size, grid_size);
    println!("Transparency probability: {:.3}", probability_transparent);
    println!("Percolation probability after {} trials: {:.4}", trials, percolation_probability);
    
    
    // Optional: Test different transparency probabilities
    println!("\nTesting different transparency probabilities:");
    for p in (40..70).step_by(5) {
        let prob = p as f32 / 100.0;
        let perc = check_transparency_requirement(grid_size, prob, trials);
        println!("p = {:.2}: percolation probability = {:.4}", prob, perc);
    }
}


// Simple test for correctness
fn basic_test() {
    println!("\nRunning basic correctness test:");
    let mut uf = QuickUnionUF::new(10);

    // Connect some elements
    uf.union(0, 1);
    uf.union(2, 3);
    uf.union(4, 5);
    uf.union(6, 7);
    uf.union(8, 9);
    uf.union(0, 2);
    uf.union(4, 6);
    uf.union(0, 4);

    // Check connectivity
    println!("0 and 9 connected: {}", uf.connected(0, 9)); // Should be false
    println!("0 and 7 connected: {}", uf.connected(0, 7)); // Should be true
    println!("Current structure: {:?}", uf.parent);

    // Final connections to merge all components
    uf.union(0, 8);

    // Verify all elements are now connected
    println!("All connected: {}", uf.connected(0, 9)); // Should be true
}


#[cfg(test)]
mod tests {
    use crate::QuickUnionUF;


    #[test]
    fn test_connection_find() {
        //should return false, then true
        let mut uf = QuickUnionUF::new(10);

        assert_eq!(uf.connected(0, 9), false);

        uf.union(0, 1);
        uf.union(0, 2); // creating branches
        uf.union(1, 4);
        uf.union(3,7);
        uf.union(3,1);  // verifying ordering and weighted union
        uf.union(6,9);
        uf.union(3,6); // now, 9 should be connected to 0

        assert!(uf.connected(0, 9));
    }

    #[test]
    fn test_percolation_optic() {
        //if we imagine that there is some crystaline or micro structure optic, who has defects in it, which are opaque. At was level of defects can we still likely get a 
        // clear line of sight? The percolation test will test this.

        //The structure is nxn large
        let n = 10;
        let mut uf = QuickUnionUF::new(n*n);

        assert_eq!(uf.connected(0, 9), false);

        uf.union(0, 1);
        uf.union(0, 2); // creating branches
        uf.union(1, 4);
        uf.union(3,7);
        uf.union(3,1);  // verifying ordering and weighted union
        uf.union(6,9);
        uf.union(3,6); // now, 9 should be connected to 0

        assert!(uf.connected(0, 9));
    }


}
