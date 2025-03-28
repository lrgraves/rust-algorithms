use std::fmt::Debug;

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

pub fn check_transparency_requirement(n:usize, odds_of_opaque_bit: f32, monte_carlo_trials: usize) -> f32 {
    /// This checks if a crystal or microscturcture, which is modeled as a NxN 3D structure, will have 'a' clear line of sight through it to transmit a signal.
    /// To check this, a percolation test is run, for M monte carlo trials, using the quick find algorithm. This reports the percent of trials that had a clear line
    /// of sight.
    
    // Create model
    //The structure is nxn large, with 2 points added for top and bottom
    let mut uf = QuickUnionUF::new(n*n + 2);

    let outcome = Vec::with_capacity(monte_carlo_trials);

    // Start a new monte carlo trial

        // Randomly union parts

        // Check if connected

        // Report success or failure
        // if connected -> outcome[i] = 1
    
    // Report total success rate


}

use rand::Rng;
use std::time::{Duration, Instant};

// Benchmark function to measure performance against input size
fn benchmark_union_find(sizes: &[usize], operations_per_size: usize) {
    println!("\nBenchmarking QuickUnionUF with weighting and path compression");
    println!(
        "{:<10} | {:<15} | {:<15} | {:<15}",
        "Size (n)", "Union Time (µs)", "Find Time (µs)", "Time/log(n)"
    );
    println!("{:-<64}", "");

    for &n in sizes {
        let mut uf = QuickUnionUF::new(n);
        let mut rng = rand::thread_rng();

        // Measure union operations
        let union_start = Instant::now();
        for _ in 0..operations_per_size {
            let p = rng.gen_range(0..n);
            let q = rng.gen_range(0..n);
            uf.union(p, q);
        }
        let union_time = union_start.elapsed();

        // Measure find/connected operations
        let find_start = Instant::now();
        for _ in 0..operations_per_size {
            let p = rng.gen_range(0..n);
            let q = rng.gen_range(0..n);
            uf.connected(p, q);
        }
        let find_time = find_start.elapsed();

        // Calculate time per log(n) to check if it's logarithmic
        let log_n = (n as f64).ln();
        let time_per_log_n = union_time.as_micros() as f64 / log_n;

        println!(
            "{:<10} | {:<15} | {:<15} | {:<15.2}",
            n,
            union_time.as_micros(),
            find_time.as_micros(),
            time_per_log_n
        );
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

fn main() {
    // Run basic correctness test
    basic_test();

    // Run performance benchmark with various input sizes
    // Uses exponentially increasing sizes to clearly show growth patterns
    let sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000];
    benchmark_union_find(&sizes, 1000);
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
