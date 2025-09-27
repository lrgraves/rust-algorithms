use std::fs::File;
use std::io::Write;
// Excercise from Ch 4, Section 3. The goal is to mode the eqation: x(t) = a * x(t-1)

//To do this we define initial x(0), the number of t steps, and the scaling factor a.
//Then plot!
pub struct DiscreteTimeSingleVariable {
    t_steps: usize,
    starting_value: f64,
    scale_factor: f64,
}

pub fn run(t_steps: usize, starting_value: f64, scale_factor: f64) -> Vec<f64>
{
    let function = DiscreteTimeSingleVariable { t_steps, starting_value, scale_factor };
    let mut values = Vec::with_capacity(function.t_steps);
    values.push(starting_value);
    for step in 1..t_steps {
        values.push(scale_factor * values[step-1]);
    }
    values
}

pub fn export_to_csv(values: Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create("ch_4_3.csv")?;
    writeln!(file, "step,value")?;

    for (step, value) in values.iter().enumerate() {
        writeln!(file, "{},{}", step, value)?;
    }

    println!("Data exported to time_series.csv");
    Ok(())
}

