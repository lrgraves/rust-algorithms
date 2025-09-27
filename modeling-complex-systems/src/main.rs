mod discrete_time_modeling_single_variable_4_3;

fn main() {
    let ch4_3_values = discrete_time_modeling_single_variable_4_3::run(91, 16., 6.);
    println!("{:#?}", ch4_3_values);
    discrete_time_modeling_single_variable_4_3::export_to_csv(ch4_3_values).expect("TODO: panic message");
    println!("The data  in in a csv in the root directory");
}
