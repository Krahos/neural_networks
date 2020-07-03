mod neural_network;
use neural_network::NeuralNetwork;
fn main() {
    let neuron_counts = vec![200, 15, 14, 10];
    let neural_network = NeuralNetwork::new(neuron_counts);

    // TODO
    let training_data = Vec::new();
    neural_network.stochastic_gradient_descent(training_data, 30, 10, 3.0);

    // TODO
    let inputs = Vec::new();
    match neural_network.compute_outputs(inputs) {
        Ok(outputs) => println!("{:?}", outputs),
        Err(error) => println!("{:?}", error),
    }
}
