pub struct NeuralNetwork {
    layers: Vec<Layer>,
}
impl NeuralNetwork {
    pub fn stochastic_gradient_descent(
        &self,
        training_data: Vec<(Vec<f64>, Vec<f64>)>,
        epochs: i32,
        mini_batch_size: usize,
        learning_rate: f64,
    ) -> Result<(), String> {
        for j in 0..epochs {
            // TODO: shuffle training data.
            // Creating mini batches
            let mut mini_batches = Vec::new();
            for k in (0..training_data.len()).step_by(mini_batch_size) {
                mini_batches.push(&training_data[k..k + mini_batch_size]);
            }

            // Feeding mini batches one by one to the net.
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, learning_rate)?;
            }
        }

        Ok(())
    }

    fn update_mini_batch(
        &self,
        mini_batch: &[(Vec<f64>, Vec<f64>)],
        learning_rate: f64,
    ) -> Result<(), String> {
        let mut nabla_b: Vec<f64> = Vec::new();
        let mut nabla_w: Vec<f64> = Vec::new();

        for (inputs, outputs) in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backpropagation(inputs, outputs)?;

            for i in 0..nabla_b.len() {
                for j in 0..delta_nabla_b.len() {
                    nabla_b[i] = nabla_b[i] + delta_nabla_b[i];
                }
            }

            for i in 0..nabla_w.len() {
                for j in 0..delta_nabla_w.len() {
                    nabla_w[i] = nabla_w[i] + delta_nabla_w[i];
                }
            }

            // TODO: update weights and biases.
        }

        Ok(())
    }

    fn backpropagation(
        &self,
        inputs: &Vec<f64>,
        outputs: &Vec<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let mut nabla_b: Vec<Vec<f64>> = Vec::new();
        let mut nabla_w: Vec<Vec<f64>> = Vec::new();
        for layer in &self.layers {
            nabla_b.push(vec![0.0; layer.get_biases().len()]);
            nabla_w.push(vec![0.0; layer.get_weights().len()]);
        }

        // FeedForward.
        // Outputs of each layer after applying the sigmoid function.
        let mut sigmoid_outputs = Vec::with_capacity(self.layers.len() + 1);
        sigmoid_outputs.push(inputs);
        // Outputs of each layer before applying the sigmoid function.
        let outputs = Vec::with_capacity(self.layers.len());

        // Compute both outputs vectors for each layer.
        for layer_index in 1..self.layers.len() {
            let (output, sigmoid_output) =
                self.layers[layer_index].compute_outputs_full(sigmoid_outputs[layer_index - 1])?;
            outputs.push(output);
            sigmoid_outputs.push(&sigmoid_output);
        }

        // Backward.
        let delta = cost_derivative(sigmoid_outputs[sigmoid_outputs.len()], expected_outputs)
            * sigmoid_derivative(sigmoid_outputs[-1]);

        return (nabla_b, nabla_w);
    }

    pub fn compute_outputs(&self, inputs: Vec<f64>) -> Result<Vec<f64>, String> {
        let mut outputs = self.layers[0].compute_outputs(inputs)?;

        for i in 1..self.layers.len() {
            outputs = self.layers[i].compute_outputs(outputs)?;
        }

        Ok(outputs)
    }

    pub fn new(neurons_counts: Vec<i32>) -> NeuralNetwork {
        let mut neural_network = NeuralNetwork {
            layers: Vec::with_capacity(neurons_counts.len() - 1),
        };

        // Each layer has a certain number of neurons.
        for i in 1..neurons_counts.len() {
            neural_network
                .layers
                .push(Layer::new(neurons_counts[i - 1], neurons_counts[i]));
        }

        return neural_network;
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}
impl Layer {
    pub fn compute_outputs(&self, inputs: Vec<f64>) -> Result<Vec<f64>, String> {
        let mut outputs = Vec::with_capacity(self.neurons.len());
        for neuron in &self.neurons {
            outputs.push(neuron.compute_output(&inputs)?);
        }

        Ok(outputs)
    }

    pub fn compute_outputs_full(&self, inputs: &Vec<f64>) -> Result<(Vec<f64>, Vec<f64>), String> {
        let mut outputs = Vec::with_capacity(self.neurons.len());
        let mut sigmoid_outputs = Vec::with_capacity(self.neurons.len());

        for neuron in &self.neurons {
            let (output, sigmoid_output) = neuron.compute_output_full(&inputs)?;
            outputs.push(output);
            sigmoid_outputs.push(sigmoid_output);
        }

        Ok((outputs, sigmoid_outputs))
    }

    pub fn get_biases(&self) -> Vec<f64> {
        let mut output = Vec::with_capacity(self.neurons.len());
        for neuron in &self.neurons {
            output.push(neuron.bias);
        }

        return output;
    }

    pub fn get_weights(&self) -> Vec<&Vec<f64>> {
        let mut output = Vec::with_capacity(self.neurons.len());
        for neuron in &self.neurons {
            output.push(&neuron.weights);
        }

        return output;
    }

    pub fn new(inputs_size: i32, neurons_count: i32) -> Layer {
        let mut layer = Layer {
            neurons: Vec::with_capacity(neurons_count as usize),
        };

        for _i in 0..neurons_count {
            layer.neurons.push(Neuron::new(inputs_size))
        }
        return layer;
    }
}

struct Neuron {
    bias: f64,
    weights: Vec<f64>,
}

impl Neuron {
    fn compute_output(&self, inputs: &Vec<f64>) -> Result<f64, String> {
        let output = sigmoid(dot_product(&inputs, &self.weights)? + self.bias);

        Ok(output)
    }

    fn compute_output_full(&self, inputs: &Vec<f64>) -> Result<(f64, f64), String> {
        let output = dot_product(&inputs, &self.weights)? + self.bias;
        let output_with_sigmoid = sigmoid(output);

        Ok((output, output_with_sigmoid))
    }

    pub fn new(inputs_number: i32) -> Neuron {
        let mut neuron = Neuron {
            bias: 1.3,
            weights: Vec::with_capacity(inputs_number as usize),
        };

        for i in 0..neuron.weights.len() {
            // TODO: properly initialize weights.
            neuron.weights[i] = 0.3;
        }

        return neuron;
    }
}

fn sigmoid(z: f64) -> f64 {
    return 1.0 / (1.0 + (-z).exp());
}

fn sigmoid_derivative(z: f64) -> f64 {
    return sigmoid(z) * (1.0 - sigmoid(z));
}

fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(String::from("The size of the 2 vectors mismatched"));
    }

    let mut result = 0.0;
    for i in 0..a.len() {
        result = result + a[i] * b[i];
    }

    Ok(result)
}

fn cost_derivative(network_outputs: &Vec<f64>, expected_outputs: &Vec<f64>) -> Vec<f64> {
    let output = Vec::with_capacity(network_outputs.len());

    for i in 0..network_outputs.len() {
        output.push(network_outputs[i] - expected_outputs[i]);
    }

    return output;
}
