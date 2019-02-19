function sigmoid(x){
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y){
  //return sigmoid(x) * (1 - sigmoid(x));
  return (y) * (1 - y);
}

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes){
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();

    this.learning_rate = 0.1;
  }

  //Set the nn learning rate
  setLearningRate(x){
    this.learning_rate = x;
  }
  
  //Pass inputs through the network, return an output
  //ih - input to hidden weights
  feedforward(input_array){

    //Generate Hidden Node outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);

    //Call activation function (sigmoid) on hidden node outputs
    hidden.map(sigmoid);

    //Generating the output's outputs!
    //ho - hidden to output weights
    let output = Matrix.multiply(this.weights_ho, hidden);
    output.add(this.bias_o);
    output.map(sigmoid);
    //Return output array to caller
    return output.toArray();
  }

  //Supervised Learning, pass in inputs with respective targets
  train(input_array, target_array){
    //Get inputs (matrix) from input array
    let inputs = Matrix.fromArray(input_array);

    //Get outputs of hidden nodes, add the bias
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);

    //Call activation function on hidden outputs
    hidden.map(sigmoid);

    //Generating the outputs outputs
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(sigmoid);

    //Convert targets array to matrix object
    let targets = Matrix.fromArray(target_array);

    //OUTPUT_ERROR = TARGETS - OUPUTS
    let output_errors = Matrix.subtract(targets, outputs);

    //Calculate the output gradients
    let gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);

    gradients.multiply(this.learning_rate);

    //Calculate hidden to output deltas
    let hidden_T = Matrix.transpose(hidden);
    let weights_ho_deltas = Matrix.multiply(gradients, hidden_T);

    //Adjust the weights by deltas
    this.weights_ho.add(weights_ho_deltas);

    //Adjust the bias by its deltas (the gradients)
    this.bias_o.add(gradients);

    //Calculate hidden Layer errors
    //multiply transposed matrix of h->o weights with the errors
    let who_t = Matrix.transpose(this.weights_ho);
    let hidden_errors = Matrix.multiply(who_t, output_errors);

    //Calculate Hidden Gradients
    let hidden_gradient = Matrix.map(hidden, dsigmoid);
    hidden_errors
    hidden_gradient.multiply(hidden_errors);
    hidden_gradient.multiply(this.learning_rate);

    //input to hidden deltas
    let inputs_T = Matrix.transpose(inputs);

    let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);
    //Adjust the weights by deltas

    this.weights_ih.add(weight_ih_deltas);

    //Adjust the bias by its delta (the gradient)
    this.bias_h.add(hidden_gradient);





    // outputs.print();
    // targets.print();
    // error.print();

  }
}
