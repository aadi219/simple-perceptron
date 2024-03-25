class Perceptron {
  // property initialization.
  // weighst initialized as array for scalability but only a single weight (weights[0]) is being used for testing purposes.
  float[] weights;
  float bias = 0;
  
  // Learning Rate
  float lr = 0.0001;
  
  public Perceptron(int num_weights) {
    weights = new float[num_weights];
    for(int i = 0; i < num_weights; i++) {
      // random weights initialized at point of construction.
      weights[i] = random(0,10);
    }
  }
  
  public float output(float input) {
    // returns a prediction based on the current state of weights and biases.
    return input * weights[0] + bias;
  }
 
  public float mean_squared_error(float y_true, float y_pred) {
    // error function chosen to be MSE, other error functions may also be valid.
    return pow((y_true - y_pred), 2);
  }
  
  public float[] train_single(float feature, float target) {
    /*
      Performs one single training cycle and returns the state of the neuron after training.
      A prediction is made based on the feature input provided.
      The required change in the weight and bias is computed for the computed guess and given target value and added to the weight and bias.
    */
    float guess = output(feature);
    float error = mean_squared_error(target, guess);
    
    // finding partial derivative of error w.r.t weights and bias
    float weight_derivative = -1 * lr * 2 * feature * (guess - target);
    float bias_derivative = -1 * lr * 2 * (guess - target);
    
    
    weights[0] += weight_derivative;
    bias += bias_derivative;
    
    // return the state of the network after training for logging and debugging.
    return new float[] {weights[0], bias, error};
  }
  
  public float[] train(float[] features, float[] targets) {
    /*
      Trains the neuron on multiple features in a single function call.
      Works the same as train_single() but performs multiple training operations in a single batch.
    */
    float guess = 0;
    float error = 0;
    for(int i = 0; i < features.length; i++) {
      guess = output(features[i]);
      error = mean_squared_error(targets[i], guess);
      
      // finding partial derivative of error w.r.t weights and bias
      float weight_derivative = -1 * lr * 2 * features[i] * (guess - targets[i]);
      float bias_derivative = -1 * lr * 2 * (guess - targets[i])*(0.5);
      
      weights[0] += weight_derivative;
      bias += bias_derivative;
    }
    float[] results = new float[] {weights[0], bias, error};
    return results;
  }
  
}
