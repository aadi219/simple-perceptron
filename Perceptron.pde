class Perceptron {
  float[] weights;
  float bias = 0;
  //float lr = 0.00022; // overshooting initial guess.
  float lr = 0.0001; // undershooting, large steps in descent.
  //float lr = 0.00023; // and above, cannot reach minima.
  
  public Perceptron(int num_weights) {
    weights = new float[num_weights];
    for(int i = 0; i < num_weights; i++) {
      weights[i] = random(0,10);
    }
  }
  
  public float output(float input) {
    return input * weights[0] + bias;
  }
 
  public float mean_squared_error(float y_true, float y_pred) {
    return pow((y_true - y_pred), 2);
  }
  
  public float[] train_single(float feature, float target) {
    float guess = output(feature);
    float error = mean_squared_error(target, guess);
    
    float weight_derivative = -1 * lr * 2 * feature * (guess - target);
    float bias_derivative = -1 * lr * 2 * (guess - target);
    
    println("Feature: " + feature + "\tGuess: " + guess);
    println("Error: " + error);
    println("w: " + weights[0] + "\tbias: " + bias);
    println("dw: " + weight_derivative + "\tdbias: "+bias_derivative);
    println("------------");
    
    weights[0] += weight_derivative;
    bias += bias_derivative;
    
    return new float[] {weights[0], bias, error};
  }
  
  public float[] train(float[] features, float[] targets) {
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
      
      //println("Iteration [" + i + "]:\nWeight: " + weights[0] + "\nBias: " + bias);
    }
    float[] results = new float[] {weights[0], bias, error};
    return results;
  }
  
}
