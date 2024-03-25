/*
  Author: Aadi Badola
  Description: 
  The program utilizes a Perceptron class along with the Processing 4.3 library to perform and visualise simple regression on Celsius & Farhenheit values.
  The purpose of the program was to understand the fundamentals of training a neural network by simplifying the process making use of a single neuron.
  The calculations for backpropagation performed and verified manually and the Processing library was utilized in order to visualize the gradual training of the neuron.
*/

// Initializing a set of train and test values to fit the model on.
// In this case x_train and y_train are parallel arrays with x_train being the Celsius values and y_train being the corresponding Farhenheit values.
float[] x_train = {32,-60,86,-47,-273,4,30,23,-34,11};
float[] y_train = {89.6,-76,186.8,-52.6,-459.4,39.2,86,73.4,-29.2,51.8};


Perceptron p = new Perceptron(1);

void setup() {
  background(50);
  size(800,800);
  fill(255);
  stroke(0,255,0);
  // creating markers to visualise the coordinates of our training data to see the progress of the neuron's training.
  for (int i = 0; i < x_train.length; i++){
    ellipse(x_train[i], y_train[i], 5,5);
  }
}


void draw() {
  
}

void mousePressed(){
  /*
    Training of the neuron is performed on every mouse click.
    The batch_size variable controls how many times the neuron will train over the provided x_train and y_train sets.
    A batch_size of 50 means that on every click, the neuron is trained on the entire dataset provided 50 times.
    Increasing batch_size would lead to larger jumps in the graph and conversely, 
    decreasing it would require a greater amount of mouse clicks to train the network to an adequate accuracy.
  */
  int batch_size = 50;
  float[] results = new float[3];
  for (int i = 0; i < batch_size; i++) {
    results = p.train(x_train, y_train);
    println("Iteration [" + i + "]:\nWeight: " + results[0] + "\nBias: " + results[1] + "\nError: " + results[2]);
  }  
  // the line created by the neuron is dark and red if the error is greater and the provided threshold (0.2).
  // the line gradually becomes brighter as the error approaches the threshold and becomes green once the error is under a reasonable amount.
  if ((9/5) - results[0] > 0.2 || 32 - results[1] > 0.2) {
    stroke(color(255 - (255 * (results[2] / 4000)),0,0));
  }
  else {
    stroke(color(0,255,0,100));
  }
  line(0, results[1], 800, (800 * results[0] + results[1]));
    
}
