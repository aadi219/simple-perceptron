float[] x_train = {32,-60,86,-47,-273,4,30,23,-34,11};
float[] y_train = {89.6,-76,186.8,-52.6,-459.4,39.2,86,73.4,-29.2,51.8};


Perceptron p = new Perceptron(1);

void setup() {
  background(50);
  size(800,800);
  fill(255);
  stroke(0,255,0);
  for (int i = 0; i < x_train.length; i++){
    ellipse(x_train[i], y_train[i], 5,5);
  }
}


void draw() {
  
}

void mousePressed(){
  int batch_size = 80;
  float[] results = new float[3];
  for (int i = 0; i < batch_size; i++) {
    results = p.train(x_train, y_train);
    println("Iteration [" + i + "]:\nWeight: " + results[0] + "\nBias: " + results[1] + "\nError: " + results[2]);
  }  
  if ((9/5) - results[0] > 0.2 || 32 - results[1] > 0.2) {
    stroke(color(255 - (255 * (results[2] / 4000)),0,0));
  }
  else {
    stroke(color(0,255,0,100));
  }
  line(0, results[1], 800, (800 * results[0] + results[1]));
   // println(results[0] + " " + results[1] + " " + results[2]);
    
}
