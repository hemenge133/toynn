let nn;
let lr_slider;
let training_data = [
  {
    inputs: [0,1],
    targets: [1]
  },
  {
    inputs: [1,0],
    targets: [1]
  },
  {
    inputs: [0,0],
    targets: [0]
  },
  {
    inputs: [1,1],
    targets: [0]
  }
];

function setup(){
  createCanvas(400,400);
  nn = new NeuralNetwork(2,4,1);
  lr_slider = createSlider(0.01,0.5,0.1,0.01);
}

function draw(){
  background(0);
  for(let i = 0; i< 100; i++){
    let data = random(training_data);
    nn.train(data.inputs, data.targets);
  }
  nn.setLearningRate(lr_slider.value());

  let resolution = 10;
  let cols = width / resolution;
  let rows = height / resolution;
  for (let i = 0; i < cols; i++){
    for (let j = 0; j < cols; j++){
      let x1 = i / cols;
      let x2 = j / rows;
      let inputs = [x1,x2];
      let y = nn.feedforward(inputs);
      fill(y*255);
      noStroke();
      rect(i*resolution, j*resolution, resolution, resolution);
    }
  }
}
















// let training_data = [
//   {
//     inputs: [0,1],
//     targets: [1]
//   },
//   {
//     inputs: [1,0],
//     targets: [1]
//   },
//   {
//     inputs: [0,0],
//     targets: [0]
//   },
//   {
//     inputs: [1,1],
//     targets: [0]
//   }
// ];
//
// function setup(){
//   let nn = new NeuralNetwork(2, 2, 1);
//   for(let i =0; i < 80000; i++){
//     let data = random(training_data);
//     nn.train(data.inputs, data.targets);
//   }
//
//   console.log(nn.feedforward([1,0]));
//   console.log(nn.feedforward([0,1]));
//   console.log(nn.feedforward([0,0]));
//   console.log(nn.feedforward([1,1]));
// }
