//bring in tensorflow, the results interface, and MNIST batches from data.js
import * as tf from '@tensorflow/tfjs';
import { MnistData } from './data';
import * as ui from './ui';

//in a sequential model the outputs of the previous layer are used as the inputs for the following layer
const model = tf.sequential();

//here we use a convolutional 2D layer and define the input shape
//the activation function we're going to use is known as a rectified linear unit
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2], 
  strides: [2, 2]
}));

//we will configure the second layer the same as the input layer
model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model.add(tf.layers.maxPooling2d({
  poolSize: [2, 2], 
  strides: [2, 2]
}));

//the final layer will flatten out the data and use the softmax activation function
model.add(tf.layers.flatten());
model.add(tf.layers.dense({
    units: 10, 
    kernelInitializer: 'varianceScaling', 
    activation: 'softmax'
  }));

//in this example we are going to use stochastic gradient descent as the optimizer function
const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;

//every few batches, test the accuracy over many examples
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;

//begin training with the model configuration defined above
async function train() {
  ui.isTraining();

  const lossValues = [];
  const accuracyValues = [];

  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const batch = data.nextTrainBatch(BATCH_SIZE);

    let testBatch;
    let validationData;
    //est the accuracy of the mode every few batches
    if (i % TEST_ITERATION_FREQUENCY === 0) {
      testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
      validationData = [
        testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
      ];
    }

    //because the dataset is too large for memory we'll call fit repeadedly
    const history = await model.fit(
        batch.xs.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1});

    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

    //plot the loss and accuracy
    lossValues.push({'batch': i, 'loss': loss, 'set': 'train'});
    ui.plotLosses(lossValues);

    if (testBatch != null) {
      accuracyValues.push({'batch': i, 'accuracy': accuracy, 'set': 'train'});
      ui.plotAccuracies(accuracyValues);
    }

    batch.xs.dispose();
    batch.labels.dispose();
    if (testBatch != null) {
      testBatch.xs.dispose();
      testBatch.labels.dispose();
    }

    await tf.nextFrame();
  }
}

async function showPredictions() {
  const testExamples = 100;
  const batch = data.nextTestBatch(testExamples);

  tf.tidy(() => {
    const output = model.predict(batch.xs.reshape([-1, 28, 28, 1]));

    const axis = 1;
    const labels = Array.from(batch.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(batch, predictions, labels);
  });
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function mnist() {
  window.onscroll = function() {myFunction()};
      
  var navbar = document.getElementById("navbar");
  var sticky = navbar.offsetTop;
  
  function myFunction() {
    if (window.pageYOffset >= sticky) {
      navbar.classList.add("sticky")
    } else {
      navbar.classList.remove("sticky");
    }
  }
  await load();
  await train();
  showPredictions();
}
mnist();