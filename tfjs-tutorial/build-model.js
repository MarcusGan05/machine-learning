
//tensorflowjs for nodejs
const tf = require('@tensorflow/tfjs-node');

//training and testing data imported
const trainDataUrl = 'file://./fashion-mnist/fashion-mnist_train.csv';
const testDataUrl = 'file://./fashion-mnist/fashion-mnist_test.csv';

//mapping of mnist labels
const labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
  ];
 
// using the first 5 classess
const numOfClasses = 5;
//image properties for one piece of clothing 
const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 1;
 
//hyperparameters
const batchSize = 100;
const epochsValue = 1;



 // load and normalize data
 const loadData = function (dataUrl, batches=batchSize) {
    // normalize data values to between 0-1
    const normalize = ({xs, ys}) => {
      return {
          xs: Object.values(xs).map(x => x / 255),
          ys: ys.label
      };
    };
 
    // transform input array (xs) to 3D tensor
    // binarize output label (ys)
    const transform = ({xs, ys}) => {
      // array of zeros
      const zeros = (new Array(numOfClasses)).fill(0);
 
      return {
          xs: tf.tensor(xs, [imageWidth, imageHeight, imageChannels]),
          ys: tf.tensor1d(zeros.map((z, i) => {
              return i === ys ? 1 : 0;
          }))
      };
    };
 
    // load, normalize, transform, batch
    return tf.data
      .csv(dataUrl, {columnConfigs: {label: {isLabel: true}}})
      .map(normalize)
      .filter(f => f.ys < numOfClasses) // only use a subset of the data 
      .map(transform)
      .batch(batchSize);
  };
 
// run 
/*
    const run = async function () {
    const trainData = loadData(trainDataUrl);

    const arr = await trainData.take(1).toArray();
    arr[0].ys.print();
    arr[0].xs.print();

    
  };
  run();
*/
 
 // Define the model architecture
  const buildModel = function () {
  const model = tf.sequential();// create an empty model

  // add the model layers
  model.add(tf.layers.conv2d({
    inputShape: [imageWidth, imageHeight, imageChannels],// only takes these parameters on the first layer
    filters: 8,
    kernelSize: 5,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2
  }));
  model.add(tf.layers.conv2d({
    filters: 16,
    kernelSize: 5,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({
    poolSize: 3,
    strides: 3
  }));
  model.add(tf.layers.flatten());//data is flattened
  model.add(tf.layers.dense({
    units: numOfClasses,
    activation: 'softmax'//find probability of classes
  }));
  
  // compile the model
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

 /* run
 const run = async function () {
  const trainData = loadData(trainDataUrl);
  const model = buildModel();
  model.summary();
};
run();*/ 



 // train the model against the training data
 const trainModel = async function (model, trainingData, epochs=epochsValue) {
  const options = {
    epochs: epochs,
    verbose: 0,
    callbacks: {
      onEpochBegin: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} of ${epochs} ...`)
      },
      onEpochEnd: async (epoch, logs) => {
        console.log(`  train-set loss: ${logs.loss.toFixed(4)}`)
        console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`)
      }
    }
  };
  return await model.fitDataset(trainingData, options);
};
 // verify the model against the test data
 const evaluateModel = async function (model, testingData) {
   const result = await model.evaluateDataset(testingData);
   const testLoss = result[0].dataSync()[0];
   const testAcc = result[1].dataSync()[0];

   console.log(`  test-set loss: ${testLoss.toFixed(4)}`);
   console.log(`  test-set accuracy: ${testAcc.toFixed(4)}`);
 };

 // run
 
 const run = async function () {
  const trainData = loadData(trainDataUrl);
  const testData = loadData(testDataUrl);
  const saveModelPath = 'file://./fashion-mnist-tfjs';

  const model = buildModel();
  model.summary();

  const info = await trainModel(model, trainData);
  console.log(info);

  console.log('Evaluating model...');
  await evaluateModel(model, testData);

  console.log('Saving model...');
  await model.save(saveModelPath);
};
run();

 /* the output
  Epoch 1 of 5 ...
  train-set loss: 0.6153
  train-set accuracy: 0.7821
Epoch 2 of 5 ...
  train-set loss: 0.3369
  train-set accuracy: 0.8809
Epoch 3 of 5 ...
  train-set loss: 0.3030
  train-set accuracy: 0.8931
Epoch 4 of 5 ...
  train-set loss: 0.2832
  train-set accuracy: 0.8996
Epoch 5 of 5 ...
  train-set loss: 0.2689
  train-set accuracy: 0.9055
History {
  validationData: null,
  params: {
    epochs: 5,
    initialEpoch: null,
    samples: null,
    steps: null,
    batchSize: null,
    verbose: 0,
    doValidation: false,
    metrics: [ 'loss', 'acc' ]
  },
  epoch: [ 0, 1, 2, 3, 4 ],
  history: {
    loss: [
      0.6153354048728943,
      0.3368665277957916,
      0.3030334711074829,
      0.2832041382789612,
      0.26893311738967896
    ],
      0.7820667028427124,
      0.8808666467666626,
      0.8931333422660828,
      0.899566650390625,
      0.9055333137512207
    ]
  }
}
Evaluating model...
*/

