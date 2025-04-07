console.log('こんにちは、TensorFlow.js！');

// 入力データと出力データのテンソルの作成
const inputsArray = [100, 200, 300, 400, 500];
const labelsArray = [920, 780, 720, 580, 500];
const maxInputs = tf.scalar(Math.max(...inputsArray));
const maxLabels = tf.scalar(Math.max(...labelsArray));
const inputs = tf.tensor1d(inputsArray).div(maxInputs);
const labels = tf.tensor1d(labelsArray).div(maxLabels);
inputs.print();
labels.print();

// モデルと層の作成
const model = tf.sequential();
model.add(tf.layers.dense({
  units: 1,
  inputShape: [1],
}));
console.log(model.summary());

// モデルのコンパイル
model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError',
});

// モデルの訓練関数
async function trainModel () {
  await model.fit(inputs, labels, {
    batchSize: 5,
    epochs: 2000,
  });
  console.log('訓練終了！')
  // 予測の実施
  const testInput = tf.tensor1d([600]).div(maxInputs);
  const prediction = model.predict(testInput);
  prediction.mul(maxLabels).print();
}

// モデルの訓練
console.log('訓練開始...');
trainModel();
