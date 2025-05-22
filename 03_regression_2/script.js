// 各種のグローバル変数
const bodyElement = document.querySelector('body');
const buttonElement = document.querySelector('button');
let data = null;
let model = null;
let tensorData = null;

// ドキュメント読み込み後に起動される関数
async function train() {
  bodyElement.insertAdjacentHTML('beforeend', '<p>処理開始</p>');

  data = await readData();
  bodyElement.insertAdjacentHTML('beforeend', '<p>オリジナルデータのプロット完了</p>');

  model = createModel();
  tfvis.show.modelSummary({name: 'モデルの概要'}, model);
  bodyElement.insertAdjacentHTML('beforeend', '<p>モデルの作成完了</p>');

  bodyElement.insertAdjacentHTML('beforeend', '<p>訓練開始</p>');
  tensorData = prepareTraining(data);
  const {inputTensor, labelTensor} = tensorData;
  await trainModel(model, inputTensor, labelTensor);
  bodyElement.insertAdjacentHTML('beforeend', '<p>訓練終了</p>');

  buttonElement.disabled = false;
  bodyElement.insertAdjacentHTML('beforeend', '<p>予測を実行できます。</p>');
}
document.addEventListener('DOMContentLoaded', train);

// ボタンクリックで起動される関数
async function predict() {
  const [powers, predicts] = doPredict(model, tensorData);
  drawResult(data, powers, predicts);
}
buttonElement.addEventListener('click', predict);

// データを読み込む関数
async function readData() {
  const carDataRes = await fetch(
    'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
  );
  const carData = await carDataRes.json();
  const cleanedData = carData
    .map(car => ({
      kpl: car.Miles_per_Gallon * 1.60934 / 3.78541,
      outputPower: car.Horsepower * 0.7355
    }))
    .filter(car => car.kpl != 0 && car.outputPower != 0);
 
  const values = cleanedData.map(d => ({
    x: d.outputPower,
    y: d.kpl
  }));

  tfvis.render.scatterplot(
    {
      name: 'エンジン出力（kW）と燃費（km/ℓ）の相関関係'
    },
    {
      values: [values],
      series: ['kW-km/ℓ']
    },
    {
      xLabel: 'エンジン出力（kW）',
      yLabel: '燃費（km/ℓ）',
      height: 300
    }
  );
  return cleanedData;
}

// モデルを作成する関数
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1], useBias: true}));
  model.add(tf.layers.dense({units: 1, useBias: true}));
  return model;
}

// 訓練を準備する関数
function prepareTraining(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputArray = data.map(d => d.outputPower);
    const labelArray = data.map(d => d.kpl);
    const inputTensor = tf.tensor1d(inputArray);
    const labelTensor = tf.tensor1d(labelArray);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();
    const normalizedInputTensor = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabelTensor = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputTensor: normalizedInputTensor,
      labelTensor: normalizedLabelTensor,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

// モデルを訓練する関数
async function trainModel(model, inputTensor, labelTensor) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  await model.fit(inputTensor, labelTensor, {
    batchSize: 32,
    epochs: 100,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      {
        name: '訓練の成果'
      },
      ['loss', 'mse'],
      {
        height: 200,
        callbacks: ['onEpochEnd']
      }
    )
  });
}

// 予測を実行する関数
function doPredict(model, normalizedData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizedData;
  return tf.tidy(() => {
    const powers = tf.linspace(0, 1, 100);
    const predicts = model.predict(powers);
    const unnormalizedPowers = powers.mul(inputMax.sub(inputMin)).add(inputMin);
    const unnormalizedPredicts = predicts.mul(labelMax.sub(labelMin)).add(labelMin);
    return [unnormalizedPowers.dataSync(), unnormalizedPredicts.dataSync()];
  });
}

// 予測結果を表示する関数
function drawResult(inputData, powers, predicts) {
  const originalPoints = inputData.map(d => ({
    x: d.outputPower, y: d.kpl,
  }));

  let outputPower = Number(document.getElementById('output_power').value).toFixed(0);
  if(outputPower < 0 || outputPower > 180) {
    outputPower = 0;
  }
  let predictedPoints = null;
  if(outputPower == 0) {
    predictedPoints = Array.from(powers).map((val, i) => {
      return {x: val, y: predicts[i]};
    });
  } else {
    for(let i = 0; i < powers.length-1; i++) {
      if(powers[i] >= outputPower) {
        predictedPoints = [{x: outputPower, y: predicts[i]}];
        break;
      }    
    }
  }

  tfvis.render.scatterplot(
    {
      name: 'オリジナル vs 予測'
    },
    {
      values: [originalPoints, predictedPoints],
      series: ['オリジナル', '予測']
    },
    {
      xLabel: 'エンジン出力（kW）',
      yLabel: '燃費（km/ℓ）',
      height: 300
    }
  );
  if(outputPower == 0) {
    bodyElement.insertAdjacentHTML('beforeend', '<p>100件の燃費を予測しました。</p>');
  } else {
    bodyElement.insertAdjacentHTML('beforeend',
      `<p>エンジン出力 ${outputPower} kW の燃費を ` +
      `${predictedPoints[0].y.toFixed(1)} km/ℓと予測しました。</p>`);
  }
}

