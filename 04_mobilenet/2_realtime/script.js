let net;
let webcam;
const webcamElement = document.getElementById('webcam');
const resultElement = document.getElementById('result');

async function init() {
  resultElement.style.display = 'none';
  console.log('mobilenetを読み込み中です...');
  net = await mobilenet.load();
  console.log('mobilenetの読み込みが終了しました');
  webcam = await tf.data.webcam(webcamElement, {facingMode: 'environment'});
}

init();
const interval = setInterval(async () => {
  const img = await webcam.capture();
  const result = await net.classify(img);
  for (let i = 0; i < result.length; i++) {
    document.getElementById(`result-class-${i + 1}`).innerText = result[i].className;
    document.getElementById(`result-probability-${i + 1}`).innerText = result[i].probability.toFixed(4);
  }
  resultElement.style.display = 'block';
  console.log(result);
  img.dispose();
  await tf.nextFrame();
}, 1000);
