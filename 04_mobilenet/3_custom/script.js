let net;
let classifier;
let webcam;
const webcamElement = document.getElementById('webcam');
const resultElement = document.getElementById('result');

async function init() {
  resultElement.style.display = 'none';
  net = await mobilenet.load();
  classifier = knnClassifier.create();
  webcam = await tf.data.webcam(webcamElement, {facingMode: 'environment'});
}

async function addExample(classId) {
    const img = await webcam.capture();
    const activation = net.infer(img, true);
    classifier.addExample(activation, classId);
    img.dispose();
};

init();
document.getElementById('class-rock').addEventListener('click',
  () => addExample('Rock'));
document.getElementById('class-scissor').addEventListener('click',
  () => addExample('Scissor'));
document.getElementById('class-paper').addEventListener('click',
  () => addExample('Paper'));

const interval = setInterval(async () => {
  if (classifier.getNumClasses() > 0) {
    const img = await webcam.capture();
    const activation = net.infer(img, 'conv_preds');
    const result = await classifier.predictClass(activation);
    //const classes = ['Rock', 'Paper', 'C'];
    document.getElementById(`result-class`).innerText = result.label;
    document.getElementById(`result-probability`).innerText = result.confidences[result.label].toFixed(4);
    resultElement.style.display = 'block';
    console.log(result);
    img.dispose();
  }
  await tf.nextFrame();
}, 1000);
