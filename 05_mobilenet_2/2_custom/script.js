let net;
let classifier;
let webcam;
const webcamElement = document.getElementById('webcam');
const resultElement = document.getElementById('result');
const classNames = ['rock', 'scissor', 'paper'];

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
  const counts = classifier.getClassExampleCount();
  document.getElementById(`count-${classId}`).innerText = counts[classId] || 0;
};

async function clearClass(classId) {
  classifier.clearClass(classId);
  document.getElementById(`count-${classId}`).innerText = 0;
}

init();
for (const className of classNames) {
  document.getElementById(`class-${className}`).addEventListener('click',
    () => addExample(className));
  document.getElementById(`clear-${className}`).addEventListener('click',
    () => clearClass(className));
}

const interval = setInterval(async () => {
  if (classifier.getNumClasses() > 0) {
    const img = await webcam.capture();
    const activation = net.infer(img, 'conv_preds');
    const result = await classifier.predictClass(activation);
    document.getElementById(`result-class`).innerText = result.label;
    document.getElementById(`result-probability`).innerText = result.confidences[result.label].toFixed(4);
    resultElement.style.display = 'block';
    console.log(result);
    img.dispose();
  }
  await tf.nextFrame();
}, 1000);
