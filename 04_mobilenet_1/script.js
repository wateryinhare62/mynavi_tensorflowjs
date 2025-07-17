let net;
const selecterElement = document.getElementById('photos');
const imageElement = document.getElementById('img');
const resultElement = document.getElementById('result');

async function init() {
  console.log('mobilenetを読み込み中です...');
  net = await mobilenet.load();
  console.log('mobilenetの読み込みが終了しました');
  selecterElement.disabled = false;
}

async function classifyImage() {
  resultElement.style.display = 'none';
  const result = await net.classify(imageElement);
  for (let i = 0; i < result.length; i++) {
    document.getElementById(`result-class-${i + 1}`).innerText = result[i].className;
    document.getElementById(`result-probability-${i + 1}`).innerText = result[i].probability.toFixed(4);
  }
  resultElement.style.display = 'block';
  console.log(result);
}

selecterElement.addEventListener('change', () => {
  const value = selecterElement.value;
  if (value !== 'default') {
    imageElement.src = `${value}.jpg`;
  }
});

imageElement.addEventListener('load', () => {
  classifyImage();
});

init();
