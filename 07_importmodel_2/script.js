let model;
const imageElement = document.getElementById('image');
let mouse = { x: 0, y: 0, down: false };

function draw() {
    if (mouse.down) {
        const ctx = imageElement.getContext('2d');
        ctx.lineTo(mouse.x, mouse.y);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 10;
        ctx.stroke();
    }
}

function predict() {
    const input = tf.browser
        .fromPixels(imageElement, 1)
        .toFloat()
        .resizeNearestNeighbor([28, 28])
        .div(tf.scalar(255))
        .expandDims();
    const score = model.predict(input).dataSync();
    console.log(score);
    const pos = score.indexOf(Math.max(...score));
    for (let i = 0; i < score.length; i++) {
        const accurance = document.getElementById(`result-probability-${i}`);
        accurance.innerText = score[i].toFixed(4);
        accurance.style.color = (i === pos) ? 'red' : 'black';
    }
}

async function app() {
    imageElement.addEventListener('mousedown', (e) => {
        const rect = imageElement.getBoundingClientRect();
        mouse.x = e.pageX - rect.left;
        mouse.y = e.pageY - rect.top;
        mouse.down = true;
    });
    imageElement.addEventListener('mouseup', (e) => {
        mouse.down = false;
    });
    imageElement.addEventListener('mousemove', (e) => {
        const rect = imageElement.getBoundingClientRect();
        mouse.x = e.pageX - rect.left;
        mouse.y = e.pageY - rect.top;
        draw();
    });
    document.getElementById('predict').addEventListener('click', () => {
        predict();
    });
    document.getElementById('clear').addEventListener('click', () => {
        const ctx = imageElement.getContext('2d');
        ctx.clearRect(0, 0, 240, 240);
        ctx.beginPath();
    });
    const FILE = "tfjs/model.json";
    model = await tf.loadGraphModel(FILE);
}
app();
