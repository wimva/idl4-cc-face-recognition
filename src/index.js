import './styles.css';
import * as faceapi from 'face-api.js';

let canvas;
let displaySize;
let faceMatcher;
const video = document.querySelector('video');

async function loadModels() {
  await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
  await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
  await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');

  const labeledFaceDescriptors = await loadLabeledImages();
  faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  startVideo();
}

async function startVideo() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
}

video.addEventListener('play', async () => {
  canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  detect();
});

async function detect() {
  const detections = await faceapi
    .detectAllFaces(video)
    .withFaceLandmarks()
    .withFaceDescriptors();

  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  const resizedDetections = faceapi.resizeResults(detections, displaySize);
  const results = resizedDetections.map((d) =>
    faceMatcher.findBestMatch(d.descriptor),
  );
  console.log(results);
  results.forEach((result, i) => {
    const box = resizedDetections[i].detection.box;
    const drawBox = new faceapi.draw.DrawBox(box, {
      label: result.toString(),
    });
    drawBox.draw(canvas);
  });

  setTimeout(detect, 100);
}

function loadLabeledImages() {
  const labels = ['Wim', 'Jonas', 'Serge'];
  return Promise.all(
    labels.map(async (label) => {
      const img = await faceapi.fetchImage(`./images/${label}.jpg`);
      const detections = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
      const descriptions = [detections.descriptor];
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    }),
  );
}

loadModels();
