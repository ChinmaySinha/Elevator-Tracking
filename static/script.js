const video = document.getElementById('camera');
const captureBtn = document.getElementById('capture-btn');
const predictionDiv = document.getElementById('prediction');

// Start the camera
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

captureBtn.addEventListener('click', async () => {
  // Capture a frame from the video feed
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert the captured frame to base64
  const imageBase64 = canvas.toDataURL('image/jpeg').split(',')[1];

  // Send the image to the server for prediction
  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageBase64 })
  });

  const result = await response.json();
  predictionDiv.innerText = `Predicted Number: ${result.prediction}`;
});
