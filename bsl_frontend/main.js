let videoStream = null;
let currentWord = "";

// Start webcam
async function startWebcam() {
  const video = document.getElementById("webcam");
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = videoStream;
    video.play();
    startAutoCapture();
  } catch (err) {
    alert("Could not access webcam.");
  }
}

// Stop webcam
function stopWebcam() {
  const video = document.getElementById("webcam");
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    videoStream = null;
    stopAutoCapture();
  }
}

// Capture frame and send to backend
async function captureAndPredict() {
  const video = document.getElementById("webcam");
  if (video.videoWidth === 0 || video.videoHeight === 0) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageBase64 = canvas.toDataURL("image/png");

  try {
    const response = await fetch("http://127.0.0.1:5000/predict_word", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageBase64 })
    });
    const data = await response.json();
    document.getElementById("predicted-letter").textContent = data.prediction || "";
    document.getElementById("current-word").textContent = data.word || "";
    // Show processed image with bounding box
    if (data.image) {
      document.getElementById("processed-image").src = "data:image/png;base64," + data.image;
    }
  } catch (err) {
    document.getElementById("predicted-letter").textContent = "Error";
    document.getElementById("current-word").textContent = "";
  }
}

// Finish word
async function finishWord() {
  try {
    const response = await fetch("http://127.0.0.1:5000/finish_word", {
      method: "POST"
    });
    const data = await response.json();
    alert("Finished word: " + (data.finished_word || ""));
    document.getElementById("current-word").textContent = "";
  } catch (err) {
    alert("Error finishing word.");
  }
}

// Auto-capture every 1 second
let autoCaptureInterval = null;
function startAutoCapture() {
  if (!autoCaptureInterval) {
    autoCaptureInterval = setInterval(captureAndPredict, 1000);
  }
}
function stopAutoCapture() {
  if (autoCaptureInterval) {
    clearInterval(autoCaptureInterval);
    autoCaptureInterval = null;
  }
}

// Attach to buttons after DOM loads
window.onload = function() {
  document.getElementById("start-webcam").onclick = startWebcam;
  document.getElementById("stop-webcam").onclick = stopWebcam;
  document.getElementById("finish-word").onclick = finishWord;
};
