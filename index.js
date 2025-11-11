// Run MLP model
async function runWeatherMLP() {
  const x = new Float32Array(10);
  for (let i = 0; i < 10; i++) {
    x[i] = parseFloat(document.getElementById(`input${i}`).value) || 0;
  }

  const tensorX = new ort.Tensor("float32", x, [1, 10]);

  try {
    const session = await ort.InferenceSession.create("./MLP_WeatherData.onnx");
    const results = await session.run({ input: tensorX });
    const output = results.output.data;

    // Get index of highest output value
    const predictedIndex = output.indexOf(Math.max(...output));
    const classes = ["Cloudy", "Rainy", "Snowy", "Sunny"]; 
    const predictedClass = classes[predictedIndex];

    // Render prediction
    const predictions = document.getElementById("predictionsMLP");
    predictions.innerHTML = `
      <h3>MLP Model Prediction</h3>
      <p><b>Predicted Weather:</b> ${predictedClass}</p>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}



// Run Deep Learning model
async function runWeatherDeep() {
  const x = new Float32Array(10);
  for (let i = 0; i < 10; i++) {
    x[i] = parseFloat(document.getElementById(`input${i}`).value) || 0;
  }

  const tensorX = new ort.Tensor("float32", x, [1, 10]);

  try {
    const session = await ort.InferenceSession.create("./Deep_WeatherData.onnx");
    const results = await session.run({ input: tensorX });
    const output = results.output.data;

    // Get index of highest output value
    const predictedIndex = output.indexOf(Math.max(...output));
    const classes = ["Cloudy", "Rainy", "Snowy", "Sunny"];
    const predictedClass = classes[predictedIndex];

    // Render prediction
    const predictions = document.getElementById("predictionsDeep");
    predictions.innerHTML = `
      <h3>Deep Learning Model Prediction</h3>
      <p><b>Predicted Weather:</b> ${predictedClass}</p>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}
