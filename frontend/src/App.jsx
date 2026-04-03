import { useState } from "react";
import "./App.css";
fetch(`${import.meta.env.VITE_API_URL}/predict`)

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setResult("");

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("https://ai-image-detector-lpi0.onrender.com/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      setResult(
        `${data.label} (${(data.confidence * 100).toFixed(2)}%)`
      );
    } catch (err) {
      setResult("Something went wrong");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <div className="card">
        <h1>AI Image Detector</h1>

        <input type="file" onChange={handleFileChange} />

        {preview && <img src={preview} alt="preview" />}

        <button onClick={handleUpload}>
          {loading ? "Processing..." : "Predict"}
        </button>

        {result && (
          <div
            className={`result ${result.includes("REAL") ? "real" : "fake"
              }`}
          >
            {result}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;