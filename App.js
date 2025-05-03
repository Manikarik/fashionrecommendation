import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [recommendations, setRecommendations] = useState([]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:5000/upload", formData);
      setRecommendations(res.data.recommendations);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">AmazonAI</div>

        <div className="search-section">
          <input type="file" onChange={handleFileChange} className="file-input" />
          <button onClick={handleUpload} className="upload-btn">Upload & Recommend</button>
        </div>

        <div className="icons">
          <span className="icon">🛒</span>
          <span className="icon">👤</span>
        </div>
      </header>

      {/* Recommendations */}
      <div className="recommendations">
        {recommendations.map((img, idx) => (
          <div className="card" key={idx}>
            <img
              src={`http://localhost:5000/${img}`}
              alt="Recommended"
              className="card-img"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;

