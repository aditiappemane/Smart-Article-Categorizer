import React, { useState } from 'react';
import InputForm from './components/InputForm';
import Results from './components/Results';
import EmbeddingViz from './components/EmbeddingViz';
import './App.css';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  return (
    <div className="App">
      <h1>Smart Article Categorizer</h1>
      <InputForm setResults={setResults} setLoading={setLoading} />
      {loading && <p>Loading...</p>}
      <Results results={results} />
      <EmbeddingViz />
    </div>
  );
}

export default App;
