import React, { useState } from 'react';
import { classifyArticle } from '../api';

const InputForm = ({ setResults, setLoading }) => {
  const [text, setText] = useState('');
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await classifyArticle(text);
      setResults(res.predictions);
    } catch (err) {
      setResults({ error: 'Failed to classify article.' });
    }
    setLoading(false);
  };
  return (
    <form onSubmit={handleSubmit}>
      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="Paste your article here..."
        rows={8}
        style={{ width: '100%' }}
      />
      <button type="submit">Classify</button>
    </form>
  );
};

export default InputForm; 