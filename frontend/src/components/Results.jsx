import React from 'react';

const Results = ({ results }) => {
  if (!results) return null;
  if (results.error) return <div style={{color:'red'}}>{results.error}</div>;
  return (
    <div>
      <h2>Model Predictions</h2>
      <table border="1" cellPadding="6">
        <thead>
          <tr>
            <th>Model</th>
            <th>Label</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(results).map(([model, preds]) => (
            Object.entries(preds).map(([label, conf], i) => (
              <tr key={model+label}>
                <td>{model}</td>
                <td>{label}</td>
                <td>{(conf*100).toFixed(1)}%</td>
              </tr>
            ))
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Results; 