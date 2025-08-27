import React from 'react';
import ReactDOM from 'react-dom/client';

function App() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>🚀 Talk Box React Interface</h1>
      <p>If you can see this, React is working!</p>
      <div style={{ backgroundColor: '#e7f5e7', padding: '10px', borderRadius: '5px', margin: '20px 0' }}>
        ✅ React is successfully rendering!
      </div>
      <p>Next step: Load the full chat component</p>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);
