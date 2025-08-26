import ReactDOM from 'react-dom/client';

function App() {
  return (
    <div style={{ padding: '20px' }}>
      <h1>Minimal React Test</h1>
      <p>This is a very basic React component.</p>
    </div>
  );
}

const container = document.getElementById('root');
const root = ReactDOM.createRoot(container!);
root.render(<App />);
