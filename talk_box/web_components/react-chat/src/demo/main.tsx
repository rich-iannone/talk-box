import React from 'react';
import ReactDOM from 'react-dom/client';
import Chat from '../components/chat/Chat';
import ErrorBoundary from '../components/ErrorBoundary';
import '../styles/simple.css';

function App() {
  const chatConfig = {
    model: "gpt-4",
    temperature: 0.7,
    systemPrompt: "You are a helpful AI assistant powered by Talk Box and Chatlas. You can help with programming, writing, analysis, and general questions."
  };

  return (
    <div className="app-container">
      <div className="logo-container">
        <img
          src="/talk-box-logo.png"
          alt="Talk Box"
          className="logo"
        />
      </div>

      <div className="centered-chat-container">
        <div className="chat-interface">
          <ErrorBoundary>
            <Chat
              config={chatConfig}
              apiEndpoint="http://127.0.0.1:8000"
            />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
