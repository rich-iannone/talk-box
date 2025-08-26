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
    <div className="container">
      <div className="header">
        <h1 className="title">
          🚀 Talk Box React Chat Interface
        </h1>
        <p className="description">
          Modern React chat component with full Talk Box + Chatlas integration
        </p>
        <p className="subdescription">
          Same powerful backend as <code>chat.show("browser")</code>, but with modern React UI
        </p>
      </div>

      <div className="chat-container">
        <div className="chat-header">
          <h2 className="chat-title">
            Live Chat Interface
          </h2>
          <p className="chat-info">
            This interface connects to your Talk Box backend running on <code>http://127.0.0.1:8000</code>
          </p>
        </div>

        <div className="chat-interface">
          <ErrorBoundary>
            <Chat
              config={chatConfig}
              apiEndpoint="http://127.0.0.1:8000"
            />
          </ErrorBoundary>
        </div>
      </div>

      <div className="setup-box">
        <h3 className="setup-title">
          🔧 Setup Instructions
        </h3>
        <div>
          <p className="setup-step"><strong>1. Start FastAPI Backend:</strong></p>
          <code className="code-block">
            cd talk_box/web_components/python_server && python3 chat_server.py
          </code>

          <p className="setup-step"><strong>2. Start React Frontend:</strong></p>
          <code className="code-block">
            cd talk_box/web_components/react-chat && npm run dev
          </code>

          <p className="setup-step"><strong>3. Or use the seamless integration:</strong></p>
          <code className="code-block">
            import talk_box.react_chat<br/>
            bot = tb.ChatBot().model("gpt-4")<br/>
            bot.show("react")  # Auto-starts both servers!
          </code>
        </div>
      </div>

      <div className="footer">
        <p>
          ⚡ Powered by Talk Box, Chatlas, React, TypeScript, and Tailwind CSS
        </p>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
