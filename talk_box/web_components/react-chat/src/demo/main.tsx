import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import Chat from '../components/chat/Chat';
import ChatHeader from '../components/chat/ChatHeader';
import ErrorBoundary from '../components/ErrorBoundary';
import '../styles/simple.css';

function App() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [chatConfig, setChatConfig] = useState({
    name: "Talk Box Assistant",
    description: "A helpful AI assistant powered by Talk Box and Chatlas",
    model: "gpt-4",
    temperature: 0.7,
    maxTokens: 1000,
    preset: "helpful_assistant",
    persona: "a knowledgeable and friendly AI assistant",
    avoidTopics: ["harmful content", "illegal activities"],
    tools: ["web_search", "code_analysis", "file_operations"],
    systemPrompt: "You are a helpful AI assistant powered by Talk Box and Chatlas. You can help with programming, writing, analysis, and general questions."
  });
  const [configLoaded, setConfigLoaded] = useState(false);

  // Fetch bot configuration from the API on component mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/config');
        if (response.ok) {
          const config = await response.json();
          console.log('Loaded bot configuration:', config);

          // Map the API response to our expected format
          setChatConfig({
            name: config.name || "Talk Box Assistant",
            description: config.description || "A helpful AI assistant powered by Talk Box and Chatlas",
            model: config.model || "gpt-4",
            temperature: config.temperature ?? 0.7,
            maxTokens: config.maxTokens || config.max_tokens || 1000,  // Support both formats
            preset: config.preset, // Don't use fallback - preserve null if no preset is set
            persona: config.persona || "a knowledgeable and friendly AI assistant",
            avoidTopics: config.avoidTopics || config.avoid_topics || ["harmful content", "illegal activities"],  // Support both formats
            tools: config.tools || ["web_search", "code_analysis", "file_operations"],
            systemPrompt: config.systemPrompt || config.system_prompt || "You are a helpful AI assistant powered by Talk Box and Chatlas."  // Support both formats
          });
        } else {
          console.warn('Failed to fetch bot config, using defaults');
        }
      } catch (error) {
        console.warn('Error fetching bot config, using defaults:', error);
      }
      setConfigLoaded(true);
    };

    fetchConfig();
  }, []);

  // Don't render until config is loaded
  if (!configLoaded) {
    return (
      <div className="app-container loading">
        <div className="logo-container">
          <img
            src="/talk-box-logo.png"
            alt="Talk Box"
            className="logo"
          />
        </div>
        <div className="loading-message">Loading configuration...</div>
      </div>
    );
  }

  return (
    <div className={`app-container ${isExpanded ? 'expanded' : ''}`}>
      <div className="logo-container">
        <img
          src="/talk-box-logo.png"
          alt="Talk Box"
          className="logo"
        />
      </div>

      <div className={`centered-chat-container ${isExpanded ? 'expanded' : ''}`}>
        <div className={`chat-interface ${isExpanded ? 'expanded' : ''}`}>
          <ErrorBoundary>
            <Chat
              config={chatConfig}
              apiEndpoint="http://127.0.0.1:8000"
              components={{
                header: (props) => (
                  <ChatHeader
                    {...props}
                    isExpanded={isExpanded}
                    onToggleExpanded={() => setIsExpanded(!isExpanded)}
                    botConfig={chatConfig}
                    showBotInfo={true}
                  />
                )
              }}
            />
          </ErrorBoundary>
        </div>
      </div>
    </div>
  );
}ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
