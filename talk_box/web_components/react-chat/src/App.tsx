import React from 'react';
import TalkBoxChat from './index';
import type { ChatBotConfig } from './types';

// Example configuration
const exampleConfig: ChatBotConfig = {
  name: 'Talk Box Assistant',
  description: 'A helpful AI assistant powered by Talk Box',
  model: 'gpt-4',
  temperature: 0.7,
  maxTokens: 1000,
  preset: 'helpful',
  persona: 'a knowledgeable and friendly AI assistant',
  avoidTopics: ['harmful content', 'illegal activities'],
  tools: ['web_search', 'code_analysis'],
};

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      {/* Interface header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-2xl font-bold text-gray-900">
            Talk Box React Chat Interface
          </h1>
          <p className="text-gray-600 mt-1">
            Interactive example of the Talk Box React chat component
          </p>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Embedded chat example */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Embedded Chat</h2>
            <div className="h-[500px]">
              <TalkBoxChat
                config={exampleConfig}
                onConversationStart={(conversation) => {
                  console.log('Conversation started:', conversation);
                }}
                onMessageSent={(message) => {
                  console.log('Message sent:', message);
                }}
                onMessageReceived={(message) => {
                  console.log('Message received:', message);
                }}
                onError={(error) => {
                  console.error('Chat error:', error);
                }}
              />
            </div>
          </div>

          {/* Configuration panel */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold mb-4">Configuration</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-sm text-gray-700 mb-2">
                  Bot Configuration
                </h3>
                <dl className="text-sm space-y-1">
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Name:</dt>
                    <dd className="font-medium">{exampleConfig.name}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Model:</dt>
                    <dd className="font-medium">{exampleConfig.model}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Temperature:</dt>
                    <dd className="font-medium">{exampleConfig.temperature}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Preset:</dt>
                    <dd className="font-medium">{exampleConfig.preset}</dd>
                  </div>
                </dl>
              </div>

              <div>
                <h3 className="font-medium text-sm text-gray-700 mb-2">
                  Persona
                </h3>
                <p className="text-sm text-gray-600">{exampleConfig.persona}</p>
              </div>

              <div>
                <h3 className="font-medium text-sm text-gray-700 mb-2">
                  Available Tools
                </h3>
                <div className="flex flex-wrap gap-1">
                  {exampleConfig.tools?.map((tool) => (
                    <span
                      key={tool}
                      className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded"
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="font-medium text-sm text-gray-700 mb-2">
                  Avoid Topics
                </h3>
                <div className="flex flex-wrap gap-1">
                  {exampleConfig.avoidTopics?.map((topic) => (
                    <span
                      key={topic}
                      className="px-2 py-1 bg-red-100 text-red-800 text-xs rounded"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Floating chat example */}
        <div className="mt-8 bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Floating Chat</h2>
          <p className="text-gray-600 mb-4">
            The floating chat appears as a popup button in the bottom-right corner.
            Click the button below to enable it:
          </p>

          <button
            onClick={() => {
              // Create floating chat instance
              const floatingChat = document.createElement('div');
              document.body.appendChild(floatingChat);

              // Note: In a real implementation, you'd use React.createRoot or ReactDOM.render
              console.log('Floating chat would be enabled here');
            }}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors"
          >
            Enable Floating Chat
          </button>
        </div>
      </main>

      {/* Floating chat component (example) */}
      <TalkBoxChat
        config={exampleConfig}
        float={true}
        onConversationStart={(conversation) => {
          console.log('Floating chat conversation started:', conversation);
        }}
      />
    </div>
  );
}

export default App;
