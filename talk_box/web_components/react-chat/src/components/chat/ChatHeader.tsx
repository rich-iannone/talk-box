import React from 'react';
import { clsx } from 'clsx';
import { X, Maximize2, Minimize2, Info, MessageSquare } from 'lucide-react';
import type { HeaderProps } from '../../types';
import { Button } from '../ui/Button';
import { Popover } from '../ui/Popover';

const ChatHeader: React.FC<HeaderProps> = ({
  botName = 'Talk Box Assistant',
  isExpanded = false,
  onToggleExpanded,
  onClose,
  className,
  botConfig,
  showBotInfo = true,
}) => {
  const renderBotInfo = () => {
    if (!botConfig) return null;

    return (
      <div className="bot-info-popover">
        <div className="bot-info-section">
          <h4 className="bot-info-title">Bot Configuration</h4>

          {botConfig.name && (
            <div className="bot-info-item">
              <span className="bot-info-label">Name:</span>
              <span className="bot-info-value">{botConfig.name}</span>
            </div>
          )}

          {botConfig.description && (
            <div className="bot-info-item">
              <span className="bot-info-label">Description:</span>
              <span className="bot-info-value">{botConfig.description}</span>
            </div>
          )}

          {botConfig.model && (
            <div className="bot-info-item">
              <span className="bot-info-label">Model:</span>
              <span className="bot-info-value">{botConfig.model}</span>
            </div>
          )}

          {botConfig.temperature !== undefined && (
            <div className="bot-info-item">
              <span className="bot-info-label">Temperature:</span>
              <span className="bot-info-value">{botConfig.temperature}</span>
            </div>
          )}

          {botConfig.maxTokens && (
            <div className="bot-info-item">
              <span className="bot-info-label">Max Tokens:</span>
              <span className="bot-info-value">{botConfig.maxTokens}</span>
            </div>
          )}

          {botConfig.preset && (
            <div className="bot-info-item">
              <span className="bot-info-label">Preset:</span>
              <span className="bot-info-value">{botConfig.preset}</span>
            </div>
          )}
        </div>

        {botConfig.persona && (
          <div className="bot-info-section">
            <h4 className="bot-info-title">Persona</h4>
            <p className="bot-info-text">{botConfig.persona}</p>
          </div>
        )}

        {botConfig.tools && botConfig.tools.length > 0 && (
          <div className="bot-info-section">
            <h4 className="bot-info-title">Available Tools</h4>
            <div className="bot-info-tags">
              {botConfig.tools.map((tool) => (
                <span key={tool} className="bot-info-tag">{tool}</span>
              ))}
            </div>
          </div>
        )}

        {botConfig.avoidTopics && botConfig.avoidTopics.length > 0 && (
          <div className="bot-info-section">
            <h4 className="bot-info-title">Avoid Topics</h4>
            <div className="bot-info-tags">
              {botConfig.avoidTopics.map((topic) => (
                <span key={topic} className="bot-info-tag avoid-topic">{topic}</span>
              ))}
            </div>
          </div>
        )}

        {botConfig.systemPrompt && (
          <div className="bot-info-section">
            <h4 className="bot-info-title">System Prompt</h4>
            <div className="bot-info-item" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
              <Popover
                content={
                  <div className="system-prompt-popover">
                    <h5 className="system-prompt-title">Full System Prompt</h5>
                    <div className="system-prompt-content">
                      <pre className="system-prompt-text">{botConfig.systemPrompt}</pre>
                    </div>
                  </div>
                }
                contentClassName="system-prompt-popover-content"
              >
                <button className="system-prompt-trigger">
                  <MessageSquare className="h-3 w-3" />
                  <span>View full prompt</span>
                </button>
              </Popover>
              <p className="bot-info-text system-prompt-preview">
                {botConfig.systemPrompt.length > 100
                  ? `${botConfig.systemPrompt.substring(0, 100)}...`
                  : botConfig.systemPrompt}
              </p>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className={clsx(
        'chat-header',
        className
      )}
    >
      <div className="chat-header-info">
        {showBotInfo && botConfig ? (
          <Popover
            content={renderBotInfo()}
            className="chat-header-avatar-container"
            contentClassName="bot-info-popover-content"
          >
            <div className="chat-header-avatar info-icon">
              <Info className="h-4 w-4" />
            </div>
          </Popover>
        ) : (
          <div className="chat-header-avatar">
            {botName.charAt(0).toUpperCase()}
          </div>
        )}
        <h3 className="chat-header-title">{botName}</h3>
      </div>

      <div className="chat-header-actions">
        {onToggleExpanded && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleExpanded}
            className="chat-header-button"
            aria-label={isExpanded ? 'Minimize chat' : 'Expand chat'}
          >
            {isExpanded ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </Button>
        )}
        {onClose && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="chat-header-button"
            aria-label="Close chat"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
};

export default ChatHeader;
