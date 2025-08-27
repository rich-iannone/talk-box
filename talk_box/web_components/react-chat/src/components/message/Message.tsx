import React from 'react';
import type { MessageProps } from '../../types';
import MessageMarkdown from './MessageMarkdown';
import { formatTimestamp } from '../../utils/api';

const Message: React.FC<MessageProps> = ({ message, isUser = false, className }) => {
  const { content, timestamp } = message;

  return (
    <div className={`message-container ${isUser ? 'message-user' : 'message-assistant'} ${className || ''}`}>
      <div className={`message-bubble ${isUser ? 'message-bubble-user' : 'message-bubble-assistant'}`}>
        <div className="message-content">
          {isUser ? (
            <div className="message-text">
              {content}
            </div>
          ) : (
            <MessageMarkdown>{content}</MessageMarkdown>
          )}
          <div className={`message-timestamp ${isUser ? 'message-timestamp-user' : 'message-timestamp-assistant'}`}>
            {formatTimestamp(timestamp)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Message;
