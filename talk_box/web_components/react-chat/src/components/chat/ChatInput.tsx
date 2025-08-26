import React, { useState, useRef } from 'react';
import { Send, Loader2 } from 'lucide-react';
import type { ChatInputProps } from '../../types';

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = 'Type your message...',
  className,
}) => {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!message.trim() || disabled || isLoading) return;

    setIsLoading(true);
    try {
      await onSendMessage(message);
      setMessage('');
      // Remove auto-focus to prevent unnecessary scrolling
      // textareaRef.current?.focus();
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={`chat-input-form ${className || ''}`}>
      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isLoading}
            className="chat-textarea"
            rows={1}
          />
        </div>
        <button
          type="submit"
          disabled={!message.trim() || disabled || isLoading}
          className={`chat-send-button ${(!message.trim() || disabled || isLoading) ? 'chat-send-button-disabled' : ''}`}
          aria-label="Send message"
        >
          {isLoading ? (
            <Loader2 className="chat-send-icon spinning" />
          ) : (
            <Send className="chat-send-icon" />
          )}
        </button>
      </div>
    </form>
  );
};

export default ChatInput;
