import React, { useEffect, useRef } from 'react';
import { clsx } from 'clsx';
import { Loader2 } from 'lucide-react';
import type { Message } from '../../types';
import MessageComponent from '../message/Message';

interface MessageListProps {
  messages: Message[];
  isTyping?: boolean;
  className?: string;
  emptyStateMessage?: string;
}

const MessageList: React.FC<MessageListProps> = ({
  messages,
  isTyping = false,
  className,
  emptyStateMessage = "👋 Hi! I'm your Talk Box assistant. How can I help you today?",
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const prevMessagesLengthRef = useRef(messages.length);

  // Auto-scroll to bottom when new messages arrive (but not when just typing status changes)
  useEffect(() => {
    // Only scroll if a new message was actually added
    if (messages.length > prevMessagesLengthRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      prevMessagesLengthRef.current = messages.length;
    }
  }, [messages]);

  // Separate effect for typing indicator - only scroll if already at bottom
  useEffect(() => {
    if (isTyping) {
      const container = messagesEndRef.current?.parentElement;
      if (container) {
        const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
        if (isNearBottom) {
          messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
      }
    }
  }, [isTyping]);

  return (
    <div
      className={clsx(
        'message-list',
        className
      )}
      role="log"
      aria-label="Chat messages"
      aria-live="polite"
    >
      {messages.length === 0 ? (
        <div className="message-list-empty">
          <div className="message-list-empty-content">
            <p>{emptyStateMessage}</p>
          </div>
        </div>
      ) : (
        <>
          {messages.map((message) => (
            <MessageComponent
              key={message.id}
              message={message}
              isUser={message.role === 'user'}
            />
          ))}

          {isTyping && (
            <div className="typing-indicator">
              <div className="message-bubble message-assistant">
                <Loader2 className="typing-spinner" />
                <span>Typing...</span>
              </div>
            </div>
          )}
        </>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;
