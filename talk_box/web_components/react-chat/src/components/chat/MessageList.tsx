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

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div
      className={clsx(
        'flex-1 overflow-y-auto p-4 space-y-4',
        className
      )}
      role="log"
      aria-label="Chat messages"
      aria-live="polite"
    >
      {messages.length === 0 ? (
        <div className="flex items-center justify-center h-full text-center">
          <div className="text-gray-500 max-w-sm">
            <div className="text-lg mb-2">💬</div>
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
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-lg px-4 py-3 flex items-center space-x-2">
                <Loader2 className="h-4 w-4 animate-spin text-gray-500" />
                <span className="text-sm text-gray-500">Typing...</span>
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
