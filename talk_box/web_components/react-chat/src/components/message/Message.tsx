import React from 'react';
import { clsx } from 'clsx';
import type { MessageProps } from '../../types';
import MessageMarkdown from './MessageMarkdown';
import { formatTimestamp } from '../../utils/api';

const Message: React.FC<MessageProps> = ({ message, isUser = false, className }) => {
  const { content, timestamp } = message;

  return (
    <div
      className={clsx(
        'flex w-full',
        isUser ? 'justify-end' : 'justify-start',
        className
      )}
    >
      <div
        className={clsx(
          'max-w-[80%] rounded-lg px-4 py-2 shadow-sm',
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-gray-100 text-gray-900'
        )}
      >
        <div className="space-y-1">
          {isUser ? (
            <div className="whitespace-pre-wrap break-words">
              {content}
            </div>
          ) : (
            <MessageMarkdown>{content}</MessageMarkdown>
          )}
          <div
            className={clsx(
              'text-xs',
              isUser ? 'text-blue-100' : 'text-gray-500'
            )}
          >
            {formatTimestamp(timestamp)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Message;
