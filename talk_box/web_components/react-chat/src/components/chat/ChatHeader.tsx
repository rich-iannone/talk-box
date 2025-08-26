import React from 'react';
import { clsx } from 'clsx';
import { X, Maximize2, Minimize2 } from 'lucide-react';
import type { HeaderProps } from '../../types';
import { Button } from '../ui/Button';

const ChatHeader: React.FC<HeaderProps> = ({
  botName = 'Talk Box Assistant',
  isExpanded = false,
  onToggleExpanded,
  onClose,
  className,
}) => {
  return (
    <div
      className={clsx(
        'chat-header',
        className
      )}
    >
      <div className="chat-header-info">
        <div className="chat-header-avatar">
          {botName.charAt(0).toUpperCase()}
        </div>
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
