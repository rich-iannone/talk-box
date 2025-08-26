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
        'flex items-center justify-between px-4 py-3 bg-blue-600 text-white rounded-t-lg',
        className
      )}
    >
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-sm font-semibold">
          {botName.charAt(0).toUpperCase()}
        </div>
        <h3 className="font-medium text-sm">{botName}</h3>
      </div>

      <div className="flex items-center space-x-1">
        {onToggleExpanded && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleExpanded}
            className="text-white hover:bg-blue-500 h-8 w-8 p-0"
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
            className="text-white hover:bg-blue-500 h-8 w-8 p-0"
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
