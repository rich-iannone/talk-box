import React from 'react';
import { clsx } from 'clsx';
import type { FooterProps } from '../../types';

const ChatFooter: React.FC<FooterProps> = ({ isTyping, className }) => {
  return (
    <div className={clsx('px-4 py-2 border-t bg-gray-50 text-center', className)}>
      <div className="flex items-center justify-center space-x-2 text-xs text-gray-500">
        <span>Powered by</span>
        <span className="font-semibold text-blue-600">Talk Box</span>
      </div>
    </div>
  );
};

export default ChatFooter;
