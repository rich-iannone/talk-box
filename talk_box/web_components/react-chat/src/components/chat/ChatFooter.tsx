import React from 'react';
import { clsx } from 'clsx';
import type { FooterProps } from '../../types';

const ChatFooter: React.FC<FooterProps> = ({ className }) => {
  return (
    <div className={clsx('chat-footer', className)}>
      <div className="chat-footer-content">
        <span className="powered-by-text">Powered by</span>
        <span className="talk-box-brand">Talk Box</span>
      </div>
    </div>
  );
};

export default ChatFooter;
