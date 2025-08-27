import React, { useState } from 'react';
import { clsx } from 'clsx';
import { MessageCircle, X } from 'lucide-react';
import type { TalkBoxChatProps } from '../../types';
import { useTalkBoxChat } from '../../hooks/useTalkBoxChat';
import ChatHeader from './ChatHeader';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import ChatFooter from './ChatFooter';
import { Button } from '../ui/Button';

const Chat: React.FC<TalkBoxChatProps> = ({
  config,
  apiEndpoint,
  float = false,
  popupButton,
  showBotInfo = true,
  className,
  classNames,
  components,
  onConversationStart,
  onMessageSent,
  onMessageReceived,
  onError,
}) => {
  const [isOpen, setIsOpen] = useState(!float);
  const [isExpanded, setIsExpanded] = useState(false);

  const {
    conversation,
    messages,
    isLoading,
    isTyping,
    error,
    sendMessage,
    clearConversation,
    retry,
  } = useTalkBoxChat(config, apiEndpoint);

  // Handle sending messages
  const handleSendMessage = async (content: string) => {
    try {
      await sendMessage(content);

      // Call event handlers
      if (!conversation && onConversationStart) {
        // This will be called when conversation is created in the hook
      }

      if (onMessageSent) {
        const userMessage = messages[messages.length - 1];
        if (userMessage?.role === 'user') {
          onMessageSent(userMessage);
        }
      }
    } catch (err) {
      if (onError) {
        onError(err instanceof Error ? err : new Error('Failed to send message'));
      }
    }
  };

  // React to new assistant messages
  React.useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage?.role === 'assistant' && onMessageReceived) {
      onMessageReceived(lastMessage);
    }
  }, [messages, onMessageReceived]);

  // Handle error state
  React.useEffect(() => {
    if (error && onError) {
      onError(new Error(error));
    }
  }, [error, onError]);

  const HeaderComponent = components?.header || ChatHeader;
  const InputComponent = components?.input || ChatInput;
  const FooterComponent = components?.footer || ChatFooter;

  // Chat container content
  const chatContent = (
    <div
      className={clsx(
        'chat-wrapper',
        isExpanded ? 'chat-expanded' : 'chat-normal',
        float && 'chat-floating',
        classNames?.chatContainer,
        className
      )}
    >
      <HeaderComponent
        botName={config.name || 'Talk Box Assistant'}
        isExpanded={isExpanded}
        onToggleExpanded={() => setIsExpanded(!isExpanded)}
        onClose={float ? () => setIsOpen(false) : undefined}
        className={classNames?.header}
        botConfig={config}
        showBotInfo={showBotInfo}
      />

      <MessageList
        messages={messages}
        isTyping={isTyping}
        className={clsx('chat-messages-container', classNames?.messageList)}
      />

      {error && (
        <div className="chat-error">
          <span>Error: {error}</span>
          <button
            className="chat-error-retry"
            onClick={retry}
          >
            Retry
          </button>
        </div>
      )}

      <InputComponent
        onSendMessage={handleSendMessage}
        disabled={isLoading}
        className={classNames?.input}
      />

      <FooterComponent
        isTyping={isTyping}
        className={classNames?.footer}
      />
    </div>
  );

  // Floating mode with popup button
  if (float) {
    return (
      <>
        {/* Popup button */}
        {!isOpen && (
          <div className="chat-popup-button-container">
            {popupButton || (
              <Button
                onClick={() => setIsOpen(true)}
                className={clsx(
                  'chat-popup-button',
                  classNames?.popupButton
                )}
                aria-label="Open chat"
              >
                <MessageCircle className="h-6 w-6 text-white" />
              </Button>
            )}
          </div>
        )}

        {/* Chat interface */}
        {isOpen && chatContent}
      </>
    );
  }

  // Embedded mode
  return chatContent;
};

export default Chat;
