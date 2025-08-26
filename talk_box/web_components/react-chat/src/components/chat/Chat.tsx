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
        'bg-white rounded-lg shadow-lg flex flex-col',
        isExpanded ? 'w-[920px] h-[880px]' : 'w-[400px] h-[600px]',
        float && 'fixed bottom-4 right-4 z-50',
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
      />

      <MessageList
        messages={messages}
        isTyping={isTyping}
        className={classNames?.messageList}
      />

      {error && (
        <div className="px-4 py-2 bg-red-50 border-t border-red-200 text-red-700 text-sm">
          <div className="flex items-center justify-between">
            <span>Error: {error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={retry}
              className="text-red-600 hover:text-red-800"
            >
              Retry
            </Button>
          </div>
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
          <div className="fixed bottom-4 right-4 z-50">
            {popupButton || (
              <Button
                onClick={() => setIsOpen(true)}
                className={clsx(
                  'w-14 h-14 rounded-full bg-blue-600 hover:bg-blue-700 shadow-lg',
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
