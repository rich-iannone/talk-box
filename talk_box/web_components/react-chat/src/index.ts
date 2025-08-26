import Chat from './components/chat/Chat';

export default Chat;

// Export all types and components for users who want to customize
export type {
  Message,
  Conversation,
  ChatBotConfig,
  TalkBoxChatProps,
  HeaderProps,
  MessageProps,
  ChatInputProps,
  FooterProps,
  UseTalkBoxChatReturn,
} from './types';

export { useTalkBoxChat } from './hooks/useTalkBoxChat';
export { TalkBoxAPI } from './utils/api';

// Export individual components for customization
export { default as ChatHeader } from './components/chat/ChatHeader';
export { default as MessageList } from './components/chat/MessageList';
export { default as ChatInput } from './components/chat/ChatInput';
export { default as ChatFooter } from './components/chat/ChatFooter';
export { default as MessageComponent } from './components/message/Message';
export { default as MessageMarkdown } from './components/message/MessageMarkdown';

// Export UI components
export { Button } from './components/ui/Button';
export { Textarea } from './components/ui/Textarea';
