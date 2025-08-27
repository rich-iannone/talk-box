import React from 'react';

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date | string | number;
  metadata?: Record<string, unknown>;
}

export interface Conversation {
  id: string;
  messages: Message[];
  metadata?: Record<string, unknown>;
}

export interface ChatBotConfig {
  name?: string;
  description?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  preset?: string;
  persona?: string;
  avoidTopics?: string[];
  tools?: string[];
  systemPrompt?: string;
}

export interface TalkBoxChatProps {
  /** Configuration for the Talk Box chatbot */
  config: ChatBotConfig;

  /** API endpoint for Talk Box server */
  apiEndpoint?: string;

  /** Whether to show as a floating popup button */
  float?: boolean;

  /** Custom popup button component */
  popupButton?: React.ReactNode;

  /** Whether to show bot info icon with configuration details */
  showBotInfo?: boolean;

  /** Chat container className */
  className?: string;

  /** Custom styling classes */
  classNames?: {
    chatContainer?: string;
    header?: string;
    messageList?: string;
    userMessage?: string;
    assistantMessage?: string;
    input?: string;
    footer?: string;
    popupButton?: string;
  };

  /** Custom component overrides */
  components?: {
    header?: React.ComponentType<HeaderProps>;
    userMessage?: React.ComponentType<MessageProps>;
    assistantMessage?: React.ComponentType<MessageProps>;
    input?: React.ComponentType<ChatInputProps>;
    footer?: React.ComponentType<FooterProps>;
  };

  /** Event handlers */
  onConversationStart?: (conversation: Conversation) => void;
  onMessageSent?: (message: Message) => void;
  onMessageReceived?: (message: Message) => void;
  onError?: (error: Error) => void;
}

export interface HeaderProps {
  botName?: string;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
  onClose?: () => void;
  className?: string;
  /** Bot configuration to display in info popover */
  botConfig?: ChatBotConfig;
  /** Whether to show the info icon with bot details */
  showBotInfo?: boolean;
}

export interface MessageProps {
  message: Message;
  isUser?: boolean;
  className?: string;
}

export interface ChatInputProps {
  onSendMessage: (message: string) => Promise<void>;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export interface FooterProps {
  isTyping?: boolean;
  className?: string;
}

export interface UseTalkBoxChatReturn {
  conversation: Conversation | null;
  messages: Message[];
  isLoading: boolean;
  isTyping: boolean;
  error: string | null;
  sendMessage: (content: string) => Promise<void>;
  clearConversation: () => void;
  retry: () => Promise<void>;
}
