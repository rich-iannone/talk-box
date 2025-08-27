import { useState, useCallback, useRef } from 'react';
import type { UseTalkBoxChatReturn, Message, Conversation, ChatBotConfig } from '../types';
import { TalkBoxAPI, createUserMessage } from '../utils/api';

export const useTalkBoxChat = (
  config: ChatBotConfig,
  apiEndpoint?: string
): UseTalkBoxChatReturn => {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apiRef = useRef(new TalkBoxAPI(apiEndpoint));
  const retryMessageRef = useRef<string | null>(null);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    setError(null);
    setIsLoading(true);
    retryMessageRef.current = content;

    try {
      // Create user message immediately for optimistic UI
      const userMessage = createUserMessage(content);
      setMessages((prev: Message[]) => [...prev, userMessage]);

      let currentConversation = conversation;

      // Create conversation if it doesn't exist
      if (!currentConversation) {
        currentConversation = await apiRef.current.createConversation(config);
        setConversation(currentConversation);
      }

      // Show typing indicator
      setIsTyping(true);

      // Send message to API
      const assistantMessage = await apiRef.current.sendMessage(
        currentConversation.id,
        content
      );

      // Add assistant response
      setMessages((prev: Message[]) => [...prev, assistantMessage]);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');

      // Remove the optimistic user message on error
      setMessages((prev: Message[]) => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  }, [conversation, config]);

  const clearConversation = useCallback(() => {
    setConversation(null);
    setMessages([]);
    setError(null);
    setIsLoading(false);
    setIsTyping(false);
    retryMessageRef.current = null;
  }, []);

  const retry = useCallback(async () => {
    if (retryMessageRef.current) {
      await sendMessage(retryMessageRef.current);
    }
  }, [sendMessage]);

  return {
    conversation,
    messages,
    isLoading,
    isTyping,
    error,
    sendMessage,
    clearConversation,
    retry,
  };
};
