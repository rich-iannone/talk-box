import type { Message, Conversation, ChatBotConfig } from '../types';

const DEFAULT_API_ENDPOINT = 'http://localhost:8000';

export class TalkBoxAPI {
  private apiEndpoint: string;

  constructor(apiEndpoint: string = DEFAULT_API_ENDPOINT) {
    this.apiEndpoint = apiEndpoint.replace(/\/$/, ''); // Remove trailing slash
  }

  async createConversation(config: ChatBotConfig): Promise<Conversation> {
    const response = await fetch(`${this.apiEndpoint}/conversation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ config }),
    });

    if (!response.ok) {
      throw new Error(`Failed to create conversation: ${response.statusText}`);
    }

    return response.json();
  }

  async sendMessage(conversationId: string, content: string): Promise<Message> {
    const response = await fetch(`${this.apiEndpoint}/conversation/${conversationId}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content }),
    });

    if (!response.ok) {
      throw new Error(`Failed to send message: ${response.statusText}`);
    }

    return response.json();
  }

  async getConversation(conversationId: string): Promise<Conversation> {
    const response = await fetch(`${this.apiEndpoint}/conversation/${conversationId}`);

    if (!response.ok) {
      throw new Error(`Failed to get conversation: ${response.statusText}`);
    }

    return response.json();
  }

  async deleteConversation(conversationId: string): Promise<void> {
    const response = await fetch(`${this.apiEndpoint}/conversation/${conversationId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      throw new Error(`Failed to delete conversation: ${response.statusText}`);
    }
  }

  // Health check endpoint
  async health(): Promise<{ status: string; timestamp: string }> {
    const response = await fetch(`${this.apiEndpoint}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }
}

// Utility function to generate unique IDs
export const generateId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
};

// Utility function to create a user message
export const createUserMessage = (content: string): Message => ({
  id: generateId(),
  content,
  role: 'user',
  timestamp: new Date(),
});

// Utility function to format timestamps
export const formatTimestamp = (timestamp: Date | string | number): string => {
  try {
    const date = timestamp instanceof Date ? timestamp : new Date(timestamp);

    // Check if the date is valid
    if (isNaN(date.getTime())) {
      return 'Just now';
    }

    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    }).format(date);
  } catch (error) {
    // If any error occurs, return a fallback
    return 'Just now';
  }
};
