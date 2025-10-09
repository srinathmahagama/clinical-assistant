import { ChatSession, Message } from '../types';
import { chatService } from './api';

class ChatSessionManager {
  private sessions: ChatSession[] = [];
  private currentSessionId: string | null = null;
  private readonly STORAGE_KEY = 'caremate_chat_sessions';
  private isInitialized = false;

  constructor() {
    // Initialize will be called when needed
  }

  // Initialize the session manager
  async initialize(): Promise<void> {
    if (this.isInitialized) return;
    
    try {
      await this.loadSessionsFromBackend();
      this.isInitialized = true;
    } catch (error) {
      console.error('Failed to initialize chat session manager:', error);
      // Fallback to localStorage if backend fails
      this.loadSessionsFromLocalStorage();
      this.isInitialized = true;
    }
  }

  // Load sessions from backend API
  private async loadSessionsFromBackend(): Promise<void> {
    try {
      const response = await chatService.getChatSessions();
      if (response.success && response.data) {
        this.sessions = response.data;
        // Set the most recent session as active if none is active
        if (this.sessions.length > 0 && !this.sessions.find(s => s.isActive)) {
          this.sessions[0].isActive = true;
          this.currentSessionId = this.sessions[0].id;
        }
      } else {
        console.error('Failed to load sessions from backend:', response.message);
        this.sessions = [];
      }
    } catch (error) {
      console.error('Error loading sessions from backend:', error);
      throw error;
    }
  }

  // Load sessions from localStorage (fallback)
  private loadSessionsFromLocalStorage(): void {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        this.sessions = JSON.parse(stored);
        // Set the most recent session as active if none is active
        if (this.sessions.length > 0 && !this.sessions.find(s => s.isActive)) {
          this.sessions[0].isActive = true;
          this.currentSessionId = this.sessions[0].id;
        }
      }
    } catch (error) {
      console.error('Failed to load chat sessions from localStorage:', error);
      this.sessions = [];
    }
  }

  // Save sessions to localStorage
  private saveSessions(): void {
    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.sessions));
    } catch (error) {
      console.error('Failed to save chat sessions:', error);
    }
  }

  // Generate a title from the first user message
  private generateTitle(firstMessage: string): string {
    const words = firstMessage.trim().split(' ');
    if (words.length <= 4) {
      return firstMessage.trim();
    }
    return words.slice(0, 4).join(' ') + '...';
  }

  // Create a new chat session
  async createNewSession(language: string = 'en'): Promise<ChatSession> {
    try {
      // Call backend API to create new session
      const response = await chatService.createChatSession(
        language === 'noongar' ? 'Mooditj Koora' : 'New Chat',
        language
      );
      
      if (response.success && response.data) {
        const newSession = response.data;
        
        // Deactivate all other sessions
        this.sessions.forEach(session => {
          session.isActive = false;
        });

        // Add new session to the beginning
        this.sessions.unshift(newSession);
        this.currentSessionId = newSession.id;
        this.saveSessions();

        return newSession;
      } else {
        throw new Error(response.message || 'Failed to create session');
      }
    } catch (error) {
      console.error('Failed to create new session via API, falling back to local creation:', error);
      
      // Fallback to local session creation
      const initialMessages = {
        en: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
        noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
      };

      const newSession: ChatSession = {
        id: Date.now().toString(),
        userId: '1', // TODO: Get from current user context
        title: language === 'noongar' ? 'Mooditj Koora' : 'New Chat',
        messages: [
          {
            id: '1',
            text: initialMessages[language as keyof typeof initialMessages] || initialMessages.en,
            isUser: false,
            timestamp: new Date().toLocaleTimeString('en-US', { 
              hour: 'numeric', 
              minute: '2-digit',
              hour12: true 
            }),
            createdAt: new Date().toISOString()
          }
        ],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        isActive: true,
        language: language
      };

      // Deactivate all other sessions
      this.sessions.forEach(session => {
        session.isActive = false;
      });

      // Add new session to the beginning
      this.sessions.unshift(newSession);
      this.currentSessionId = newSession.id;
      this.saveSessions();

      return newSession;
    }
  }

  // Get all sessions
  getAllSessions(): ChatSession[] {
    return [...this.sessions];
  }

  // Get current active session
  getCurrentSession(): ChatSession | null {
    return this.sessions.find(session => session.isActive) || null;
  }

  // Switch to a different session
  switchToSession(sessionId: string): ChatSession | null {
    // Deactivate all sessions
    this.sessions.forEach(session => {
      session.isActive = false;
    });

    // Activate the selected session
    const session = this.sessions.find(s => s.id === sessionId);
    if (session) {
      session.isActive = true;
      this.currentSessionId = sessionId;
      this.saveSessions();
      return session;
    }

    return null;
  }

  // Add a message to the current session
  addMessage(message: Message): void {
    const currentSession = this.getCurrentSession();
    if (currentSession) {
      currentSession.messages.push(message);
      currentSession.updatedAt = new Date().toISOString();

      // Update title if this is the first user message
      if (message.isUser && currentSession.title === 'New Chat') {
        currentSession.title = this.generateTitle(message.text);
      }

      this.saveSessions();
    }
  }

  // Delete a session
  deleteSession(sessionId: string): boolean {
    const index = this.sessions.findIndex(s => s.id === sessionId);
    if (index !== -1) {
      const wasActive = this.sessions[index].isActive;
      this.sessions.splice(index, 1);

      // If we deleted the active session, activate the first remaining session
      if (wasActive && this.sessions.length > 0) {
        this.sessions[0].isActive = true;
        this.currentSessionId = this.sessions[0].id;
      } else if (this.sessions.length === 0) {
        this.currentSessionId = null;
      }

      this.saveSessions();
      return true;
    }
    return false;
  }

  // Rename a session
  renameSession(sessionId: string, newTitle: string): boolean {
    const session = this.sessions.find(s => s.id === sessionId);
    if (session) {
      session.title = newTitle;
      session.updatedAt = new Date().toISOString();
      this.saveSessions();
      return true;
    }
    return false;
  }

  // Clear all sessions
  clearAllSessions(): void {
    this.sessions = [];
    this.currentSessionId = null;
    this.saveSessions();
  }

  // Get session by ID
  getSessionById(sessionId: string): ChatSession | null {
    return this.sessions.find(s => s.id === sessionId) || null;
  }

  // Create sample sessions for demonstration
  // Note: Sample sessions are now created in the backend and loaded via API
  // This method is no longer needed as we fetch data from the backend
  private createSampleSessions(): void {
    const sampleSessions: ChatSession[] = [
      {
        id: 'sample-1',
        userId: '1',
        title: 'Headache and fever symptoms',
        messages: [
          {
            id: '1',
            text: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
            isUser: false,
            timestamp: '9:30 AM',
            createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
          },
          {
            id: '2',
            text: 'I have been experiencing headaches and fever for the past 2 days. What should I do?',
            isUser: true,
            timestamp: '9:31 AM',
            createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000 + 60000).toISOString()
          },
          {
            id: '3',
            text: 'I understand your concern about the headaches and fever. These symptoms can have various causes. I recommend monitoring your temperature regularly and staying hydrated. If your fever persists above 101°F (38.3°C) or if you experience severe headaches, please consult with a healthcare provider.',
            isUser: false,
            timestamp: '9:32 AM',
            createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000 + 120000).toISOString()
          }
        ],
        createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000 + 120000).toISOString(),
        isActive: false,
        language: 'en'
      },
      {
        id: 'sample-2',
        userId: '1',
        title: 'Chest pain concerns',
        messages: [
          {
            id: '1',
            text: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
            isUser: false,
            timestamp: '2:15 PM',
            createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString()
          },
          {
            id: '2',
            text: 'I have been having chest pain on and off for a week. Should I be worried?',
            isUser: true,
            timestamp: '2:16 PM',
            createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000 + 60000).toISOString()
          },
          {
            id: '3',
            text: 'Chest pain is a symptom that should not be ignored. While it can have various causes, it\'s important to rule out serious conditions. I strongly recommend seeking immediate medical attention, especially if the pain is severe, radiates to your arm or jaw, or is accompanied by shortness of breath, nausea, or sweating.',
            isUser: false,
            timestamp: '2:17 PM',
            createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000 + 120000).toISOString()
          }
        ],
        createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000 + 120000).toISOString(),
        isActive: false,
        language: 'en'
      },
      {
        id: 'sample-3',
        userId: '1',
        title: 'New Chat',
        messages: [
          {
            id: '1',
            text: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
            isUser: false,
            timestamp: new Date().toLocaleTimeString('en-US', { 
              hour: 'numeric', 
              minute: '2-digit',
              hour12: true 
            }),
            createdAt: new Date().toISOString()
          }
        ],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        isActive: true,
        language: 'en'
      }
    ];

    this.sessions = sampleSessions;
    this.currentSessionId = 'sample-3';
    this.saveSessions();
  }
}

// Export singleton instance
export const chatSessionManager = new ChatSessionManager();
export default chatSessionManager;
