import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Send, Bot, Menu, Image, Mic } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import FileMessage from '../components/FileMessage/FileMessage';
import ChatSidebar from '../components/ChatSidebar/ChatSidebar';
import Avatar from '../components/Avatar/Avatar';
import VoiceMessage from '../components/VoiceMessage/VoiceMessage';
import VoiceRecorder from '../components/VoiceRecorder/VoiceRecorder';
// import TextToSpeech from '../components/TextToSpeech/TextToSpeech';
import SymptomSelector from '../components/SymptomSelector/SymptomSelector';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';
import { chatService } from '../services/api';
import { Message, ChatSession } from '../types';
import chatSessionManager from '../services/chatSessionManager';

interface AssistantPageProps {
  onNavigate: (page: string) => void;
  onLogout: () => void;
  user?: {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
  };
  isGuest?: boolean;
  onSignIn?: () => void;
}

const AssistantPage: React.FC<AssistantPageProps> = ({ onLogout, user, isGuest, onSignIn }) => {
  const [searchParams] = useSearchParams();
  const { language, t, setChatSessionLanguage } = useLanguage();
  const { theme } = useTheme();
  
  // Chat session management
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [showSidebar, setShowSidebar] = useState(false); // Always hide sidebar by default
  
  // Message state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSymptomSelector, setShowSymptomSelector] = useState(false);
  const [showVoiceRecorder, setShowVoiceRecorder] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    if (currentSession) {
      console.log('🔄 Setting messages from current session:', currentSession.messages.length);
      setMessages(currentSession.messages);
    }
  }, [currentSession]);

  // Initialize chat sessions on component mount
  useEffect(() => {
    const initializeSessions = async () => {
      const sessionId = searchParams.get('sessionId');
      
      if (isGuest) {
        // For guest users, create a temporary session that won't be saved
        const initialMessages = {
          en: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
          noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
        };

        const tempSession: ChatSession = {
          id: 'guest-session',
          userId: 'guest',
          title: language === 'noongar' ? 'Mooditj Koora' : 'Guest Chat',
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
          language: language
        };
        setCurrentSession(tempSession);
        setMessages(tempSession.messages);
        setSessions([]);
      } else {
        // For logged-in users, load sessions from backend
        try {
          // Load all sessions from backend
          const sessionsResponse = await chatService.getChatSessions();
          if (sessionsResponse.success && sessionsResponse.data) {
            setSessions(sessionsResponse.data);
            
            // Check if we need to load a specific session from URL
            if (sessionId) {
              const specificSession = sessionsResponse.data.find(s => s.id === sessionId);
              if (specificSession) {
                // Load messages for the specific session
                const messagesResponse = await chatService.getChatMessages(sessionId);
                if (messagesResponse.success && messagesResponse.data) {
                  const sessionWithMessages = { ...specificSession, messages: messagesResponse.data };
                  setCurrentSession(sessionWithMessages);
                  setMessages(messagesResponse.data);
                  if (specificSession.language) {
                    setChatSessionLanguage(specificSession.id, specificSession.language as 'en' | 'noongar');
                  }
                } else {
                  setCurrentSession(specificSession);
                  setMessages(specificSession.messages);
                }
              } else {
                // Session not found, create a new one
                const newSessionResponse = await chatService.createChatSession('New Chat', language);
                if (newSessionResponse.success && newSessionResponse.data) {
                  setCurrentSession(newSessionResponse.data);
                  setMessages(newSessionResponse.data.messages);
                  // Refresh sessions list
                  const updatedSessionsResponse = await chatService.getChatSessions();
                  if (updatedSessionsResponse.success && updatedSessionsResponse.data) {
                    setSessions(updatedSessionsResponse.data);
                  }
                }
              }
            } else {
              // No specific session requested, use the most recent session or create new one
              if (sessionsResponse.data.length > 0) {
                const mostRecentSession = sessionsResponse.data[0]; // Sessions are sorted by updatedAt desc
                // Load messages for the most recent session
                const messagesResponse = await chatService.getChatMessages(mostRecentSession.id);
                if (messagesResponse.success && messagesResponse.data) {
                  const sessionWithMessages = { ...mostRecentSession, messages: messagesResponse.data };
                  setCurrentSession(sessionWithMessages);
                  setMessages(messagesResponse.data);
                  if (mostRecentSession.language) {
                    setChatSessionLanguage(mostRecentSession.id, mostRecentSession.language as 'en' | 'noongar');
                  }
                } else {
                  setCurrentSession(mostRecentSession);
                  setMessages(mostRecentSession.messages);
                }
              } else {
                // No sessions exist, create a new one
                const newSessionResponse = await chatService.createChatSession('New Chat', language);
                if (newSessionResponse.success && newSessionResponse.data) {
                  setCurrentSession(newSessionResponse.data);
                  setMessages(newSessionResponse.data.messages);
                  setSessions([newSessionResponse.data]);
                }
              }
            }
          } else {
            // Failed to load sessions, create a new one
            const newSessionResponse = await chatService.createChatSession('New Chat', language);
            if (newSessionResponse.success && newSessionResponse.data) {
              setCurrentSession(newSessionResponse.data);
              setMessages(newSessionResponse.data.messages);
              setSessions([newSessionResponse.data]);
            }
          }
        } catch (error) {
          console.error('Failed to initialize chat sessions:', error);
          // Fallback to local session manager
          try {
            await chatSessionManager.initialize();
            const allSessions = chatSessionManager.getAllSessions();
            setSessions(allSessions);
            
            const activeSession = chatSessionManager.getCurrentSession();
            if (activeSession) {
              setCurrentSession(activeSession);
              setMessages(activeSession.messages);
            } else {
              const newSession = await chatSessionManager.createNewSession(language);
              setCurrentSession(newSession);
              setMessages(newSession.messages);
            }
          } catch (fallbackError) {
            console.error('Fallback initialization also failed:', fallbackError);
            // Create a minimal session as last resort
            const initialMessages = {
              en: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
              noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
            };

            const fallbackSession: ChatSession = {
              id: `fallback-session-${Date.now()}`,
              userId: user?.id || '1',
              title: 'New Chat',
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
              language: language
            };
            setCurrentSession(fallbackSession);
            setMessages(fallbackSession.messages);
            setSessions([]);
          }
        }
      }
    };

    initializeSessions();
  }, [isGuest, language, setChatSessionLanguage, searchParams, user]);

  // Update messages when current session changes
  useEffect(() => {
    if (currentSession) {
      setMessages(currentSession.messages);
    }
  }, [currentSession]);

  // Chat session management functions
  const handleNewChat = async () => {
    if (isGuest) {
      // For guest users, create a local session
      const initialMessages = {
        en: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
        noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
      };

      const tempSession: ChatSession = {
        id: `guest-session-${Date.now()}`,
        userId: 'guest',
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
        language: language
      };
      setCurrentSession(tempSession);
      setMessages(tempSession.messages);
      setSessions([]);
      return;
    }

    try {
      const response = await chatService.createChatSession('New Chat', language);
      if (response.success && response.data) {
        setCurrentSession(response.data);
        setMessages(response.data.messages);
        // Refresh sessions list
        const sessionsResponse = await chatService.getChatSessions();
        if (sessionsResponse.success && sessionsResponse.data) {
          setSessions(sessionsResponse.data);
        }
      }
    } catch (error) {
      console.error('Failed to create new chat session:', error);
      // Fallback to local session manager
      const newSession = await chatSessionManager.createNewSession(language);
      setCurrentSession(newSession);
      setMessages(newSession.messages);
      setSessions(chatSessionManager.getAllSessions());
    }
  };

  const handleSelectSession = async (sessionId: string) => {
    if (isGuest) {
      // For guest users, just switch locally
      const session = chatSessionManager.switchToSession(sessionId);
      if (session) {
        setCurrentSession(session);
        setMessages(session.messages);
        if (session.language) {
          setChatSessionLanguage(sessionId, session.language as 'en' | 'noongar');
        }
      }
      return;
    }

    try {
      // Get session messages from backend
      const response = await chatService.getChatMessages(sessionId);
      if (response.success && response.data) {
        // Find the session in our current sessions list
        const session = sessions.find(s => s.id === sessionId);
        if (session) {
          const updatedSession = { ...session, messages: response.data };
          setCurrentSession(updatedSession);
          setMessages(response.data);
          // Set the chat session language if it exists
          if (session.language) {
            setChatSessionLanguage(sessionId, session.language as 'en' | 'noongar');
          }
        }
      }
    } catch (error) {
      console.error('Failed to load session messages:', error);
      // Fallback to local session manager
      const session = chatSessionManager.switchToSession(sessionId);
      if (session) {
        setCurrentSession(session);
        setMessages(session.messages);
        if (session.language) {
          setChatSessionLanguage(sessionId, session.language as 'en' | 'noongar');
        }
      }
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (isGuest) {
      // For guest users, handle locally
      if (sessions.length <= 1) {
        // Don't delete the last session, just clear its messages
        const session = chatSessionManager.getSessionById(sessionId);
        if (session) {
          const initialMessages = {
            en: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
            noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
          };

          session.messages = [
            {
              id: '1',
              text: initialMessages[session.language as keyof typeof initialMessages] || initialMessages.en,
              isUser: false,
              timestamp: new Date().toLocaleTimeString('en-US', { 
                hour: 'numeric', 
                minute: '2-digit',
                hour12: true 
              }),
              createdAt: new Date().toISOString()
            }
          ];
          session.title = session.language === 'noongar' ? 'Mooditj Koora' : 'New Chat';
          session.updatedAt = new Date().toISOString();
          chatSessionManager['saveSessions']();
          setMessages(session.messages);
        }
      } else {
        chatSessionManager.deleteSession(sessionId);
        const updatedSessions = chatSessionManager.getAllSessions();
        setSessions(updatedSessions);
        
        const newActiveSession = chatSessionManager.getCurrentSession();
        if (newActiveSession) {
          setCurrentSession(newActiveSession);
          setMessages(newActiveSession.messages);
        }
      }
      return;
    }

    try {
      const response = await chatService.deleteChatSession(sessionId);
      if (response.success) {
        // Remove from local state
        const updatedSessions = sessions.filter(s => s.id !== sessionId);
        setSessions(updatedSessions);
        
        // If we deleted the current session, switch to another one
        if (currentSession?.id === sessionId) {
          if (updatedSessions.length > 0) {
            // Switch to the first available session
            await handleSelectSession(updatedSessions[0].id);
          } else {
            // Create a new session if no sessions left
            await handleNewChat();
          }
        }
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
      // Fallback to local session manager
      if (sessions.length <= 1) {
        const session = chatSessionManager.getSessionById(sessionId);
        if (session) {
          const initialMessages = {
            en: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
            noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
          };

          session.messages = [
            {
              id: '1',
              text: initialMessages[session.language as keyof typeof initialMessages] || initialMessages.en,
              isUser: false,
              timestamp: new Date().toLocaleTimeString('en-US', { 
                hour: 'numeric', 
                minute: '2-digit',
                hour12: true 
              }),
              createdAt: new Date().toISOString()
            }
          ];
          session.title = session.language === 'noongar' ? 'Mooditj Koora' : 'New Chat';
          session.updatedAt = new Date().toISOString();
          chatSessionManager['saveSessions']();
          setMessages(session.messages);
        }
      } else {
        chatSessionManager.deleteSession(sessionId);
        const updatedSessions = chatSessionManager.getAllSessions();
        setSessions(updatedSessions);
        
        const newActiveSession = chatSessionManager.getCurrentSession();
        if (newActiveSession) {
          setCurrentSession(newActiveSession);
          setMessages(newActiveSession.messages);
        }
      }
    }
  };

  const handleRenameSession = async (sessionId: string, newTitle: string) => {
    if (isGuest) {
      // For guest users, handle locally
      chatSessionManager.renameSession(sessionId, newTitle);
      setSessions(chatSessionManager.getAllSessions());
      return;
    }

    try {
      const response = await chatService.updateChatSession(sessionId, { title: newTitle });
      if (response.success && response.data) {
        // Update local state
        const updatedSessions = sessions.map(s => 
          s.id === sessionId ? { ...s, title: newTitle, updatedAt: response.data?.updatedAt || new Date().toISOString() } : s
        );
        setSessions(updatedSessions);
        
        // Update current session if it's the one being renamed
        if (currentSession?.id === sessionId && response.data) {
          setCurrentSession({ ...currentSession, title: newTitle, updatedAt: response.data.updatedAt });
        }
      }
    } catch (error) {
      console.error('Failed to rename session:', error);
      // Fallback to local session manager
      chatSessionManager.renameSession(sessionId, newTitle);
      setSessions(chatSessionManager.getAllSessions());
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const messageText = inputText;
    setInputText('');
    setIsLoading(true);

    // Create user message immediately for better UX
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      text: messageText,
      isUser: true,
      timestamp: new Date().toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
      }),
      createdAt: new Date().toISOString(),
      type: 'text'
    };

    try {
      // Add user message immediately to show in UI
      setMessages(prev => [...prev, userMessage]);

      // Send message to backend
      const response = await chatService.sendMessage(
        currentSession?.id || 'guest-session', 
        messageText
      );
      
      if (response.success && response.data) {
        const { userMessage: backendUserMessage, response: aiResponse } = response.data;
        
        // Replace the temporary user message with backend version and add AI response
        setMessages(prev => {
          const withoutLast = prev.slice(0, -1); // Remove temporary user message
          return [...withoutLast, backendUserMessage, aiResponse];
        });

        // Update current session for logged-in users
        if (!isGuest && currentSession) {
          const updatedMessages = [...currentSession.messages, backendUserMessage, aiResponse];
          const updatedSession = { 
            ...currentSession, 
            messages: updatedMessages, 
            updatedAt: new Date().toISOString() 
          };
          setCurrentSession(updatedSession);
        }
      } else {
        throw new Error('Failed to send message');
      }
    } catch (error) {
      console.error('Chat message failed:', error);
      
      // Fallback response
      const fallbackMessage: Message = {
        id: `ai-${Date.now()}`,
        text: "I'm here to help with your health concerns. Please try again.",
        isUser: false,
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour: 'numeric', 
          minute: '2-digit',
          hour12: true 
        }),
        createdAt: new Date().toISOString(),
        type: 'text'
      };

      // Update messages with fallback
      setMessages(prev => {
        const withoutLast = prev.slice(0, -1); // Remove temporary user message  
        return [...withoutLast, userMessage, fallbackMessage];
      });

      // Update session for logged-in users
      if (!isGuest && currentSession) {
        const updatedMessages = [...currentSession.messages, userMessage, fallbackMessage];
        const updatedSession = { 
          ...currentSession, 
          messages: updatedMessages, 
          updatedAt: new Date().toISOString() 
        };
        setCurrentSession(updatedSession);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteMessage = async (messageId: string) => {
    try {
      // Delete from backend if not guest
      if (!isGuest && currentSession) {
        const response = await chatService.deleteMessage(currentSession.id, messageId);
        if (!response.success) {
          console.error('Failed to delete message from backend:', response.message);
        }
      }
      
      // Update local state
      if (currentSession) {
        const updatedMessages = currentSession.messages.filter(msg => msg.id !== messageId);
        currentSession.messages = updatedMessages;
        currentSession.updatedAt = new Date().toISOString();
        
        if (!isGuest) {
          chatSessionManager['saveSessions']();
        }
        setMessages(updatedMessages);
      }
    } catch (error) {
      console.error('Error deleting message:', error);
      // Still update local state even if backend fails
      if (currentSession) {
        const updatedMessages = currentSession.messages.filter(msg => msg.id !== messageId);
        currentSession.messages = updatedMessages;
        setMessages(updatedMessages);
      }
    }
  };

  const handleSendVoiceMessage = async (audioBlob: Blob, duration: number) => {
    console.log('🎤 Sending voice message:', audioBlob, 'duration:', duration);
    setIsLoading(true);
    setShowVoiceRecorder(false);
    
    try {
      // Create audio URL for immediate playback
      const audioUrl = URL.createObjectURL(audioBlob);
      
      // Create user voice message immediately
      const userVoiceMessage: Message = {
        id: (Date.now()).toString(),
        text: 'Voice message...', // Placeholder, will be updated with transcript
        isUser: true,
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour: 'numeric', 
          minute: '2-digit',
          hour12: true 
        }),
        createdAt: new Date().toISOString(),
        type: 'voice',
        audioUrl: audioUrl,
        duration: duration,
        transcript: 'Voice message...'
      };
      
      // Add user message immediately
      if (!isGuest && currentSession) {
        const updatedMessages = [...currentSession.messages, userVoiceMessage];
        const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
        setCurrentSession(updatedSession);
        setMessages(updatedMessages);
        
        const updatedSessions = sessions.map(s => 
          s.id === currentSession.id ? updatedSession : s
        );
        setSessions(updatedSessions);
      } else {
        setMessages(prev => [...prev, userVoiceMessage]);
      }
      
      // Send to backend
      const response = await chatService.sendVoiceMessage(
        currentSession?.id || 'guest-session',
        audioBlob,
        duration
      );
      
      if (response.success && response.data) {
        const { userMessage: backendUserMessage, response: aiResponse } = response.data;
        
        // Update user message with backend data (transcript)
        const updatedUserMessage = {
          ...backendUserMessage,
          audioUrl: audioUrl // Keep local URL for immediate playback
        };
        
        // Replace user message and add AI response
        if (!isGuest && currentSession) {
          const currentMessages = currentSession.messages;
          const updatedMessages = [
            ...currentMessages.slice(0, -1), // Remove placeholder user message
            updatedUserMessage, // Add updated user message with transcript
            aiResponse // Add AI response
          ];
          
          const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
          setCurrentSession(updatedSession);
          setMessages(updatedMessages);
          
          const updatedSessions = sessions.map(s => 
            s.id === currentSession.id ? updatedSession : s
          );
          setSessions(updatedSessions);
        } else {
          setMessages(prev => {
            const withoutLast = prev.slice(0, -1); // Remove placeholder user message
            return [...withoutLast, updatedUserMessage, aiResponse];
          });
        }
        
        // Auto-play AI response if it's a voice message
        if (aiResponse && aiResponse.type === 'voice' && aiResponse.audioUrl) {
          setTimeout(() => {
            const audioUrl = aiResponse.audioUrl!.startsWith('http') ? aiResponse.audioUrl! : `http://localhost:8000${aiResponse.audioUrl!}`;
            const audio = new Audio(audioUrl);
            audio.play().catch(error => {
              console.log('Auto-play failed (user interaction required):', error);
            });
          }, 500);
        }
      } else {
        // Fallback response
        const fallbackMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: "I'm here to help with your health concerns. Please try again.",
          isUser: false,
          timestamp: new Date().toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
          }),
          createdAt: new Date().toISOString(),
          type: 'text'
        };

        if (!isGuest && currentSession) {
          const updatedMessages = [...currentSession.messages, fallbackMessage];
          const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
          setCurrentSession(updatedSession);
          setMessages(updatedMessages);
          
          const updatedSessions = sessions.map(s => 
            s.id === currentSession.id ? updatedSession : s
          );
          setSessions(updatedSessions);
        } else {
          setMessages(prev => [...prev, fallbackMessage]);
        }
      }
    } catch (error) {
      console.error('Voice message failed:', error);
      // Fallback response
      const fallbackMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I'm here to help with your health concerns. Please try again.",
        isUser: false,
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour: 'numeric', 
          minute: '2-digit',
          hour12: true 
        }),
        createdAt: new Date().toISOString(),
        type: 'text'
      };

      if (!isGuest && currentSession) {
        const updatedMessages = [...currentSession.messages, fallbackMessage];
        const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
        setCurrentSession(updatedSession);
        setMessages(updatedMessages);
        
        const updatedSessions = sessions.map(s => 
          s.id === currentSession.id ? updatedSession : s
        );
        setSessions(updatedSessions);
      } else {
        setMessages(prev => [...prev, fallbackMessage]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const sendMessageToAssistant = async (_messageText: string, symptoms: { name: string; tags: string[] }[]) => {
    setIsLoading(true);
    
    try {
      // Extract symptom names for backend processing
      const symptomNames = symptoms.map(symptom => symptom.name);
      
      // Send symptoms directly to chat API (no user message needed)
      const response = await chatService.sendSymptoms(currentSession?.id || 'guest-session', symptomNames);
      
      if (response.success && response.data) {
        // Add AI response to messages
        if (!isGuest) {
          // For logged-in users, update the current session with the AI response
          if (currentSession) {
            // First add messages to local state for immediate UI update
            const updatedMessages = [...currentSession.messages, response.data];
            const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
            setCurrentSession(updatedSession);
            setMessages(updatedMessages);
            
            // Update sessions list to reflect the updated session
            const updatedSessions = sessions.map(s => 
              s.id === currentSession.id ? updatedSession : s
            );
            setSessions(updatedSessions);
            
            // Refresh session data from backend to ensure consistency
            try {
              const messagesResponse = await chatService.getChatMessages(currentSession.id);
              if (messagesResponse.success && messagesResponse.data) {
                const sessionWithMessages = { ...currentSession, messages: messagesResponse.data };
                setCurrentSession(sessionWithMessages);
                setMessages(messagesResponse.data);
              }
            } catch (error) {
              console.error('Failed to refresh session data:', error);
            }
          }
        } else {
          setMessages(prev => [...prev, response.data!]);
        }
      } else {
        // Fallback response if backend fails
        const fallbackMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: "Thank you for sharing your symptoms. I understand you're experiencing: " + symptomNames.join(', ') + ". Based on this information, I recommend monitoring your symptoms and consulting with a healthcare provider if they persist or worsen.",
          isUser: false,
          timestamp: new Date().toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
          }),
          createdAt: new Date().toISOString(),
          type: 'text'
        };

        if (!isGuest) {
          // For logged-in users, update the current session with the fallback message
          if (currentSession) {
            const updatedMessages = [...currentSession.messages, fallbackMessage];
            const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
            setCurrentSession(updatedSession);
            setMessages(updatedMessages);
            
            // Update sessions list to reflect the updated session
            const updatedSessions = sessions.map(s => 
              s.id === currentSession.id ? updatedSession : s
            );
            setSessions(updatedSessions);
          }
        } else {
          setMessages(prev => [...prev, fallbackMessage]);
        }
      }
    } catch (error) {
      console.error('Failed to send symptoms to assistant:', error);
      
      // Error fallback response
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I've received your symptom information. While I'm having trouble processing it right now, I recommend keeping track of your symptoms and consulting with a healthcare provider for proper evaluation.",
        isUser: false,
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour: 'numeric', 
          minute: '2-digit',
          hour12: true 
        }),
        createdAt: new Date().toISOString(),
        type: 'text'
      };

      if (!isGuest) {
        // For logged-in users, update the current session with the error message
        if (currentSession) {
          const updatedMessages = [...currentSession.messages, errorMessage];
          const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
          setCurrentSession(updatedSession);
          setMessages(updatedMessages);
          
          // Update sessions list to reflect the updated session
          const updatedSessions = sessions.map(s => 
            s.id === currentSession.id ? updatedSession : s
          );
          setSessions(updatedSessions);
        }
      } else {
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSymptomSelection = (symptoms: { name: string; tags: string[] }[]) => {
    if (symptoms.length === 0) return;
    
    const symptomText = symptoms.map(symptom => 
      `${symptom.name} (${symptom.tags.join(', ')})`
    ).join(', ');
    
    const messageText = `I'm experiencing these symptoms: ${symptomText}`;
    
    // Create user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      isUser: true,
      timestamp: new Date().toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
      }),
      createdAt: new Date().toISOString(),
      type: 'text'
    };

    // Add user message to current session and update messages
    if (!isGuest) {
      chatSessionManager.addMessage(userMessage);
      const updatedMessages = chatSessionManager.getCurrentSession()?.messages || [];
      setMessages(updatedMessages);
    } else {
      setMessages(prev => [...prev, userMessage]);
    }

    // Close popup after 0.3s delay
    setTimeout(() => {
      setShowSymptomSelector(false);
    }, 300);

    // Send to assistant
    sendMessageToAssistant(messageText, symptoms);
  };

  return (
    <Layout showLanguageButton={false}>
      <Header onLogout={onLogout} showLanguage={false} user={user} isGuest={isGuest} onSignIn={onSignIn} />
      <div className="h-screen p-4 pt-20 overflow-hidden">
        <div className="max-w-7xl mx-auto h-full">
          <div className={`transition-all duration-300 ${showSymptomSelector ? 'blur-sm' : ''}`}>
            <BackButton to="/dashboard" />
          </div>
          
          <div className={`flex h-[calc(110vh-180px)] rounded-2xl shadow-2xl overflow-hidden transition-all duration-300 my-6 mx-2 ${
            theme === 'dark' ? 'bg-slate-800' : 'bg-white'
          } ${showSymptomSelector ? 'blur-sm' : ''}`}>
            {/* Sidebar */}
            {showSidebar && (
              <ChatSidebar
                sessions={sessions}
                currentSessionId={currentSession?.id || null}
                onNewChat={handleNewChat}
                onSelectSession={handleSelectSession}
                onDeleteSession={handleDeleteSession}
                onRenameSession={handleRenameSession}
              />
            )}
            
                      {/* Main Chat Area */}
                      <div className="flex-1 flex flex-col">
              {/* Chat Header */}
              <div className={`p-4 flex items-center justify-between border-b transition-colors duration-300 ${
                theme === 'dark' 
                  ? 'bg-gradient-to-r from-slate-700 to-slate-600 border-slate-600' 
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 border-gray-200'
              }`}>
                <div className="flex items-center space-x-3">
                  {!isGuest && (
                    <button
                      onClick={() => setShowSidebar(!showSidebar)}
                      className={`p-2 rounded-lg transition-colors ${
                        theme === 'dark' 
                          ? 'text-white hover:bg-slate-600' 
                          : 'text-white hover:bg-white/20'
                      }`}
                      title="Toggle chat history"
                    >
                      <Menu className="w-5 h-5" />
                    </button>
                  )}
                  <div>
                    <h2 className={`font-semibold text-lg ${
                      theme === 'dark' ? 'text-white' : 'text-white'
                    }`}>
                      {currentSession?.title || t('healthAssistant')}
                    </h2>
                    <p className={`text-sm ${
                      theme === 'dark' ? 'text-slate-300' : 'text-white/80'
                    }`}>
                      {t('available247')} • {t('voiceTextSupport')}
                      {isGuest && (
                        <span className={`ml-2 ${
                          theme === 'dark' ? 'text-yellow-400' : 'text-yellow-300'
                        }`}>{t('guestUser')}</span>
                      )}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {isGuest && (
                    <button
                      onClick={onSignIn}
                      className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                        theme === 'dark'
                          ? 'bg-yellow-500/20 text-yellow-300 hover:bg-yellow-500/30'
                          : 'bg-yellow-500/20 text-yellow-200 hover:bg-yellow-500/30'
                      }`}
                    >
                      {t('signInToSave')}
                    </button>
                  )}
                  <Bot className={`w-6 h-6 ${theme === 'dark' ? 'text-white' : 'text-white'}`} />
                </div>
              </div>

              {/* Messages Area */}
              <div className={`flex-1 overflow-y-auto p-6 space-y-4 transition-colors duration-300 ${
                theme === 'dark' ? 'bg-slate-900' : 'bg-gray-50'
              }`}>
                {messages && messages.length > 0 ? (
                  messages.map((message, index) => {
                    console.log(`🎨 RENDERING MESSAGE ${index}:`, message.id, message.text.substring(0, 50));
                    return (
                      <div key={message.id} className={`flex items-start space-x-3 ${message.isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
                        {/* Avatar */}
                        <Avatar 
                          type={message.isUser ? 'user' : 'assistant'}
                          userName={message.isUser ? user?.firstName : undefined}
                          isGuest={message.isUser ? isGuest : false}
                          size="md"
                        />
                        
                        {/* Message Content */}
                        <div className={`flex flex-col ${message.isUser ? 'items-end' : 'items-start'} max-w-[70%]`}>
                          {/* User/Assistant Name */}
                          <div className={`text-xs mb-1 px-2 ${
                            theme === 'dark' ? 'text-slate-400' : 'text-gray-500'
                          }`}>
                            {message.isUser 
                              ? (isGuest ? 'Guest User' : user?.firstName || 'User')
                              : 'CareMate Assistant'
                            }
                          </div>
                          
                          {/* Message Bubble */}
                          <div className={`max-w-md px-4 py-3 rounded-lg relative group transition-colors duration-300 ${
                            message.isUser 
                              ? theme === 'dark' 
                                ? 'bg-blue-900/45 text-white' 
                                : 'bg-blue-800/45 text-white'
                              : theme === 'dark'
                                ? 'bg-slate-700 text-slate-100'
                                : 'bg-gray-200 text-gray-800'
                          }`}>
                            <p className="text-sm break-words whitespace-pre-wrap">{message.text}</p>
                            <div className="flex items-center justify-between mt-1">
                              <div className={`text-xs ${
                                message.isUser 
                                  ? 'text-blue-100' 
                                  : theme === 'dark' 
                                    ? 'text-slate-400' 
                                    : 'text-gray-500'
                              }`}>{message.timestamp}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    No messages yet. Start a conversation!
                  </div>
                )}
                
                {isLoading && (
                  <div className="flex items-start space-x-3">
                    <Avatar type="assistant" size="md" />
                    <div className="flex flex-col items-start max-w-[70%]">
                      <div className={`text-xs mb-1 px-2 ${
                        theme === 'dark' ? 'text-slate-400' : 'text-gray-500'
                      }`}>
                        CareMate Assistant
                      </div>
                      <div className={`max-w-md px-4 py-3 rounded-lg transition-colors duration-300 ${
                        theme === 'dark' 
                          ? 'bg-slate-700 text-slate-100' 
                          : 'bg-gray-200 text-gray-800'
                      }`}>
                        <div className="flex items-center space-x-2">
                          <div className={`animate-spin rounded-full h-4 w-4 border-b-2 ${
                            theme === 'dark' ? 'border-slate-300' : 'border-gray-600'
                          }`}></div>
                          <span className="text-sm">Assistant is typing...</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Auto-scroll anchor */}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className={`p-4 border-t transition-colors duration-300 ${
                theme === 'dark' 
                  ? 'bg-slate-800 border-slate-600' 
                  : 'bg-white border-gray-200'
              }`}>
                <div className="flex items-center space-x-3">
                  {/* Text Input */}
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={t('typeYourHealthQuestion')}
                      className={`w-full px-4 py-3 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors duration-300 ${
                        theme === 'dark'
                          ? 'bg-slate-700 border-slate-600 text-white placeholder-slate-400'
                          : 'bg-white border border-gray-300 text-gray-900 placeholder-gray-500'
                      }`}
                      disabled={isLoading}
                    />
                  </div>

                  {/* Voice Message Button */}
                  {!showVoiceRecorder ? (
                    <button
                      onClick={() => setShowVoiceRecorder(true)}
                      className={`p-3 rounded-full text-white transition-all transform hover:scale-105 shadow-lg ${
                        theme === 'dark' 
                          ? 'bg-green-600 hover:bg-green-700' 
                          : 'bg-green-500 hover:bg-green-600'
                      }`}
                      title="Record voice message"
                    >
                      <Mic className="w-5 h-5" />
                    </button>
                  ) : (
                    <VoiceRecorder
                      onSendVoiceMessage={handleSendVoiceMessage}
                      onCancel={() => setShowVoiceRecorder(false)}
                    />
                  )}

                  {/* Symptom Selector Button */}
                  <button
                    onClick={() => setShowSymptomSelector(true)}
                    className={`p-3 rounded-full text-white transition-all transform hover:scale-105 shadow-lg ${
                      theme === 'dark' 
                        ? 'bg-blue-600 hover:bg-blue-500' 
                        : 'bg-blue-500 hover:bg-blue-600'
                    }`}
                    title="Select symptoms"
                  >
                    <Image className="w-5 h-5" />
                  </button>

                  {/* Send Button */}
                  <button
                    onClick={sendMessage}
                    disabled={!inputText.trim() || isLoading}
                    className="p-3 rounded-full bg-blue-500 text-white hover:bg-blue-600 transition-all transform hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                    title={t('sendMessage')}
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Symptom Selector Modal */}
      <SymptomSelector
        isOpen={showSymptomSelector}
        onClose={() => {
          console.log('Closing symptom selector...');
          setShowSymptomSelector(false);
        }}
        onSelectSymptoms={handleSymptomSelection}
        isLoading={isLoading}
      />
    </Layout>
  );
};

export default AssistantPage;