import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Send, Bot, Menu, Image, Mic, Volume2, Play, Heart, Users, Shield } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import FileMessage from '../components/FileMessage/FileMessage';
import ChatSidebar from '../components/ChatSidebar/ChatSidebar';
import Avatar from '../components/Avatar/Avatar';
import VoiceMessage from '../components/VoiceMessage/VoiceMessage';
import VoiceRecorder from '../components/VoiceRecorder/VoiceRecorder';
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
  const [showSidebar, setShowSidebar] = useState(false);
  
  // Message state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSymptomSelector, setShowSymptomSelector] = useState(false);
  const [showVoiceRecorder, setShowVoiceRecorder] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [culturalWelcome, setCulturalWelcome] = useState(true);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Cultural elements
  const culturalElements = [
    {
      icon: "🌿",
      title: "Boodja Ngangk",
      description: "Country Healing",
      noongarDesc: "Ngangk boodja boola djerap"
    },
    {
      icon: "👨‍👩‍👧‍👦",
      title: "Moort Kwop",
      description: "Family Wellbeing",
      noongarDesc: "Moort kwop koorliny"
    },
    {
      icon: "🦘",
      title: "Koorliny Boodja",
      description: "Walking Country",
      noongarDesc: "Koorliny boodja ngaangk"
    }
  ];

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
          en: "Kaya! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
          noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
        };

        const tempSession: ChatSession = {
          id: 'guest-session',
          userId: 'guest',
          title: language === 'noongar' ? 'Mooditj Koora' : 'Healing Journey',
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
          const sessionsResponse = await chatService.getChatSessions();
          if (sessionsResponse.success && sessionsResponse.data) {
            setSessions(sessionsResponse.data);
            
            if (sessionId) {
              const specificSession = sessionsResponse.data.find(s => s.id === sessionId);
              if (specificSession) {
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
                const newSessionResponse = await chatService.createChatSession('Healing Journey', language);
                if (newSessionResponse.success && newSessionResponse.data) {
                  setCurrentSession(newSessionResponse.data);
                  setMessages(newSessionResponse.data.messages);
                  const updatedSessionsResponse = await chatService.getChatSessions();
                  if (updatedSessionsResponse.success && updatedSessionsResponse.data) {
                    setSessions(updatedSessionsResponse.data);
                  }
                }
              }
            } else {
              if (sessionsResponse.data.length > 0) {
                const mostRecentSession = sessionsResponse.data[0];
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
                const newSessionResponse = await chatService.createChatSession('Healing Journey', language);
                if (newSessionResponse.success && newSessionResponse.data) {
                  setCurrentSession(newSessionResponse.data);
                  setMessages(newSessionResponse.data.messages);
                  setSessions([newSessionResponse.data]);
                }
              }
            }
          } else {
            const newSessionResponse = await chatService.createChatSession('Healing Journey', language);
            if (newSessionResponse.success && newSessionResponse.data) {
              setCurrentSession(newSessionResponse.data);
              setMessages(newSessionResponse.data.messages);
              setSessions([newSessionResponse.data]);
            }
          }
        } catch (error) {
          console.error('Failed to initialize chat sessions:', error);
          // Fallback session creation
          const initialMessages = {
            en: "Kaya! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
            noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
          };

          const fallbackSession: ChatSession = {
            id: `fallback-session-${Date.now()}`,
            userId: user?.id || '1',
            title: 'Healing Journey',
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
    };

    initializeSessions();
    
    // Hide cultural welcome after 3 seconds
    const timer = setTimeout(() => {
      setCulturalWelcome(false);
    }, 3000);

    return () => clearTimeout(timer);
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
      const initialMessages = {
        en: "Kaya! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
        noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
      };

      const tempSession: ChatSession = {
        id: `guest-session-${Date.now()}`,
        userId: 'guest',
        title: language === 'noongar' ? 'Mooditj Koora' : 'Healing Journey',
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
      const response = await chatService.createChatSession('Healing Journey', language);
      if (response.success && response.data) {
        setCurrentSession(response.data);
        setMessages(response.data.messages);
        const sessionsResponse = await chatService.getChatSessions();
        if (sessionsResponse.success && sessionsResponse.data) {
          setSessions(sessionsResponse.data);
        }
      }
    } catch (error) {
      console.error('Failed to create new chat session:', error);
    }
  };

  // ADD MISSING FUNCTIONS
  const handleSelectSession = async (sessionId: string) => {
    if (isGuest) {
      const session = sessions.find(s => s.id === sessionId);
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
      const response = await chatService.getChatMessages(sessionId);
      if (response.success && response.data) {
        const session = sessions.find(s => s.id === sessionId);
        if (session) {
          const updatedSession = { ...session, messages: response.data };
          setCurrentSession(updatedSession);
          setMessages(response.data);
          if (session.language) {
            setChatSessionLanguage(sessionId, session.language as 'en' | 'noongar');
          }
        }
      }
    } catch (error) {
      console.error('Failed to load session messages:', error);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (isGuest) {
      const updatedSessions = sessions.filter(s => s.id !== sessionId);
      setSessions(updatedSessions);
      
      if (currentSession?.id === sessionId) {
        if (updatedSessions.length > 0) {
          await handleSelectSession(updatedSessions[0].id);
        } else {
          await handleNewChat();
        }
      }
      return;
    }

    try {
      const response = await chatService.deleteChatSession(sessionId);
      if (response.success) {
        const updatedSessions = sessions.filter(s => s.id !== sessionId);
        setSessions(updatedSessions);
        
        if (currentSession?.id === sessionId) {
          if (updatedSessions.length > 0) {
            await handleSelectSession(updatedSessions[0].id);
          } else {
            await handleNewChat();
          }
        }
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  const handleRenameSession = async (sessionId: string, newTitle: string) => {
    if (isGuest) {
      const updatedSessions = sessions.map(s => 
        s.id === sessionId ? { ...s, title: newTitle, updatedAt: new Date().toISOString() } : s
      );
      setSessions(updatedSessions);
      
      if (currentSession?.id === sessionId) {
        setCurrentSession({ ...currentSession, title: newTitle, updatedAt: new Date().toISOString() });
      }
      return;
    }

    try {
      const response = await chatService.updateChatSession(sessionId, { title: newTitle });
      if (response.success && response.data) {
        const updatedSessions = sessions.map(s => 
          s.id === sessionId ? { ...s, title: newTitle, updatedAt: response.data?.updatedAt || new Date().toISOString() } : s
        );
        setSessions(updatedSessions);
        
        if (currentSession?.id === sessionId && response.data) {
          setCurrentSession({ ...currentSession, title: newTitle, updatedAt: response.data.updatedAt });
        }
      }
    } catch (error) {
      console.error('Failed to rename session:', error);
    }
  };

  const playWelcomeAudio = () => {
    setIsPlayingAudio(true);
    setTimeout(() => setIsPlayingAudio(false), 3000);
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const messageText = inputText;
    setInputText('');
    setIsLoading(true);

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
      setMessages(prev => [...prev, userMessage]);

      const response = await chatService.sendMessage(
        currentSession?.id || 'guest-session', 
        messageText
      );

      setTimeout(() => {
        setIsLoading(false);
      }, 3000);

      if (response.success && response.data) {
        const { userMessage: backendUserMessage, response: aiResponse, patientMessage } = response.data;
        
        setMessages(prev => {
          const withoutLast = prev.slice(0, -1);
          return [...withoutLast, backendUserMessage, patientMessage, aiResponse];
        });

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

      setMessages(prev => {
        const withoutLast = prev.slice(0, -1);
        return [...withoutLast, userMessage, fallbackMessage];
      });

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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleSendVoiceMessage = async (audioBlob: Blob, duration: number) => {
    console.log('🎤 Sending voice message:', audioBlob, 'duration:', duration);
    setIsLoading(true);
    setShowVoiceRecorder(false);
    
    try {
      const audioUrl = URL.createObjectURL(audioBlob);
      
      const userVoiceMessage: Message = {
        id: (Date.now()).toString(),
        text: 'Voice message...',
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
      
      const response = await chatService.sendVoiceMessage(
        currentSession?.id || 'guest-session',
        audioBlob,
        duration
      );
      
      if (response.success && response.data) {
        const { userMessage: backendUserMessage, response: aiResponse } = response.data;
        
        const updatedUserMessage = {
          ...backendUserMessage,
          audioUrl: audioUrl
        };
        
        if (!isGuest && currentSession) {
          const currentMessages = currentSession.messages;
          const updatedMessages = [
            ...currentMessages.slice(0, -1),
            updatedUserMessage,
            aiResponse
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
            const withoutLast = prev.slice(0, -1);
            return [...withoutLast, updatedUserMessage, aiResponse];
          });
        }
        
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

  // ADD MISSING SYMPTOM FUNCTION
  const handleSymptomSelection = (symptoms: { name: string; tags: string[] }[]) => {
    if (symptoms.length === 0) return;
    
    const symptomText = symptoms.map(symptom => 
      `${symptom.name} (${symptom.tags.join(', ')})`
    ).join(', ');
    
    const messageText = `I'm experiencing these symptoms: ${symptomText}`;
    
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

    setMessages(prev => [...prev, userMessage]);

    setTimeout(() => {
      setShowSymptomSelector(false);
    }, 300);

    // Send to assistant
    sendMessageToAssistant(messageText, symptoms);
  };

  // ADD MISSING SEND MESSAGE TO ASSISTANT FUNCTION
  const sendMessageToAssistant = async (messageText: string, symptoms: { name: string; tags: string[] }[]) => {
    setIsLoading(true);
    
    try {
      const symptomNames = symptoms.map(symptom => symptom.name);
      const response = await chatService.sendSymptoms(currentSession?.id || 'guest-session', symptomNames);
      setTimeout(() => {
        setIsLoading(false);
      }, 3000);
      if (response.success && response.data) {
        if (!isGuest && currentSession) {
          const updatedMessages = [...currentSession.messages,response.patientMessage, response.data,];
          const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
          setCurrentSession(updatedSession);
          setMessages(updatedMessages);
          
          const updatedSessions = sessions.map(s => 
            s.id === currentSession.id ? updatedSession : s
          );
          setSessions(updatedSessions);
        } else {
          setMessages(prev => [...prev, response.patientMessage, response.data!]);
        }
      } else {
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
      console.error('Failed to send symptoms to assistant:', error);
      
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

      if (!isGuest && currentSession) {
        const updatedMessages = [...currentSession.messages, errorMessage];
        const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
        setCurrentSession(updatedSession);
        setMessages(updatedMessages);
        
        const updatedSessions = sessions.map(s => 
          s.id === currentSession.id ? updatedSession : s
        );
        setSessions(updatedSessions);
      } else {
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout showLanguageButton={false}>
      <Header onLogout={onLogout} showLanguage={false} user={user} isGuest={isGuest} onSignIn={onSignIn} />
      
      {/* Animated Background */}
      <div className="fixed inset-0 -z-10">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1519066629447-267fffa62d4b?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80')`,
          }}
        />
        <div className={`absolute inset-0 ${
          theme === "dark"
            ? "bg-gradient-to-br from-emerald-900/80 via-slate-900/70 to-amber-900/60"
            : "bg-gradient-to-br from-emerald-600/60 via-blue-500/50 to-amber-400/50"
        }`} />
        
        {/* Animated Cultural Patterns */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-10 left-10 w-20 h-20 bg-amber-300 rounded-full animate-pulse"></div>
          <div className="absolute bottom-20 right-20 w-16 h-16 bg-emerald-300 rounded-full animate-bounce"></div>
        </div>
      </div>

      <div className="h-screen p-4 pt-20 overflow-hidden">
        <div className="max-w-7xl mx-auto h-full">
          {/* Cultural Welcome Animation */}
          {culturalWelcome && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
              <div className="text-center animate-bounce">
                <div className="text-6xl mb-4">🌿</div>
                <h2 className="text-4xl font-bold text-white mb-2">Kaya! Welcome</h2>
                <p className="text-xl text-amber-300">Ngangk Moort Boodja</p>
                <p className="text-white/80 mt-2">Your healing journey begins</p>
              </div>
            </div>
          )}

          <div className={`transition-all duration-300 ${showSymptomSelector ? 'blur-sm' : ''}`}>
            <BackButton to="/dashboard" />
          </div>
          
          <div className={`flex h-[calc(110vh-180px)] rounded-2xl shadow-2xl overflow-hidden transition-all duration-300 my-6 mx-2 ${
            theme === 'dark' ? 'bg-slate-800/80 backdrop-blur-sm' : 'bg-white/90 backdrop-blur-sm'
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
              {/* Chat Header - Cultural Design */}
              <div className={`p-4 flex items-center justify-between border-b transition-colors duration-300 ${
                theme === 'dark' 
                  ? 'bg-gradient-to-r from-emerald-800 to-amber-800 border-amber-700' 
                  : 'bg-gradient-to-r from-emerald-500 to-amber-500 border-amber-400'
              }`}>
                <div className="flex items-center space-x-3">
                  {!isGuest && (
                    <button
                      onClick={() => setShowSidebar(!showSidebar)}
                      className={`p-2 rounded-lg transition-colors ${
                        theme === 'dark' 
                          ? 'text-white hover:bg-amber-700/50' 
                          : 'text-white hover:bg-amber-600/50'
                      }`}
                      title="Toggle chat history"
                    >
                      <Menu className="w-5 h-5" />
                    </button>
                  )}
                  <div>
                    <h2 className={`font-bold text-lg ${
                      theme === 'dark' ? 'text-white' : 'text-white'
                    }`}>
                      {currentSession?.title || 'Ngangk Moort - Healing Journey'}
                    </h2>
                    <p className={`text-sm ${
                      theme === 'dark' ? 'text-amber-200' : 'text-amber-100'
                    }`}>
                      {language === 'noongar' ? 'Boola ngaangk - Always healing' : 'Available 24/7'} • {t('voiceTextSupport')}
                      {isGuest && (
                        <span className={`ml-2 ${
                          theme === 'dark' ? 'text-yellow-300' : 'text-yellow-200'
                        }`}>{t('guestUser')}</span>
                      )}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={playWelcomeAudio}
                    className="flex items-center space-x-2 px-3 py-1 rounded-lg transition-colors bg-white/20 text-white hover:bg-white/30"
                  >
                    <Volume2 className="w-4 h-4" />
                    <span className="text-sm">Noongar Audio</span>
                  </button>
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
                  <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                </div>
              </div>

              {/* Cultural Elements Bar */}
              <div className="flex items-center justify-center space-x-4 py-3 border-b bg-white/10 backdrop-blur-sm">
                {culturalElements.map((element, index) => (
                  <div key={index} className="flex items-center space-x-2 text-sm">
                    <span className="text-lg">{element.emoji}</span>
                    <div className="text-center">
                      <div className="font-semibold text-white text-xs">{element.title}</div>
                      <div className="text-amber-200 text-xs">{element.noongarDesc}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Messages Area */}
              <div className={`flex-1 overflow-y-auto p-6 space-y-4 transition-colors duration-300 ${
                theme === 'dark' ? 'bg-slate-900/50' : 'bg-gray-50/80'
              }`}>
                {messages && messages.length > 0 ? (
                  messages.map((message, index) => {
                    return (
                      <div key={message.id} className={`flex items-start space-x-3 animate-fade-in ${message.isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
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
                            theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
                          }`}>
                            {message.isUser 
                              ? (isGuest ? 'Guest User' : user?.firstName || 'User')
                              : 'Ngangk Moort - CareMate'
                            }
                          </div>
                          
                          {/* Message Bubble */}
                          <div className={`max-w-md px-4 py-3 rounded-2xl relative group transition-all duration-300 hover:scale-105 ${
                            message.isUser 
                              ? theme === 'dark' 
                                ? 'bg-amber-600 text-white shadow-lg' 
                                : 'bg-amber-500 text-white shadow-lg'
                              : theme === 'dark'
                                ? 'bg-emerald-700 text-slate-100 shadow-lg'
                                : 'bg-emerald-500 text-white shadow-lg'
                          }`}>
                            <p className="text-sm break-words whitespace-pre-wrap">{message.text}</p>
                            <div className="flex items-center justify-between mt-1">
                              <div className={`text-xs ${
                                message.isUser 
                                  ? 'text-amber-100' 
                                  : theme === 'dark' 
                                    ? 'text-emerald-200' 
                                    : 'text-emerald-100'
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
                        theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
                      }`}>
                        Ngangk Moort - CareMate
                      </div>
                      <div className={`max-w-md px-4 py-3 rounded-2xl transition-colors duration-300 ${
                        theme === 'dark' 
                          ? 'bg-emerald-700 text-slate-100' 
                          : 'bg-emerald-500 text-white'
                      }`}>
                        <div className="flex items-center space-x-2">
                          <div className={`animate-spin rounded-full h-4 w-4 border-b-2 ${
                            theme === 'dark' ? 'border-slate-300' : 'border-white'
                          }`}></div>
                          <span className="text-sm">
                            {language === 'noongar' ? 'Ngangk wangkiny...' : 'Assistant is typing...'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Auto-scroll anchor */}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area - Cultural Design */}
              <div className={`p-4 border-t transition-colors duration-300 ${
                theme === 'dark' 
                  ? 'bg-slate-800/80 border-amber-700' 
                  : 'bg-white/90 border-amber-300'
              }`}>
                <div className="flex items-center space-x-3">
                  {/* Text Input */}
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={language === 'noongar' ? 'Warrima ngaangk...' : 'Type your health question...'}
                      className={`w-full px-4 py-3 rounded-full focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent transition-colors duration-300 ${
                        theme === 'dark'
                          ? 'bg-slate-700 border-amber-600 text-white placeholder-slate-400'
                          : 'bg-white border border-amber-300 text-gray-900 placeholder-gray-500'
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
                          ? 'bg-emerald-600 hover:bg-emerald-700' 
                          : 'bg-emerald-500 hover:bg-emerald-600'
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
                        ? 'bg-amber-600 hover:bg-amber-500' 
                        : 'bg-amber-500 hover:bg-amber-600'
                    }`}
                    title="Select symptoms"
                  >
                    <Image className="w-5 h-5" />
                  </button>

                  {/* Send Button */}
                  <button
                    onClick={sendMessage}
                    disabled={!inputText.trim() || isLoading}
                    className={`p-3 rounded-full text-white transition-all transform hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed ${
                      theme === 'dark' 
                        ? 'bg-amber-600 hover:bg-amber-500' 
                        : 'bg-amber-500 hover:bg-amber-600'
                    }`}
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

      {/* Add custom animations */}
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.5s ease-out;
        }
      `}</style>
    </Layout>
  );
};

export default AssistantPage;