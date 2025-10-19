import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Send, Bot, Menu, Image, Mic, Volume2, Cpu, Brain, Sparkles } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import ChatSidebar from '../components/ChatSidebar/ChatSidebar';
import Avatar from '../components/Avatar/Avatar';
import VoiceRecorder from '../components/VoiceRecorder/VoiceRecorder';
import SymptomSelector from '../components/SymptomSelector/SymptomSelector';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';
import { chatService } from '../services/api';
import { Message, ChatSession } from '../types';

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
  const [culturalWelcome, setCulturalWelcome] = useState(true);
  const [processingStage, setProcessingStage] = useState<'nlp' | 'ml' | 'response' | null>(null);
  
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

  // Processing stages with delays
  const processingStages = [
    { stage: 'nlp', text: 'Analyzing your message with NLP...', duration: 2000, icon: Cpu, color: 'blue' },
    { stage: 'ml', text: 'Processing with ML models...', duration: 2500, icon: Brain, color: 'purple' },
    { stage: 'response', text: 'Generating personalized response...', duration: 1500, icon: Sparkles, color: 'amber' }
  ];

  // Auto-scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, processingStage]);

  // Initialize chat sessions on component mount
  useEffect(() => {
    const initializeSessions = async () => {
      const sessionId = searchParams.get('sessionId');
      
      if (isGuest) {
        // For guest users, load from localStorage or create new
        const savedGuestSessions = localStorage.getItem('guestChatSessions');
        const savedCurrentSessionId = localStorage.getItem('currentGuestSessionId');
        
        if (savedGuestSessions && savedCurrentSessionId) {
          const guestSessions: ChatSession[] = JSON.parse(savedGuestSessions);
          setSessions(guestSessions);
          
          const currentSession = guestSessions.find(s => s.id === savedCurrentSessionId);
          if (currentSession) {
            setCurrentSession(currentSession);
            setMessages(currentSession.messages);
          } else if (guestSessions.length > 0) {
            setCurrentSession(guestSessions[0]);
            setMessages(guestSessions[0].messages);
          }
        } else {
          // Create initial guest session
          createNewGuestSession();
        }
      } else {
        // For logged-in users, load sessions from backend
        try {
          const sessionsResponse = await chatService.getChatSessions();
          if (sessionsResponse.success && sessionsResponse.data) {
            setSessions(sessionsResponse.data);
            
            if (sessionId) {
              const specificSession = sessionsResponse.data.find(s => s.id === sessionId);
              if (specificSession) {
                await loadSessionMessages(specificSession);
              } else {
                await createNewSession();
              }
            } else {
              if (sessionsResponse.data.length > 0) {
                await loadSessionMessages(sessionsResponse.data[0]);
              } else {
                await createNewSession();
              }
            }
          } else {
            await createNewSession();
          }
        } catch (error) {
          console.error('Failed to initialize chat sessions:', error);
          createFallbackSession();
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

  // Helper functions for session management
  const createNewGuestSession = () => {
    const initialMessages = {
      en: "Kaya! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
      noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
    };

    const tempSession: ChatSession = {
      id: `guest-${Date.now()}`,
      userId: 'guest',
      title: language === 'noongar' ? 'Mooditj Koora' : 'Healing Journey',
      messages: [
        {
          id: `welcome-${Date.now()}`,
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
    setSessions([tempSession]);
    
    // Save to localStorage
    localStorage.setItem('guestChatSessions', JSON.stringify([tempSession]));
    localStorage.setItem('currentGuestSessionId', tempSession.id);
  };

  const loadSessionMessages = async (session: ChatSession) => {
    try {
      const messagesResponse = await chatService.getChatMessages(session.id);
      if (messagesResponse.success && messagesResponse.data) {
        const sessionWithMessages = { ...session, messages: messagesResponse.data };
        setCurrentSession(sessionWithMessages);
        setMessages(messagesResponse.data);
      } else {
        setCurrentSession(session);
        setMessages(session.messages);
      }
      if (session.language) {
        setChatSessionLanguage(session.id, session.language as 'en' | 'noongar');
      }
    } catch (error) {
      setCurrentSession(session);
      setMessages(session.messages);
    }
  };

  const createNewSession = async () => {
    try {
      const response = await chatService.createChatSession('Healing Journey', language);
      if (response.success && response.data) {
        setCurrentSession(response.data);
        setMessages(response.data.messages);
        setSessions([response.data]);
      }
    } catch (error) {
      createFallbackSession();
    }
  };

  const createFallbackSession = () => {
    const initialMessages = {
      en: "Kaya! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
      noongar: "Kaya! Ngany mooditj moort. Ngany mooditj wangkiny, ngany mooditj koora, ngany mooditj wangkiny. Ngany mooditj?"
    };

    const fallbackSession: ChatSession = {
      id: `fallback-${Date.now()}`,
      userId: user?.id || '1',
      title: 'Healing Journey',
      messages: [
        {
          id: `welcome-${Date.now()}`,
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
    setSessions([fallbackSession]);
  };

  // Save guest sessions to localStorage whenever they change
  useEffect(() => {
    if (isGuest && sessions.length > 0 && currentSession) {
      localStorage.setItem('guestChatSessions', JSON.stringify(sessions));
      localStorage.setItem('currentGuestSessionId', currentSession.id);
    }
  }, [sessions, currentSession, isGuest]);

  // Chat session management functions
  const handleNewChat = async () => {
    if (isGuest) {
      createNewGuestSession();
    } else {
      await createNewSession();
    }
  };

  const handleSelectSession = async (sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      if (isGuest) {
        setCurrentSession(session);
        setMessages(session.messages);
      } else {
        await loadSessionMessages(session);
      }
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    const updatedSessions = sessions.filter(s => s.id !== sessionId);
    setSessions(updatedSessions);
    
    if (currentSession?.id === sessionId) {
      if (updatedSessions.length > 0) {
        await handleSelectSession(updatedSessions[0].id);
      } else {
        await handleNewChat();
      }
    }
  };

  const handleRenameSession = async (sessionId: string, newTitle: string) => {
    const updatedSessions = sessions.map(s => 
      s.id === sessionId ? { ...s, title: newTitle, updatedAt: new Date().toISOString() } : s
    );
    setSessions(updatedSessions);
    
    if (currentSession?.id === sessionId) {
      setCurrentSession(prev => prev ? { ...prev, title: newTitle, updatedAt: new Date().toISOString() } : null);
    }
  };

  // Enhanced sendMessage with processing animations
  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const messageText = inputText;
    setInputText('');
    setIsLoading(true);

    const userMessage: Message = {
      id: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
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

    // Add user message immediately
    setMessages(prev => [...prev, userMessage]);

    // Start processing animation sequence
    let currentDelay = 0;
    const processingMessageIds: string[] = [];
    
    processingStages.forEach((stage, index) => {
      setTimeout(() => {
        setProcessingStage(stage.stage as 'nlp' | 'ml' | 'response');
        
        // Add processing message to chat
        const processingMessage: Message = {
          id: `processing-${stage.stage}-${Date.now()}-${index}`,
          text: stage.text,
          isUser: false,
          timestamp: new Date().toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
          }),
          createdAt: new Date().toISOString(),
          type: 'processing'
        };
        
        processingMessageIds.push(processingMessage.id);
        setMessages(prev => [...prev, processingMessage]);
        
      }, currentDelay);
      currentDelay += stage.duration;
    });

    // Send actual message after all animations
    setTimeout(async () => {
      try {
        const response = await chatService.sendMessage(
          currentSession?.id || 'guest-session', 
          messageText
        );

        if (response.success && response.data) {
          const { userMessage: backendUserMessage, response: aiResponse, patientMessage, patientMessageNoongar } = response.data;

          // Remove processing messages and add final response
          setMessages(prev => {
            const withoutProcessing = prev.filter(msg => !processingMessageIds.includes(msg.id));
            return [...withoutProcessing, backendUserMessage, patientMessageNoongar, patientMessage, aiResponse];
          });

          // Update session
          if (currentSession) {
            const updatedMessages = [...currentSession.messages, backendUserMessage, aiResponse];
            const updatedSession = { 
              ...currentSession, 
              messages: updatedMessages, 
              updatedAt: new Date().toISOString() 
            };
            setCurrentSession(updatedSession);
            setSessions(prev => prev.map(s => s.id === currentSession.id ? updatedSession : s));
          }
        } else {
          throw new Error('Failed to send message');
        }
      } catch (error) {
        console.error('Chat message failed:', error);
        const fallbackMessage: Message = {
          id: `ai-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
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
          const withoutProcessing = prev.filter(msg => !processingMessageIds.includes(msg.id));
          return [...withoutProcessing, userMessage, fallbackMessage];
        });
      } finally {
        setIsLoading(false);
        setProcessingStage(null);
      }
    }, currentDelay + 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Voice message handler
  const handleSendVoiceMessage = async (audioBlob: Blob, duration: number) => {
    console.log('🎤 Sending voice message:', audioBlob, 'duration:', duration);
    setIsLoading(true);
    setShowVoiceRecorder(false);
    
    try {
      const audioUrl = URL.createObjectURL(audioBlob);
      
      const userVoiceMessage: Message = {
        id: `voice-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
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
      
      // Add user voice message immediately
      setMessages(prev => [...prev, userVoiceMessage]);

      // Start processing animation for voice message
      let currentDelay = 0;
      const processingMessageIds: string[] = [];
      
      processingStages.forEach((stage, index) => {
        setTimeout(() => {
          setProcessingStage(stage.stage as 'nlp' | 'ml' | 'response');
          
          const processingMessage: Message = {
            id: `processing-voice-${stage.stage}-${Date.now()}-${index}`,
            text: stage.text,
            isUser: false,
            timestamp: new Date().toLocaleTimeString('en-US', { 
              hour: 'numeric', 
              minute: '2-digit',
              hour12: true 
            }),
            createdAt: new Date().toISOString(),
            type: 'processing'
          };
          
          processingMessageIds.push(processingMessage.id);
          setMessages(prev => [...prev, processingMessage]);
          
        }, currentDelay);
        currentDelay += stage.duration;
      });

      setTimeout(async () => {
        try {
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
            
            setMessages(prev => {
              const withoutProcessing = prev.filter(msg => !processingMessageIds.includes(msg.id));
              return [...withoutProcessing, updatedUserMessage, aiResponse];
            });

            if (currentSession) {
              const updatedMessages = [...currentSession.messages, updatedUserMessage, aiResponse];
              const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
              setCurrentSession(updatedSession);
              
              setSessions(prev => prev.map(s => 
                s.id === currentSession.id ? updatedSession : s
              ));

              // Save guest session
              if (isGuest) {
                const updatedSessions = sessions.map(s => 
                  s.id === currentSession.id ? updatedSession : s
                );
                localStorage.setItem('guestChatSessions', JSON.stringify(updatedSessions));
              }
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
            throw new Error('Voice message failed');
          }
        } catch (error) {
          console.error('Voice message failed:', error);
          const fallbackMessage: Message = {
            id: `ai-voice-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
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
            const withoutProcessing = prev.filter(msg => !processingMessageIds.includes(msg.id));
            return [...withoutProcessing, fallbackMessage];
          });
        } finally {
          setIsLoading(false);
          setProcessingStage(null);
        }
      }, currentDelay + 1000);

    } catch (error) {
      console.error('Voice message failed:', error);
      setIsLoading(false);
      setProcessingStage(null);
    }
  };

  // Symptom selection handler
  const handleSymptomSelection = (symptoms: { name: string; tags: string[] }[]) => {
    if (symptoms.length === 0) return;
    
    const symptomText = symptoms.map(symptom => 
      `${symptom.name} (${symptom.tags.join(', ')})`
    ).join(', ');
    
    const messageText = `I'm experiencing these symptoms: ${symptomText}`;
    
    const userMessage: Message = {
      id: `symptoms-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
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
    setShowSymptomSelector(false);

    // Send symptoms to assistant
    sendSymptomsToAssistant(userMessage, symptoms);
  };

  const sendSymptomsToAssistant = async (userMessage: Message, symptoms: { name: string; tags: string[] }[]) => {
    setIsLoading(true);

    // Start processing animation for symptoms
    let currentDelay = 0;
    const processingMessageIds: string[] = [];
    
    processingStages.forEach((stage, index) => {
      setTimeout(() => {
        setProcessingStage(stage.stage as 'nlp' | 'ml' | 'response');
        
        const processingMessage: Message = {
          id: `processing-symptoms-${stage.stage}-${Date.now()}-${index}`,
          text: stage.text,
          isUser: false,
          timestamp: new Date().toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
          }),
          createdAt: new Date().toISOString(),
          type: 'processing'
        };
        
        processingMessageIds.push(processingMessage.id);
        setMessages(prev => [...prev, processingMessage]);
        
      }, currentDelay);
      currentDelay += stage.duration;
    });

    setTimeout(async () => {
      try {
        const symptomNames = symptoms.map(symptom => symptom.name);
        const response = await chatService.sendSymptoms(currentSession?.id || 'guest-session', symptomNames);
        
        if (response.success && response.data) {
          setMessages(prev => {
            const withoutProcessing = prev.filter(msg => !processingMessageIds.includes(msg.id));
            return [...withoutProcessing, response.patientMessageNoongar, response.patientMessage, response.data!];
          });

          if (currentSession) {
            const updatedMessages = [...currentSession.messages, userMessage, response.patientMessageNoongar, response.patientMessage, response.data!];
            const updatedSession = { ...currentSession, messages: updatedMessages, updatedAt: new Date().toISOString() };
            setCurrentSession(updatedSession);
            
            setSessions(prev => prev.map(s => 
              s.id === currentSession.id ? updatedSession : s
            ));

            // Save guest session
            if (isGuest) {
              const updatedSessions = sessions.map(s => 
                s.id === currentSession.id ? updatedSession : s
              );
              localStorage.setItem('guestChatSessions', JSON.stringify(updatedSessions));
            }
          }
        } else {
          throw new Error('Symptoms processing failed');
        }
      } catch (error) {
        console.error('Failed to send symptoms to assistant:', error);
        const fallbackMessage: Message = {
          id: `ai-symptoms-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          text: "Thank you for sharing your symptoms. I've received your information and will help you with your health concerns.",
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
          const withoutProcessing = prev.filter(msg => !processingMessageIds.includes(msg.id));
          return [...withoutProcessing, fallbackMessage];
        });
      } finally {
        setIsLoading(false);
        setProcessingStage(null);
      }
    }, currentDelay + 1000);
  };

  // Get processing stage styling
  const getProcessingStageStyle = (stage: string) => {
    const stageInfo = processingStages.find(s => s.stage === stage);
    if (!stageInfo) return { icon: Cpu, color: 'blue', gradient: 'from-blue-900/50 to-blue-800/30' };
    
    const gradients = {
      blue: theme === 'dark' ? 'from-blue-900/50 to-blue-800/30' : 'from-blue-400/20 to-blue-300/10',
      purple: theme === 'dark' ? 'from-purple-900/50 to-purple-800/30' : 'from-purple-400/20 to-purple-300/10',
      amber: theme === 'dark' ? 'from-amber-900/50 to-amber-800/30' : 'from-amber-400/20 to-amber-300/10'
    };
    
    return {
      icon: stageInfo.icon,
      color: stageInfo.color,
      gradient: gradients[stageInfo.color as keyof typeof gradients]
    };
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
      </div>

      <div className="h-screen p-4 pt-20 overflow-hidden">
        <div className="max-w-7xl mx-auto h-full">
          {/* Cultural Welcome Animation */}
          {culturalWelcome && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-fade-in">
              <div className="text-center animate-bounce">
                <div className="text-6xl mb-4">🌿</div>
                <h2 className="text-4xl font-bold text-white mb-2">Kaya! Welcome</h2>
                <p className="text-xl text-amber-300">Ngangk Moort Boodja</p>
                <p className="text-white/80 mt-2">Your healing journey begins</p>
              </div>
            </div>
          )}

          <BackButton to="/dashboard" />
          
          <div className={`flex h-[calc(100vh-140px)] rounded-2xl shadow-2xl overflow-hidden transition-all duration-300 my-4 ${
            theme === 'dark' ? 'bg-slate-800/80 backdrop-blur-sm' : 'bg-white/90 backdrop-blur-sm'
          }`}>
            
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
              <div className={`p-4 flex items-center justify-between border-b ${
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
                    >
                      <Menu className="w-5 h-5" />
                    </button>
                  )}
                  <div>
                    <h2 className="font-bold text-lg text-white">
                      {currentSession?.title || 'Ngangk Moort - Healing Journey'}
                    </h2>
                    <p className={`text-sm ${
                      theme === 'dark' ? 'text-amber-200' : 'text-amber-100'
                    }`}>
                      {language === 'noongar' ? 'Boola ngaangk - Always healing' : 'Available 24/7'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                    <Bot className="w-5 h-5 text-white" />
                  </div>
                </div>
              </div>

              {/* Messages Area */}
              <div className={`flex-1 overflow-y-auto p-6 space-y-4 ${
                theme === 'dark' ? 'bg-slate-900/50' : 'bg-gray-50/80'
              }`}>
                {messages.map((message) => (
                  <div 
                    key={message.id} 
                    className={`flex items-start space-x-3 ${
                      message.isUser ? 'flex-row-reverse space-x-reverse' : ''
                    } animate-message-in`}
                  >
                    <Avatar 
                      type={message.isUser ? 'user' : 'assistant'}
                      userName={message.isUser ? user?.firstName : undefined}
                      isGuest={message.isUser ? isGuest : false}
                      size="md"
                    />
                    
                    <div className={`flex flex-col ${message.isUser ? 'items-end' : 'items-start'} max-w-[70%]`}>
                      <div className={`text-xs mb-1 px-2 ${
                        theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
                      }`}>
                        {message.isUser 
                          ? (isGuest ? 'Guest User' : user?.firstName || 'User')
                          : 'Ngangk Moort - CareMate'
                        }
                      </div>
                      
                      {/* Processing Message Bubble */}
                      {message.type === 'processing' && processingStage && (
                        <div className={`max-w-md px-4 py-3 rounded-2xl bg-gradient-to-r ${
                          getProcessingStageStyle(processingStage).gradient
                        } border ${
                          theme === 'dark' ? 'border-blue-500/30' : 'border-blue-300'
                        } animate-pulse`}>
                          <div className="flex items-center space-x-3">
                            <div className={`p-2 rounded-full bg-${getProcessingStageStyle(processingStage).color}-500/20`}>
                              {React.createElement(getProcessingStageStyle(processingStage).icon, {
                                className: `w-4 h-4 text-${getProcessingStageStyle(processingStage).color}-400 animate-pulse`
                              })}
                            </div>
                            <div className="flex-1">
                              <div className="text-sm font-medium text-white">
                                {message.text}
                              </div>
                              <div className="flex space-x-1 mt-2">
                                {[1, 2, 3].map((dot) => (
                                  <div
                                    key={dot}
                                    className={`w-2 h-2 rounded-full ${
                                      theme === 'dark' ? 'bg-blue-400' : 'bg-blue-500'
                                    } animate-bounce`}
                                    style={{ animationDelay: `${dot * 0.2}s` }}
                                  />
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {/* Regular Message Bubble */}
                      {message.type !== 'processing' && (
                        <div className={`max-w-md px-4 py-3 rounded-2xl transition-all duration-300 hover:scale-105 ${
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
                      )}
                    </div>
                  </div>
                ))}
                
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className={`p-4 border-t ${
                theme === 'dark' 
                  ? 'bg-slate-800/80 border-amber-700' 
                  : 'bg-white/90 border-amber-300'
              }`}>
                <div className="flex items-center space-x-3">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={language === 'noongar' ? 'Warrima ngaangk...' : 'Type your health question...'}
                      className={`w-full px-4 py-3 rounded-full focus:outline-none focus:ring-2 focus:ring-amber-500 transition-colors duration-300 ${
                        theme === 'dark'
                          ? 'bg-slate-700 text-white placeholder-slate-400'
                          : 'bg-white text-gray-900 placeholder-gray-500 border border-amber-300'
                      }`}
                      disabled={isLoading}
                    />
                  </div>

                  {/* Voice Recorder Button */}
                  {!showVoiceRecorder ? (
                    <button
                      onClick={() => setShowVoiceRecorder(true)}
                      className={`p-3 rounded-full text-white transition-all transform hover:scale-105 ${
                        theme === 'dark' 
                          ? 'bg-emerald-600 hover:bg-emerald-500' 
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
                      onTranscribedText={(text) => setInputText(text)}
                    />
                  )}

                  <button
                    onClick={() => setShowSymptomSelector(true)}
                    className={`p-3 rounded-full text-white transition-all transform hover:scale-105 ${
                      theme === 'dark' 
                        ? 'bg-amber-600 hover:bg-amber-500' 
                        : 'bg-amber-500 hover:bg-amber-600'
                    }`}
                  >
                    <Image className="w-5 h-5" />
                  </button>

                  <button
                    onClick={sendMessage}
                    disabled={!inputText.trim() || isLoading}
                    className={`p-3 rounded-full text-white transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed ${
                      theme === 'dark' 
                        ? 'bg-amber-600 hover:bg-amber-500' 
                        : 'bg-amber-500 hover:bg-amber-600'
                    }`}
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
        onClose={() => setShowSymptomSelector(false)}
        onSelectSymptoms={handleSymptomSelection}
        isLoading={isLoading}
      />

      {/* Add Tailwind CSS animations */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes messageIn {
          from { opacity: 0; transform: translateY(20px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        .animate-fade-in {
          animation: fadeIn 0.6s ease-out;
        }
        .animate-message-in {
          animation: messageIn 0.5s ease-out;
        }
      `}</style>
    </Layout>
  );
};

export default AssistantPage;