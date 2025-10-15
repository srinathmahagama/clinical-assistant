import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, MessageCircle, Clock, Mic, FileText, X, ArrowRight, Bot } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import ChatOptions from '../components/ChatOptions/ChatOptions';
import ConfirmationDialog from '../components/ConfirmationDialog/ConfirmationDialog';
import { useTheme } from '../contexts/ThemeContext';
import { useLanguage } from '../contexts/LanguageContext';
import { chatService } from '../services/api';
import { ChatSession as ApiChatSession } from '../types';

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: string;
  messageCount: number;
  hasVoiceMessages: boolean;
}

interface HistoryPageProps {
  onNavigate: (page: string) => void;
  onLogout: () => void;
  user?: any;
  isGuest?: boolean;
}

const HistoryPage: React.FC<HistoryPageProps> = ({ onNavigate, onLogout, user, isGuest }) => {
  const navigate = useNavigate();
  const { theme } = useTheme();
  const { t } = useLanguage();
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);

  useEffect(() => {
    loadChatSessions();
  }, []);

  const loadChatSessions = async () => {
    try {
      // Fetch chat sessions from backend API
      const response = await chatService.getChatSessions();
      
      if (response.success && response.data) {
        // Transform API data to match the interface expected by the UI
        const transformedSessions: ChatSession[] = response.data.map((session: ApiChatSession) => {
          const lastMessage = session.messages && session.messages.length > 0 
            ? session.messages[session.messages.length - 1].text 
            : 'No messages yet';
          
          const hasVoiceMessages = session.messages && session.messages.some(msg => msg.type === 'voice');
          
          // Calculate relative time
          const createdAt = new Date(session.createdAt);
          const now = new Date();
          const diffInHours = Math.floor((now.getTime() - createdAt.getTime()) / (1000 * 60 * 60));
          const diffInDays = Math.floor(diffInHours / 24);
          
          let timestamp = '';
          if (diffInHours < 1) {
            timestamp = 'Just now';
          } else if (diffInHours < 24) {
            timestamp = `${diffInHours} hour${diffInHours > 1 ? 's' : ''} ago`;
          } else if (diffInDays < 7) {
            timestamp = `${diffInDays} day${diffInDays > 1 ? 's' : ''} ago`;
          } else {
            timestamp = createdAt.toLocaleDateString();
          }
          
          return {
            id: session.id,
            title: session.title,
            lastMessage: lastMessage,
            timestamp: timestamp,
            messageCount: session.messages ? session.messages.length : 0,
            hasVoiceMessages: hasVoiceMessages
          };
        });
        
        setChatSessions(transformedSessions);
      } else {
        console.error('Failed to load chat sessions:', response.message);
        // Fallback to empty array if API fails
        setChatSessions([]);
      }
    } catch (error) {
      console.error('Failed to load chat sessions:', error);
      // Fallback to empty array if API fails
      setChatSessions([]);
    } finally {
      setIsLoading(false);
    }
  };

  const filteredSessions = chatSessions.filter(session =>
    session.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    session.lastMessage.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleStartNewChat = () => {
    navigate('/assistant');
  };

  const handleViewChat = (sessionId: string) => {
    // Navigate to assistant page with the specific session ID
    navigate(`/assistant?sessionId=${sessionId}`);
  };

  const handleDeleteChat = (sessionId: string) => {
    setSessionToDelete(sessionId);
    setShowDeleteDialog(true);
  };

  const confirmDelete = async () => {
    if (sessionToDelete) {
      try {
        // Call backend API to delete session
        const response = await chatService.deleteChatSession(sessionToDelete);
        
        if (response.success) {
          // Update local state
          setChatSessions(prev => prev.filter(session => session.id !== sessionToDelete));
        } else {
          console.error('Failed to delete session:', response.message);
          // You could show a toast notification here
        }
      } catch (error) {
        console.error('Error deleting session:', error);
        // You could show a toast notification here
      } finally {
        setSessionToDelete(null);
        setShowDeleteDialog(false);
      }
    }
  };

  const cancelDelete = () => {
    setShowDeleteDialog(false);
    setSessionToDelete(null);
  };

  const handleEditTitle = (sessionId: string, currentTitle: string) => {
    setEditingId(sessionId);
    setEditTitle(currentTitle);
  };

  const handleSaveTitle = async (sessionId: string) => {
    if (editTitle.trim()) {
      try {
        // Call backend API to update session title
        const response = await chatService.updateChatSession(sessionId, { title: editTitle.trim() });
        
        if (response.success) {
          // Update local state
          setChatSessions(prev => prev.map(session => 
            session.id === sessionId 
              ? { ...session, title: editTitle.trim() }
              : session
          ));
        } else {
          console.error('Failed to update title:', response.message);
          // You could show a toast notification here
        }
      } catch (error) {
        console.error('Error updating title:', error);
        // You could show a toast notification here
      }
    }
    setEditingId(null);
    setEditTitle('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditTitle('');
  };

  return (
    <Layout showLanguageButton={false}>
      <Header onLogout={onLogout} showLanguage={false} user={user} isGuest={isGuest} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-4xl mx-auto">
          {/* Page Header */}
          <div className="mb-8">
            <BackButton to="/dashboard" />
            <div className="text-center mt-4">
              <h1 className="text-3xl font-bold text-white mb-2">{t('healthHistory')}</h1>
              <p className="text-white/80">{t('yourPastAssessments')}</p>
            </div>
          </div>

          {/* Search */}
          <div className="mb-8">
            <div className="relative">
              <Search className={`absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 ${
                theme === 'dark' ? 'text-gray-400' : 'text-gray-400'
              }`} />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder={t('searchYourAssessment')}
                className={`w-full pl-12 pr-4 py-4 rounded-xl border-0 focus:ring-2 focus:outline-none shadow-lg transition-colors ${
                  (theme === 'noongar-dark' || theme === 'noongar-light')
                    ? theme === 'noongar-dark'
                      ? 'bg-gradient-to-r from-orange-500/10 to-orange-600/10 text-white placeholder-gray-400 border border-white/10 backdrop-blur-sm focus:ring-orange-500'
                      : 'bg-white text-orange-900 placeholder-orange-500 focus:ring-orange-500'
                    : theme === 'dark' 
                    ? 'bg-gradient-to-r from-blue-500/10 to-purple-600/10 text-black placeholder-gray-500 border border-white/10 backdrop-blur-sm focus:ring-blue-500' 
                    : 'bg-white text-gray-700 placeholder-gray-500 focus:ring-blue-500'
                }`}
              />
            </div>
          </div>

          {/* Start New Chat Button */}
          <div className="mb-8">
            <button
              onClick={handleStartNewChat}
              className={`w-full text-white py-4 px-6 rounded-xl font-semibold text-lg transition-all transform hover:scale-105 shadow-lg flex items-center justify-center ${
                (theme === 'noongar-dark' || theme === 'noongar-light')
                  ? 'bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700'
                  : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700'
              }`}
            >
              <MessageCircle className="w-6 h-6 mr-3" />
              Start New Consultation
              <ArrowRight className="w-6 h-6 ml-3" />
            </button>
          </div>

          {/* Loading */}
          {isLoading && (
            <div className="text-center py-8">
              <div className={`w-8 h-8 border-4 rounded-full animate-spin mx-auto ${
                (theme === 'noongar-dark' || theme === 'noongar-light')
                  ? theme === 'noongar-dark'
                    ? 'border-white/20 border-t-orange-400'
                    : 'border-orange-200 border-t-orange-600'
                  : theme === 'dark' 
                  ? 'border-white/20 border-t-white/60' 
                  : 'border-blue-200 border-t-blue-600'
              }`}></div>
              <p className={`mt-2 ${
                theme === 'dark' ? 'text-white' : 'text-gray-800'
              }`}>{t('loadingYourHistory')}</p>
            </div>
          )}

          {/* Chat Sessions List */}
          <div className="space-y-4">
            {filteredSessions.map((session) => (
              <div key={session.id} className={`rounded-xl p-6 shadow-lg hover:shadow-xl transition-all ${
                (theme === 'noongar-dark' || theme === 'noongar-light')
                  ? theme === 'noongar-dark'
                    ? 'bg-gradient-to-r from-orange-500/20 to-orange-600/20 backdrop-blur-sm border border-white/10 hover:from-orange-500/30 hover:to-orange-600/30'
                    : 'bg-white hover:bg-orange-50'
                  : theme === 'dark' 
                  ? 'bg-gradient-to-r from-blue-500/20 to-purple-600/20 backdrop-blur-sm border border-white/10 hover:from-blue-500/30 hover:to-purple-600/30' 
                  : 'bg-white hover:bg-gray-50'
              }`}>
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center flex-1 cursor-pointer" onClick={() => handleViewChat(session.id)}>
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center mr-4 ${
                      (theme === 'noongar-dark' || theme === 'noongar-light')
                        ? 'bg-gradient-to-r from-orange-500 to-orange-600'
                        : 'bg-gradient-to-r from-blue-500 to-purple-600'
                    }`}>
                      <Bot className="w-6 h-6 text-white" />
                    </div>
                    <div className="flex-1">
                      {editingId === session.id ? (
                        <div className="flex items-center space-x-2">
                          <input
                            type="text"
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            className={`px-2 py-1 rounded text-lg font-bold border-0 focus:ring-2 focus:outline-none ${
                              (theme === 'noongar-dark' || theme === 'noongar-light')
                                ? theme === 'noongar-dark'
                                  ? 'bg-slate-600 text-white focus:ring-orange-500'
                                  : 'bg-orange-100 text-orange-900 focus:ring-orange-500'
                                : theme === 'dark' 
                                ? 'bg-slate-600 text-white focus:ring-blue-500' 
                                : 'bg-gray-100 text-gray-800 focus:ring-blue-500'
                            }`}
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') {
                                handleSaveTitle(session.id);
                              } else if (e.key === 'Escape') {
                                handleCancelEdit();
                              }
                            }}
                          />
                          <button
                            onClick={() => handleSaveTitle(session.id)}
                            className="text-green-500 hover:text-green-400"
                            title="Save"
                          >
                            ✓
                          </button>
                          <button
                            onClick={handleCancelEdit}
                            className="text-red-500 hover:text-red-400"
                            title="Cancel"
                          >
                            ✕
                          </button>
                        </div>
                      ) : (
                        <h3 className={`font-bold text-lg mb-1 ${
                          theme === 'dark' ? 'text-white' : 'text-gray-800'
                        }`}>{session.title}</h3>
                      )}
                      <div className={`flex items-center text-sm ${
                        theme === 'dark' ? 'text-white/70' : 'text-gray-600'
                      }`}>
                        <Clock className="w-4 h-4 mr-1" />
                        {session.timestamp}
                        <span className="mx-2">•</span>
                        <MessageCircle className="w-4 h-4 mr-1" />
                        {session.messageCount} messages
                        {session.hasVoiceMessages && (
                          <>
                            <span className="mx-2">•</span>
                            <Mic className="w-4 h-4 mr-1" />
                            Voice messages
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <ChatOptions
                      onDelete={() => handleDeleteChat(session.id)}
                      onEdit={() => handleEditTitle(session.id, session.title)}
                    />
                    <ArrowRight className={`w-5 h-5 ${
                      theme === 'dark' ? 'text-white/60' : 'text-gray-400'
                    }`} />
                  </div>
                </div>

                <div className={`rounded-lg p-4 ${
                  theme === 'dark' 
                    ? 'bg-gradient-to-r from-blue-500/5 to-purple-600/5 border border-white/5 backdrop-blur-sm' 
                    : 'bg-gray-50'
                }`}>
                  <p className={`text-sm ${
                    theme === 'dark' ? 'text-white/80' : 'text-gray-700'
                  }`}>
                    <span className={`font-medium ${
                      theme === 'dark' ? 'text-white' : 'text-gray-800'
                    }`}>Last message:</span> {session.lastMessage}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Empty State */}
          {!isLoading && filteredSessions.length === 0 && (
            <div className="text-center py-12">
              <MessageCircle className={`w-16 h-16 mx-auto mb-4 ${
                theme === 'dark' ? 'text-white/40' : 'text-gray-400'
              }`} />
              <h3 className={`text-xl font-medium mb-2 ${
                theme === 'dark' ? 'text-white' : 'text-gray-800'
              }`}>
                {searchTerm ? t('noMatchingAssessments') : t('noAssessmentsYet')}
              </h3>
              <p className={`mb-6 ${
                theme === 'dark' ? 'text-white/80' : 'text-gray-600'
              }`}>
                {searchTerm 
                  ? t('tryAdjustingSearch')
                  : t('startFirstAssessment')
                }
              </p>
              {!searchTerm && (
                <button
                  onClick={handleStartNewChat}
                  className={`text-white px-8 py-3 rounded-xl font-semibold transition-all transform hover:scale-105 shadow-lg ${
                    (theme === 'noongar-dark' || theme === 'noongar-light')
                      ? 'bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700'
                      : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700'
                  }`}
                >
                  {t('startAssessment')}
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Confirmation Dialog */}
      <ConfirmationDialog
        isOpen={showDeleteDialog}
        onClose={cancelDelete}
        onConfirm={confirmDelete}
        title="Delete Chat Session"
        message="Are you sure you want to delete this chat session? This action cannot be undone."
        confirmText="Delete"
        cancelText="Cancel"
        type="danger"
      />
    </Layout>
  );
};

export default HistoryPage;