import React, { useState, useEffect, useRef } from 'react';
import { Plus, MessageSquare, Trash2, Edit2, Check, X, MoreVertical } from 'lucide-react';
import { ChatSession } from '../../types';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';

interface ChatSidebarProps {
  sessions: ChatSession[];
  currentSessionId: string | null;
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onRenameSession: (sessionId: string, newTitle: string) => void;
}

const ChatSidebar: React.FC<ChatSidebarProps> = ({
  sessions,
  currentSessionId,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onRenameSession
}) => {
  const { t } = useLanguage();
  const { theme } = useTheme();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [openDropdownId, setOpenDropdownId] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setOpenDropdownId(null);
      }
    };

    if (openDropdownId) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [openDropdownId]);

  const handleStartEdit = (session: ChatSession) => {
    setEditingId(session.id);
    setEditTitle(session.title);
  };

  const handleSaveEdit = () => {
    if (editingId && editTitle.trim()) {
      onRenameSession(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditTitle('');
  };

  const handleToggleDropdown = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenDropdownId(openDropdownId === sessionId ? null : sessionId);
  };

  const handleCloseDropdown = () => {
    setOpenDropdownId(null);
  };

  const handleEditClick = (session: ChatSession, e: React.MouseEvent) => {
    e.stopPropagation();
    handleStartEdit(session);
    handleCloseDropdown();
  };

  const handleDeleteClick = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    onDeleteSession(sessionId);
    handleCloseDropdown();
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) {
      return 'Today';
    } else if (diffDays === 2) {
      return 'Yesterday';
    } else if (diffDays <= 7) {
      return `${diffDays - 1} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  return (
    <div className={`w-80 flex flex-col h-full transition-colors duration-300 ${
      theme === 'noongar-dark' 
        ? 'bg-slate-800 text-white' 
        : theme === 'noongar-light'
        ? 'bg-white text-orange-800'
        : theme === 'dark' 
        ? 'bg-slate-800 text-white' 
        : 'bg-white text-gray-800'
    }`}>
      {/* Header */}
      <div className={`p-4 border-b transition-colors duration-300 ${
        theme === 'noongar-dark' 
          ? 'border-slate-600' 
          : theme === 'noongar-light'
          ? 'border-orange-200'
          : theme === 'dark' 
          ? 'border-slate-600' 
          : 'border-gray-200'
      }`}>
        <button
          onClick={onNewChat}
          className={`w-full flex items-center justify-center space-x-2 px-4 py-3 rounded-lg transition-colors ${
            theme === 'noongar-dark' 
              ? 'bg-slate-700 hover:bg-slate-600 text-white'
              : theme === 'noongar-light'
              ? 'bg-orange-600 hover:bg-orange-700 text-white'
              : theme === 'dark' 
              ? 'bg-slate-700 hover:bg-slate-600 text-white' 
              : 'bg-blue-600 hover:bg-blue-700 text-white'
          }`}
        >
          <Plus className="w-5 h-5" />
          <span>{t('newChat')}</span>
        </button>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-2">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`group relative p-3 rounded-lg cursor-pointer transition-colors ${
                session.id === currentSessionId
                  ? theme === 'noongar-dark' 
                    ? 'bg-slate-600'
                    : theme === 'noongar-light'
                    ? 'bg-orange-200'
                    : theme === 'dark' 
                    ? 'bg-slate-600' 
                    : 'bg-blue-200'
                  : theme === 'noongar-dark'
                    ? 'bg-slate-700 hover:bg-slate-600'
                    : theme === 'noongar-light'
                    ? 'bg-orange-50 hover:bg-orange-100'
                    : theme === 'dark'
                    ? 'bg-slate-700 hover:bg-slate-600'
                    : 'bg-blue-50 hover:bg-blue-100'
              }`}
              onClick={() => onSelectSession(session.id)}
            >
              {editingId === session.id ? (
                <div className="flex items-center space-x-2">
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') handleSaveEdit();
                      if (e.key === 'Escape') handleCancelEdit();
                    }}
                    className={`flex-1 px-2 py-1 rounded text-sm focus:outline-none focus:ring-2 ${
                      theme === 'noongar-dark' 
                        ? 'bg-gray-600 text-white focus:ring-blue-500'
                        : theme === 'noongar-light'
                        ? 'bg-white text-orange-800 border border-orange-300 focus:ring-orange-500'
                        : theme === 'dark' 
                        ? 'bg-gray-600 text-white focus:ring-blue-500' 
                        : 'bg-white text-gray-800 border border-gray-300 focus:ring-blue-500'
                    }`}
                    autoFocus
                  />
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSaveEdit();
                    }}
                    className="p-1 text-green-400 hover:text-green-300"
                  >
                    <Check className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCancelEdit();
                    }}
                    className="p-1 text-red-400 hover:text-red-300"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <>
                  <div className="flex items-start space-x-3">
                    <MessageSquare className={`w-4 h-4 mt-0.5 flex-shrink-0 ${
                      theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                    }`} />
                    <div className="flex-1 min-w-0">
                      <h3 className={`text-sm font-medium truncate ${
                        theme === 'dark' ? 'text-white' : 'text-gray-800'
                      }`}>
                        {session.title}
                      </h3>
                      <p className={`text-xs mt-1 ${
                        theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
                      }`}>
                        {formatDate(session.updatedAt)}
                      </p>
                    </div>
                  </div>

                  {/* 3-Dots Menu */}
                  <div className="absolute right-2 top-2">
                    <div className="relative" ref={dropdownRef}>
                      <button
                        onClick={(e) => handleToggleDropdown(session.id, e)}
                        className={`p-1 rounded-full transition-colors ${
                          theme === 'dark' 
                            ? 'text-gray-400 hover:text-white hover:bg-slate-600' 
                            : 'text-gray-500 hover:text-gray-700 hover:bg-gray-200'
                        }`}
                        title="More options"
                      >
                        <MoreVertical className="w-4 h-4" />
                      </button>
                      
                      {/* Dropdown Menu */}
                      {openDropdownId === session.id && (
                        <div className={`absolute right-0 top-8 z-50 min-w-[120px] rounded-lg shadow-lg border transition-colors ${
                          theme === 'dark' 
                            ? 'bg-slate-700 border-slate-600' 
                            : 'bg-white border-gray-200'
                        }`}>
                          <div className="py-1">
                            <button
                              onClick={(e) => handleEditClick(session, e)}
                              className={`w-full px-3 py-2 text-left text-sm transition-colors flex items-center space-x-2 ${
                                theme === 'dark' 
                                  ? 'text-gray-300 hover:bg-slate-600 hover:text-white' 
                                  : 'text-gray-700 hover:bg-gray-100'
                              }`}
                            >
                              <Edit2 className="w-3 h-3" />
                              <span>Rename</span>
                            </button>
                            <button
                              onClick={(e) => handleDeleteClick(session.id, e)}
                              className={`w-full px-3 py-2 text-left text-sm transition-colors flex items-center space-x-2 ${
                                theme === 'dark' 
                                  ? 'text-gray-300 hover:bg-slate-600 hover:text-red-400' 
                                  : 'text-gray-700 hover:bg-gray-100 hover:text-red-600'
                              }`}
                            >
                              <Trash2 className="w-3 h-3" />
                              <span>Delete</span>
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>

        {/* Empty State */}
        {sessions.length === 0 && (
          <div className="text-center py-8">
            <MessageSquare className={`w-12 h-12 mx-auto mb-4 ${
              theme === 'dark' ? 'text-gray-600' : 'text-gray-400'
            }`} />
            <p className={`text-sm ${
              theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
            }`}>No chat history yet</p>
            <p className={`text-xs mt-1 ${
              theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
            }`}>Start a new conversation</p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className={`p-4 border-t ${
        theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
      }`}>
        <div className={`text-xs text-center ${
          theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
        }`}>
          CareMate Health Assistant
        </div>
      </div>
    </div>
  );
};

export default ChatSidebar;
