import React, { useState, useRef, useEffect } from 'react';
import { MoreVertical, Trash2, Edit3 } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface ChatOptionsProps {
  onDelete: () => void;
  onEdit: () => void;
}

const ChatOptions: React.FC<ChatOptionsProps> = ({ onDelete, onEdit }) => {
  const { theme } = useTheme();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleDelete = () => {
    onDelete();
    setIsOpen(false);
  };

  const handleEdit = () => {
    onEdit();
    setIsOpen(false);
  };

  return (
    <div className="relative" ref={menuRef}>
      {/* 3-dots button */}
      <button
        onClick={(e) => {
          e.stopPropagation(); // Prevent triggering the parent click
          setIsOpen(!isOpen);
        }}
        className={`p-1 rounded-full transition-colors ${
          theme === 'dark' 
            ? 'text-slate-400 hover:text-slate-300 hover:bg-slate-600' 
            : 'text-gray-500 hover:text-gray-700 hover:bg-gray-200'
        }`}
        title="Chat options"
      >
        <MoreVertical className="w-4 h-4" />
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div className={`absolute right-0 top-8 z-50 min-w-[140px] rounded-lg shadow-lg border transition-colors duration-300 ${
          theme === 'dark' 
            ? 'bg-slate-700 border-slate-600' 
            : 'bg-white border-gray-200'
        }`}>
          <div className="py-1">
            <button
              onClick={handleEdit}
              className={`w-full px-3 py-2 text-left text-sm flex items-center space-x-2 transition-colors ${
                theme === 'dark' 
                  ? 'text-blue-400 hover:bg-slate-600 hover:text-blue-300' 
                  : 'text-blue-600 hover:bg-gray-100 hover:text-blue-700'
              }`}
            >
              <Edit3 className="w-4 h-4" />
              <span>Edit Title</span>
            </button>
            <button
              onClick={handleDelete}
              className={`w-full px-3 py-2 text-left text-sm flex items-center space-x-2 transition-colors ${
                theme === 'dark' 
                  ? 'text-red-400 hover:bg-slate-600 hover:text-red-300' 
                  : 'text-red-600 hover:bg-gray-100 hover:text-red-700'
              }`}
            >
              <Trash2 className="w-4 h-4" />
              <span>Delete</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatOptions;
