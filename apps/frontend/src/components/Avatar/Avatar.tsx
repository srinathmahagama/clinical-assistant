import React from 'react';
import { User, Bot } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface AvatarProps {
  type: 'user' | 'assistant';
  userName?: string;
  isGuest?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const Avatar: React.FC<AvatarProps> = ({ 
  type, 
  userName, 
  isGuest = false, 
  size = 'md' 
}) => {
  const { theme } = useTheme();
  
  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-8 h-8 text-sm',
    lg: 'w-12 h-12 text-lg'
  };

  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(word => word.charAt(0))
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const getDisplayName = () => {
    if (type === 'assistant') return 'AI';
    if (isGuest) return 'GU';
    if (userName) return getInitials(userName);
    return 'U';
  };

  const getBackgroundColor = () => {
    if (type === 'assistant') {
      return theme === 'dark' 
        ? 'bg-purple-600' 
        : 'bg-purple-500';
    }
    
    if (isGuest) {
      return theme === 'dark' 
        ? 'bg-gray-600' 
        : 'bg-gray-500';
    }
    
    return theme === 'dark' 
      ? 'bg-blue-600' 
      : 'bg-blue-500';
  };

  return (
    <div className={`${sizeClasses[size]} ${getBackgroundColor()} rounded-full flex items-center justify-center text-white font-semibold flex-shrink-0`}>
      {type === 'assistant' ? (
        <Bot className={`${size === 'sm' ? 'w-3 h-3' : size === 'md' ? 'w-4 h-4' : 'w-6 h-6'}`} />
      ) : (
        <span className="font-bold">{getDisplayName()}</span>
      )}
    </div>
  );
};

export default Avatar;
