import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface BackButtonProps {
  onClick?: () => void;
  to?: string;
}

const BackButton: React.FC<BackButtonProps> = ({ onClick, to }) => {
  const navigate = useNavigate();
  const { theme } = useTheme();

  const handleClick = () => {
    if (onClick) {
      onClick();
    } else if (to) {
      navigate(to);
    } else {
      navigate(-1); // Go back in history
    }
  };

  return (
    <button
      onClick={handleClick}
      className={`fixed top-28 left-6 z-50 w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110 shadow-lg ${
        theme === 'noongar-dark'
          ? 'bg-white/10 backdrop-blur-sm border border-white/20 text-white hover:bg-white/20'
          : theme === 'noongar-light'
          ? 'bg-black/10 backdrop-blur-sm border border-black/20 text-black hover:bg-black/20'
          : theme === 'dark' 
          ? 'bg-white/10 backdrop-blur-sm border border-white/20 text-white hover:bg-white/20' 
          : 'bg-black/10 backdrop-blur-sm border border-black/20 text-black hover:bg-black/20'
      }`}
      title="Go back"
    >
      <ArrowLeft className="w-6 h-6" />
    </button>
  );
};

export default BackButton;