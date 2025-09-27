import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';

interface BackButtonProps {
  onClick?: () => void;
  text?: string;
  to?: string;
}

const BackButton: React.FC<BackButtonProps> = ({ onClick, text = 'Back', to }) => {
  const navigate = useNavigate();

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
      className="flex items-center text-white hover:text-white/80 transition-colors mb-4"
    >
      <ArrowLeft className="w-5 h-5 mr-2" />
      <span className="text-lg">{text}</span>
    </button>
  );
};

export default BackButton;