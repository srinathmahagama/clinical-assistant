import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  // Determine if we're in a light theme
  const isLightTheme = theme === 'light' || theme === 'noongar-light';

  return (
    <button
      onClick={toggleTheme}
      className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-white transition-all duration-200 backdrop-blur-sm"
      title={`Switch to ${isLightTheme ? 'dark' : 'light'} mode`}
    >
      {isLightTheme ? (
        <Moon className="w-5 h-5" />
      ) : (
        <Sun className="w-5 h-5" />
      )}
    </button>
  );
};

export default ThemeToggle;
