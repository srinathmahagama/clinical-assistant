import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';

interface LanguageDropdownProps {
  className?: string;
}

const LanguageDropdown: React.FC<LanguageDropdownProps> = ({ className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const { language, setLanguage, t } = useLanguage();
  const { theme } = useTheme();

  const languages = [
    { code: 'en', name: t('english'), flag: '🇺🇸' },
    { code: 'noongar', name: t('noongar'), flag: '🇦🇺' }
  ];

  const currentLanguage = languages.find(lang => lang.code === language) || languages[0];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleLanguageSelect = (languageCode: string) => {
    setLanguage(languageCode as any);
    setIsOpen(false);
  };

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-2 ${
          theme === 'dark' 
            ? 'bg-white/10 text-white hover:bg-white/20 backdrop-blur-sm border border-white/20' 
            : 'bg-white/10 text-white hover:bg-white/20 backdrop-blur-sm border border-white/20'
        }`}
      >
        <span>{currentLanguage.name}</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className={`absolute top-full right-0 mt-2 w-48 rounded-lg shadow-lg border z-50 ${
          theme === 'dark' 
            ? 'bg-slate-800 border-slate-700' 
            : 'bg-white border-gray-200'
        }`}>
          {languages.map((language) => (
            <button
              key={language.code}
              onClick={() => handleLanguageSelect(language.code)}
              className={`w-full px-4 py-3 text-left transition-colors flex items-center space-x-3 first:rounded-t-lg last:rounded-b-lg ${
                theme === 'dark' 
                  ? 'hover:bg-slate-700 hover:text-white text-slate-300' 
                  : 'hover:bg-blue-100 hover:text-blue-800 text-gray-700'
              }`}
            >
              <span className="text-lg">{language.flag}</span>
              <span className={`font-medium ${
                theme === 'dark' ? 'text-slate-300' : 'text-gray-700'
              }`}>{language.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default LanguageDropdown;
