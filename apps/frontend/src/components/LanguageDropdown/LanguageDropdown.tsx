import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';

interface LanguageDropdownProps {
  className?: string;
}

const LanguageDropdown: React.FC<LanguageDropdownProps> = ({ className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const { language, setLanguage, t } = useLanguage();

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
        className="bg-[#183172] text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-[#183172]/80 transition-colors flex items-center space-x-2"
      >
        <span>{currentLanguage.name}</span>
        <ChevronDown className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
          {languages.map((language) => (
            <button
              key={language.code}
              onClick={() => handleLanguageSelect(language.code)}
              className="w-full px-4 py-3 text-left hover:bg-blue-100 hover:text-blue-800 transition-colors flex items-center space-x-3 first:rounded-t-lg last:rounded-b-lg"
            >
              <span className="text-lg">{language.flag}</span>
              <span className="text-gray-700 font-medium">{language.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default LanguageDropdown;
