import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { translations, Language, TranslationKey } from '../translations/translations';

interface LanguageContextType {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: TranslationKey) => string;
  // Chat session language management
  getChatSessionLanguage: (sessionId?: string) => Language;
  setChatSessionLanguage: (sessionId: string, language: Language) => void;
  getCurrentChatLanguage: () => Language;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

interface LanguageProviderProps {
  children: ReactNode;
}

export const LanguageProvider: React.FC<LanguageProviderProps> = ({ children }) => {
  // Get saved language from localStorage or default to English
  const getSavedLanguage = (): Language => {
    const saved = localStorage.getItem('caremate-language');
    if (saved && saved in translations) {
      return saved as Language;
    }
    return 'en';
  };

  const [language, setLanguageState] = useState<Language>(getSavedLanguage);

  // Save language to localStorage whenever it changes
  const setLanguage = (newLanguage: Language) => {
    setLanguageState(newLanguage);
    localStorage.setItem('caremate-language', newLanguage);
  };

  // Translation function
  const t = (key: TranslationKey): string => {
    return translations[language][key] || translations.en[key] || key;
  };

  // Chat session language management
  const getChatSessionLanguage = (sessionId?: string): Language => {
    if (!sessionId) return language;
    
    const sessionLanguage = localStorage.getItem(`caremate-chat-language-${sessionId}`);
    if (sessionLanguage && sessionLanguage in translations) {
      return sessionLanguage as Language;
    }
    return language;
  };

  const setChatSessionLanguage = (sessionId: string, newLanguage: Language) => {
    localStorage.setItem(`caremate-chat-language-${sessionId}`, newLanguage);
  };

  const getCurrentChatLanguage = (): Language => {
    // This will be used by components to get the current chat session language
    // For now, return the global language, but this can be enhanced to track current session
    return language;
  };

  // Load saved language on mount
  useEffect(() => {
    const savedLanguage = getSavedLanguage();
    setLanguageState(savedLanguage);
  }, []);

  const value: LanguageContextType = {
    language,
    setLanguage,
    t,
    getChatSessionLanguage,
    setChatSessionLanguage,
    getCurrentChatLanguage,
  };

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = (): LanguageContextType => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};
