import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { translations, Language, TranslationKey } from '../translations/translations';

interface LanguageContextType {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: string) => string;
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
    
    // Trigger theme change based on language
    if (newLanguage === 'noongar') {
      // Default to noongar-dark when switching to Noongar
      localStorage.setItem('caremate-theme', 'noongar-dark');
      // Dispatch a custom event to notify theme context
      window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: newLanguage } }));
    } else if (newLanguage === 'en') {
      // Only change theme if it was noongar
      const currentTheme = localStorage.getItem('caremate-theme');
      if (currentTheme === 'noongar-light' || currentTheme === 'noongar-dark') {
        localStorage.setItem('caremate-theme', 'dark');
        window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: newLanguage } }));
      }
    }
  };

  // Translation function with support for nested keys
  const t = (key: string): string => {
    const keys = key.split('.');
    let value: any = translations[language];
    
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        // Fallback to English
        value = translations.en;
        for (const fallbackKey of keys) {
          if (value && typeof value === 'object' && fallbackKey in value) {
            value = value[fallbackKey];
          } else {
            return key; // Return the key if not found
          }
        }
        break;
      }
    }
    
    return typeof value === 'string' ? value : key;
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
