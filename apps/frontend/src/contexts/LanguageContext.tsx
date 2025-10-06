import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { translations, Language, TranslationKey } from '../translations/translations';

interface LanguageContextType {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: TranslationKey) => string;
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

  // Load saved language on mount
  useEffect(() => {
    const savedLanguage = getSavedLanguage();
    setLanguageState(savedLanguage);
  }, []);

  const value: LanguageContextType = {
    language,
    setLanguage,
    t,
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
