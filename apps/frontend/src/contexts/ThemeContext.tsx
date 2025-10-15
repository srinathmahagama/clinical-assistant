import React, { createContext, useContext, useState, useEffect } from 'react';

type Theme = 'light' | 'dark' | 'noongar-light' | 'noongar-dark';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
  getEffectiveTheme: () => 'light' | 'dark';
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setThemeState] = useState<Theme>(() => {
    // Check localStorage first, then system preference
    const savedTheme = localStorage.getItem('caremate-theme') as Theme;
    if (savedTheme) {
      return savedTheme;
    }
    
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    
    return 'light';
  });

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
    localStorage.setItem('caremate-theme', newTheme);
  };

  const toggleTheme = () => {
    if (theme === 'light') {
      setTheme('noongar-dark');
    } else if (theme === 'dark') {
      setTheme('noongar-light');
    } else if (theme === 'noongar-light') {
      setTheme('noongar-dark');
    } else if (theme === 'noongar-dark') {
      setTheme('noongar-light');
    }
  };

  const getEffectiveTheme = (): 'light' | 'dark' => {
    if (theme === 'noongar-light') {
      return 'light';
    } else if (theme === 'noongar-dark') {
      return 'dark';
    }
    return theme;
  };

  useEffect(() => {
    // Apply theme to document
    document.documentElement.setAttribute('data-theme', theme);
    
    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      if (theme === 'noongar-light' || theme === 'noongar-dark') {
        metaThemeColor.setAttribute('content', '#cd853f'); // Authentic Noongar theme color
      } else {
        metaThemeColor.setAttribute('content', theme === 'dark' ? '#1a1a1a' : '#ffffff');
      }
    }
  }, [theme]);

  // Listen for language changes and localStorage changes
  useEffect(() => {
    const handleLanguageChange = (event: CustomEvent) => {
      const newLanguage = event.detail.language;
      
      // Immediately update theme based on language
      // if (newLanguage === 'noongar') {
        // Default to noongar-dark, but preserve existing noongar theme if it exists
        const currentTheme = localStorage.getItem('caremate-theme');
        if (currentTheme === 'noongar-light') {
          setThemeState('noongar-light');
        } else {
          setThemeState('noongar-dark');
        }
      // } else if (newLanguage === 'en') {
      //   setThemeState('dark');
      // }
    };

    // Check on mount
    const currentLanguage = localStorage.getItem('caremate-language');
    // if (currentLanguage === 'noongar') {
      const currentTheme = localStorage.getItem('caremate-theme');
      if (currentTheme === 'noongar-light' || currentTheme === 'noongar-dark') {
        setThemeState(currentTheme as Theme);
      } else {
        setThemeState('noongar-dark');
      }
    // }

    // Listen for custom language change events
    window.addEventListener('languageChanged', handleLanguageChange as EventListener);
    
    return () => {
      window.removeEventListener('languageChanged', handleLanguageChange as EventListener);
    };
  }, []);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      // Only update if user hasn't manually set a preference
      if (!localStorage.getItem('caremate-theme')) {
        setThemeState(e.matches ? 'dark' : 'light');
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  const value = {
    theme,
    toggleTheme,
    setTheme,
    getEffectiveTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};
