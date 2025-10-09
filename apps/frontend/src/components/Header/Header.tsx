import React, { useState, useRef, useEffect } from 'react';
import { LogOut, User, ChevronDown } from 'lucide-react';
import LanguageDropdown from '../LanguageDropdown/LanguageDropdown';
import Logo from '../Logo/Logo';
import ThemeToggle from '../ThemeToggle/ThemeToggle';
import { User as UserType } from '../../types';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';

interface HeaderProps {
  onLogout?: () => void;
  showLanguage?: boolean;
  title?: string;
  showSignIn?: boolean;
  onSignIn?: () => void;
  user?: UserType | null;
  isGuest?: boolean;
}

const Header: React.FC<HeaderProps> = ({ onLogout, showLanguage = true, title = "CareMate", showSignIn = false, onSignIn, user, isGuest }) => {
  const { t } = useLanguage();
  const { theme } = useTheme();
  const [isProfileDropdownOpen, setIsProfileDropdownOpen] = useState(false);
  const profileDropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (profileDropdownRef.current && !profileDropdownRef.current.contains(event.target as Node)) {
        setIsProfileDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Get display name
  const getDisplayName = () => {
    if (isGuest) return 'Guest';
    if (user?.firstName) return user.firstName;
    return 'User';
  };
  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-[#183172]/50 backdrop-blur-md border-b border-white/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo/Title */}
          <div className="flex items-center space-x-3">
            <Logo size="sm" />
            <h1 className="text-white text-xl font-semibold">{title}</h1>
          </div>

          {/* Right side buttons */}
          <div className="flex items-center space-x-4">
            {/* Theme Toggle */}
            <ThemeToggle />
            
            {/* Language Button */}
            {showLanguage && (
              <LanguageDropdown className="text-white/80 hover:text-white transition-colors duration-200" />
            )}
            
            {/* Profile Dropdown */}
            {showSignIn ? (
              <button
                onClick={onSignIn}
                className="flex items-center space-x-2 text-white hover:text-white/80 transition-colors duration-200 bg-white/10 hover:bg-white/20 px-3 py-2 rounded-lg backdrop-blur-sm"
              >
                <span className="text-sm font-medium">{t('signIn')}</span>
                <span className="text-sm">→</span>
              </button>
            ) : (
              <div className="relative" ref={profileDropdownRef}>
                <button
                  onClick={() => setIsProfileDropdownOpen(!isProfileDropdownOpen)}
                  className="flex items-center space-x-2 text-white hover:text-white/80 transition-colors duration-200 bg-white/10 hover:bg-white/20 px-3 py-2 rounded-lg backdrop-blur-sm"
                >
                  <User className="w-4 h-4" />
                  <span className="text-sm font-medium">{getDisplayName()}</span>
                  <ChevronDown className={`w-3 h-3 transition-transform ${isProfileDropdownOpen ? 'rotate-180' : ''}`} />
                </button>

                {isProfileDropdownOpen && (
                  <div className={`absolute top-full right-0 mt-2 w-48 rounded-lg shadow-lg border z-50 ${
                    theme === 'dark' 
                      ? 'bg-slate-800 border-slate-700' 
                      : 'bg-white border-gray-200'
                  }`}>
                    {/* User Info */}
                    <div className={`px-4 py-3 border-b ${
                      theme === 'dark' ? 'border-slate-700' : 'border-gray-100'
                    }`}>
                      <div className="flex items-center space-x-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          theme === 'dark' 
                            ? 'bg-blue-500/20' 
                            : 'bg-blue-100'
                        }`}>
                          <User className={`w-4 h-4 ${
                            theme === 'dark' ? 'text-blue-400' : 'text-blue-600'
                          }`} />
                        </div>
                        <div>
                          <p className={`text-sm font-medium ${
                            theme === 'dark' ? 'text-white' : 'text-gray-900'
                          }`}>{getDisplayName()}</p>
                          <p className={`text-xs ${
                            theme === 'dark' ? 'text-slate-400' : 'text-gray-500'
                          }`}>
                            {isGuest ? t('guestUser') : user?.email || 'User'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Sign Out Button */}
                    {onLogout && (
                      <button
                        onClick={() => {
                          onLogout();
                          setIsProfileDropdownOpen(false);
                        }}
                        className={`w-full px-4 py-3 text-left transition-colors flex items-center space-x-3 ${
                          theme === 'dark' 
                            ? 'hover:bg-red-500/20 hover:text-red-400 text-slate-300' 
                            : 'hover:bg-red-50 hover:text-red-600 text-gray-700'
                        }`}
                      >
                        <LogOut className="w-4 h-4" />
                        <span className="text-sm font-medium">{t('signOut')}</span>
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Header;
