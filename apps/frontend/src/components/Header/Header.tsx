import React from 'react';
import { LogOut, User } from 'lucide-react';
import LanguageDropdown from '../LanguageDropdown/LanguageDropdown';
import Logo from '../Logo/Logo';

interface HeaderProps {
  onLogout?: () => void;
  showLanguage?: boolean;
  title?: string;
  showSignIn?: boolean;
  onSignIn?: () => void;
}

const Header: React.FC<HeaderProps> = ({ onLogout, showLanguage = true, title = "CareMate", showSignIn = false, onSignIn }) => {
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
            {/* Language Button */}
            {showLanguage && (
              <LanguageDropdown className="text-white/80 hover:text-white transition-colors duration-200" />
            )}
            
            {/* Sign In or Logout Button */}
            {showSignIn ? (
              <button
                onClick={onSignIn}
                className="flex items-center space-x-2 text-white hover:text-white/80 transition-colors duration-200 bg-white/10 hover:bg-white/20 px-3 py-2 rounded-lg backdrop-blur-sm"
              >
                <span className="text-sm font-medium">Sign In</span>
                <span className="text-sm">→</span>
              </button>
            ) : (
              onLogout && (
                <button
                  onClick={onLogout}
                  className="flex items-center space-x-2 text-white hover:text-white/80 transition-colors duration-200 bg-white/10 hover:bg-white/20 px-3 py-2 rounded-lg backdrop-blur-sm"
                >
                  <User className="w-4 h-4" />
                  <span className="text-sm font-medium">Sign Out</span>
                </button>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Header;
