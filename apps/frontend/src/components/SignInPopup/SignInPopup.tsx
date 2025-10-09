import React from 'react';
import { X, User, LogIn, UserPlus } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface SignInPopupProps {
  isOpen: boolean;
  onClose: () => void;
  onSignIn: () => void;
  onSignUp: () => void;
  onContinueAsGuest: () => void;
}

const SignInPopup: React.FC<SignInPopupProps> = ({
  isOpen,
  onClose,
  onSignIn,
  onSignUp,
  onContinueAsGuest
}) => {
  const { theme } = useTheme();
  
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className={`rounded-2xl p-8 max-w-md w-full shadow-2xl transition-colors duration-300 ${
        theme === 'dark' ? 'bg-slate-800' : 'bg-white'
      }`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <User className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className={`text-2xl font-bold transition-colors duration-300 ${
                theme === 'dark' ? 'text-white' : 'text-gray-800'
              }`}>Welcome to CareMate</h2>
              <p className={`text-sm transition-colors duration-300 ${
                theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
              }`}>Your health assistant is ready</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className={`p-2 rounded-full transition-colors duration-300 ${
              theme === 'dark' 
                ? 'hover:bg-slate-700 text-slate-400 hover:text-slate-300' 
                : 'hover:bg-gray-100 text-gray-500'
            }`}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="space-y-4 mb-6">
          <p className={`text-center transition-colors duration-300 ${
            theme === 'dark' ? 'text-slate-300' : 'text-gray-700'
          }`}>
            Sign in to save your chat history and get personalized health guidance, 
            or continue as a guest to try our assistant.
          </p>
        </div>

        {/* Actions */}
        <div className="space-y-3">
          <button
            onClick={onSignIn}
            className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-6 rounded-xl font-semibold hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg flex items-center justify-center"
          >
            <LogIn className="w-5 h-5 mr-2" />
            Sign In
          </button>
          
          <button
            onClick={onSignUp}
            className={`w-full border-2 border-blue-500 py-3 px-6 rounded-xl font-semibold transition-all transform hover:scale-105 flex items-center justify-center ${
              theme === 'dark'
                ? 'bg-slate-800 text-blue-400 hover:bg-slate-700'
                : 'bg-white text-blue-600 hover:bg-blue-50'
            }`}
          >
            <UserPlus className="w-5 h-5 mr-2" />
            Create Account
          </button>
          
          <button
            onClick={onContinueAsGuest}
            className={`w-full py-3 px-6 rounded-xl font-medium transition-colors duration-300 ${
              theme === 'dark'
                ? 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Continue as Guest
          </button>
        </div>

        {/* Footer */}
        <div className={`mt-6 pt-4 border-t transition-colors duration-300 ${
          theme === 'dark' ? 'border-slate-600' : 'border-gray-200'
        }`}>
          <p className={`text-xs text-center transition-colors duration-300 ${
            theme === 'dark' ? 'text-slate-400' : 'text-gray-500'
          }`}>
            As a guest, your chat history won't be saved. Sign in to access all features.
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignInPopup;
