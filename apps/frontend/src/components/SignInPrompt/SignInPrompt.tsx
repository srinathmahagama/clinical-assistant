import React from 'react';
import { User, LogIn, UserPlus } from 'lucide-react';

interface SignInPromptProps {
  onSignIn: () => void;
  onSignUp: () => void;
  title?: string;
  message?: string;
}

const SignInPrompt: React.FC<SignInPromptProps> = ({
  onSignIn,
  onSignUp,
  title = "Sign in to view history",
  message = "Save your conversations and access your chat history anytime"
}) => {
  return (
    <div className="bg-white rounded-2xl p-8 shadow-lg text-center">
      <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
        <User className="w-8 h-8 text-white" />
      </div>
      
      <h2 className="text-2xl font-bold text-gray-800 mb-4">{title}</h2>
      <p className="text-gray-600 mb-8 max-w-md mx-auto">{message}</p>
      
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
          className="w-full bg-white border-2 border-blue-500 text-blue-600 py-3 px-6 rounded-xl font-semibold hover:bg-blue-50 transition-all transform hover:scale-105 flex items-center justify-center"
        >
          <UserPlus className="w-5 h-5 mr-2" />
          Create Account
        </button>
      </div>
      
      <p className="text-xs text-gray-500 mt-6">
        Sign in to access all features and save your chat history
      </p>
    </div>
  );
};

export default SignInPrompt;
