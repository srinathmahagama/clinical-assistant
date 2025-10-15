import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import Logo from '../components/Logo/Logo';
import LanguageDropdown from '../components/LanguageDropdown/LanguageDropdown';
import { authService } from '../services/api';
import { User } from '../types';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';

interface LoginPageProps {
  onNavigate: (page: string) => void;
  onLogin: (user: User) => void;
}

const LoginPage: React.FC<LoginPageProps> = ({ onNavigate, onLogin }) => {
  const navigate = useNavigate();
  const { t } = useLanguage();
  const { theme } = useTheme();
  const [email, setEmail] = useState('test@example.com');
  const [password, setPassword] = useState('password123');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await authService.login(email, password);
      if (response.success && response.data) {
        onLogin(response.data);
        navigate('/dashboard');
      } else {
        setError(response.message || 'Login failed');
      }
    } catch (err) {
      setError('An error occurred during login');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout backgroundType="auth">
      <Header 
        showLanguage={true}
        title="CareMate"
      />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-2xl mx-auto pt-8">
          <BackButton to="/" />
          
          <div className="flex items-center justify-center">
            <div className={`rounded-3xl p-8 w-full max-w-md shadow-2xl transition-colors duration-300 ${
              theme === 'noongar-dark' 
                ? 'bg-slate-800' 
                : theme === 'noongar-light'
                ? 'bg-white'
                : theme === 'dark' 
                ? 'bg-slate-800' 
                : 'bg-white'
            }`}>
          <div className="text-center mb-8">
            <h1 className={`text-2xl font-bold mb-2 ${
              (theme === 'noongar-dark' || theme === 'noongar-light')
                ? theme === 'noongar-dark'
                  ? 'text-white'
                  : 'text-gray-800'
                : theme === 'dark' 
                ? 'text-white' 
                : 'text-gray-800'
            }`}>{t('welcomeBack')}</h1>
            <p className={`mb-6 ${
              (theme === 'noongar-dark' || theme === 'noongar-light')
                ? theme === 'noongar-dark'
                  ? 'text-slate-300'
                  : 'text-gray-600'
                : theme === 'dark' 
                ? 'text-slate-300' 
                : 'text-gray-600'
            }`}>{t('signInToAccess')}</p>
            
            <div className="mb-6">
              <Logo size="lg" />
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className={`px-4 py-3 rounded-lg text-sm transition-colors duration-300 ${
                (theme === 'noongar-dark' || theme === 'noongar-light')
                  ? theme === 'noongar-dark'
                    ? 'bg-red-900/50 border border-red-700 text-red-300'
                    : 'bg-red-50 border border-red-200 text-red-600'
                  : theme === 'dark' 
                  ? 'bg-red-900/50 border border-red-700 text-red-300' 
                  : 'bg-red-50 border border-red-200 text-red-600'
              }`}>
                {error}
              </div>
            )}

            <div>
              <label className={`block text-sm font-medium mb-2 ${
                (theme === 'noongar-dark' || theme === 'noongar-light')
                  ? theme === 'noongar-dark'
                    ? 'text-slate-300'
                    : 'text-gray-700'
                  : theme === 'dark' 
                  ? 'text-slate-300' 
                  : 'text-gray-700'
              }`}>
                {t('emailAddress')}
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="example@gmail.com"
                className={`w-full px-4 py-3 border-0 rounded-lg focus:ring-2 transition-colors ${
                  (theme === 'noongar-dark' || theme === 'noongar-light')
                    ? theme === 'noongar-dark'
                      ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600 focus:ring-orange-500'
                      : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white focus:ring-orange-500'
                    : theme === 'dark'
                    ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600 focus:ring-blue-500'
                    : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white focus:ring-blue-500'
                }`}
                required
              />
            </div>

            <div>
              <label className={`block text-sm font-medium mb-2 ${
                (theme === 'noongar-dark' || theme === 'noongar-light')
                  ? theme === 'noongar-dark'
                    ? 'text-slate-300'
                    : 'text-gray-700'
                  : theme === 'dark' 
                  ? 'text-slate-300' 
                  : 'text-gray-700'
              }`}>
                {t('password')}
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="enter your password"
                  className={`w-full px-4 py-3 border-0 rounded-lg focus:ring-2 transition-colors pr-12 ${
                    (theme === 'noongar-dark' || theme === 'noongar-light')
                      ? theme === 'noongar-dark'
                        ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600 focus:ring-orange-500'
                        : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white focus:ring-orange-500'
                      : theme === 'dark'
                      ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600 focus:ring-blue-500'
                      : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white focus:ring-blue-500'
                  }`}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 transition-colors ${
                    (theme === 'noongar-dark' || theme === 'noongar-light')
                      ? theme === 'noongar-dark'
                        ? 'text-slate-400 hover:text-slate-300'
                        : 'text-gray-500 hover:text-gray-700'
                      : theme === 'dark' 
                      ? 'text-slate-400 hover:text-slate-300' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className={`w-full py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                theme === 'noongar-light' || theme === 'noongar-dark'
                  ? 'bg-orange-600 hover:bg-orange-700 text-white'
                  : 'bg-gray-600 hover:bg-[#183172] text-white'
              }`}
            >
              {isLoading ? t('signingIn') : t('signIn')}
            </button>

            <div className="text-center">
              <button
                type="button"
                onClick={() => navigate('/')}
                className={`text-sm underline transition-colors ${
                  (theme === 'noongar-dark' || theme === 'noongar-light')
                    ? theme === 'noongar-dark'
                      ? 'text-orange-400 hover:text-orange-300'
                      : 'text-orange-600 hover:text-orange-500'
                    : theme === 'dark' 
                    ? 'text-blue-400 hover:text-blue-300' 
                    : 'text-[#183172] hover:text-[#183172]/80'
                }`}
              >
                {t('forgetPassword')}
              </button>
            </div>

            <div className={`text-center text-sm ${
              (theme === 'noongar-dark' || theme === 'noongar-light')
                ? theme === 'noongar-dark'
                  ? 'text-slate-300'
                  : 'text-gray-600'
                : theme === 'dark' 
                ? 'text-slate-300' 
                : 'text-gray-600'
            }`}>
              {t('createNewAccount')}{' '}
              <button
                type="button"
                onClick={() => navigate('/signup')}
                className={`underline transition-colors ${
                  (theme === 'noongar-dark' || theme === 'noongar-light')
                    ? theme === 'noongar-dark'
                      ? 'text-orange-400 hover:text-orange-300'
                      : 'text-orange-600 hover:text-orange-500'
                    : theme === 'dark' 
                    ? 'text-blue-400 hover:text-blue-300' 
                    : 'text-[#183172] hover:text-[#183172]/80'
                }`}
              >
                {t('signUp')}
              </button>
            </div>
          </form>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default LoginPage;