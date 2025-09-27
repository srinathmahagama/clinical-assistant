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

interface LoginPageProps {
  onNavigate: (page: string) => void;
  onLogin: (user: User) => void;
}

const LoginPage: React.FC<LoginPageProps> = ({ onNavigate, onLogin }) => {
  const navigate = useNavigate();
  const { t } = useLanguage();
  const [email, setEmail] = useState('nimash@example.com');
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
        showLanguage={false}
        title="CareMate"
      />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-2xl mx-auto pt-8">
          <BackButton to="/" />
          
          <div className="flex items-center justify-center">
            <div className="bg-white rounded-3xl p-8 w-full max-w-md shadow-2xl">
          <div className="text-center mb-8">
            <h1 className="text-2xl font-bold text-gray-800 mb-2">{t('welcomeBack')}</h1>
            <p className="text-gray-600 mb-6">{t('signInToAccess')}</p>
            
            <div className="mb-6">
              <Logo size="lg" />
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg text-sm">
                {error}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t('emailAddress')}
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="example@gmail.com"
                className="w-full px-4 py-3 bg-gray-100 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                {t('password')}
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="enter your password"
                  className="w-full px-4 py-3 bg-gray-100 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors pr-12"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gray-600 hover:bg-[#183172] text-white py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? t('signingIn') : t('signIn')}
            </button>

            <div className="text-center">
              <button
                type="button"
                onClick={() => navigate('/')}
                className="text-[#183172] hover:text-[#183172]/80 text-sm underline"
              >
                {t('forgetPassword')}
              </button>
            </div>

            <div className="text-center text-sm text-gray-600">
              {t('createNewAccount')}{' '}
              <button
                type="button"
                onClick={() => navigate('/signup')}
                className="text-[#183172] hover:text-[#183172]/80 underline"
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