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

interface SignUpPageProps {
  onNavigate: (page: string) => void;
  onLogin: (user: User) => void;
}

const SignUpPage: React.FC<SignUpPageProps> = ({ onNavigate, onLogin }) => {
  const navigate = useNavigate();
  const { t } = useLanguage();
  const { theme } = useTheme();
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState<{[key: string]: string}>({});
  const [agreements, setAgreements] = useState({
    terms: false,
    privacy: false
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    
    // Clear field error when user starts typing
    if (fieldErrors[name]) {
      setFieldErrors({
        ...fieldErrors,
        [name]: ''
      });
    }
  };

  const validateField = (name: string, value: string): string => {
    switch (name) {
      case 'firstName':
        if (!value.trim()) return t('firstNameRequired');
        if (value.trim().length < 2) return t('firstNameMinLength');
        return '';
      case 'email':
        if (!value.trim()) return t('emailRequired');
        const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
        if (!emailPattern.test(value)) return t('validEmailAddress');
        return '';
      case 'password':
        if (!value) return t('passwordRequired');
        if (value.length < 8) return t('passwordMinLength');
        if (!/(?=.*[a-z])/.test(value)) return t('passwordLowercase');
        if (!/(?=.*[A-Z])/.test(value)) return t('passwordUppercase');
        if (!/(?=.*\d)/.test(value)) return t('passwordNumber');
        return '';
      case 'confirmPassword':
        if (!value) return t('confirmPasswordRequired');
        if (value !== formData.password) return t('passwordsDoNotMatch');
        return '';
      default:
        return '';
    }
  };

  const handleAgreementChange = (type: 'terms' | 'privacy') => {
    setAgreements({
      ...agreements,
      [type]: !agreements[type]
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setFieldErrors({});

    // Validate all fields
    const errors: {[key: string]: string} = {};
    let hasErrors = false;

    // Validate required fields
    if (!formData.firstName.trim()) {
      errors.firstName = t('firstNameRequired');
      hasErrors = true;
    }
    if (!formData.email.trim()) {
      errors.email = t('emailRequired');
      hasErrors = true;
    }
    if (!formData.password) {
      errors.password = t('passwordRequired');
      hasErrors = true;
    }
    if (!formData.confirmPassword) {
      errors.confirmPassword = t('confirmPasswordRequired');
      hasErrors = true;
    }

    // Validate field formats
    Object.keys(formData).forEach(field => {
      const error = validateField(field, formData[field as keyof typeof formData]);
      if (error) {
        errors[field] = error;
        hasErrors = true;
      }
    });

    // Check agreements
    if (!agreements.terms || !agreements.privacy) {
      setError(t('agreeToTerms'));
      return;
    }

    if (hasErrors) {
      setFieldErrors(errors);
      return;
    }

    setIsLoading(true);

    try {
      console.log('🔄 Attempting to register user:', formData.email);
      const response = await authService.register({
        firstName: formData.firstName.trim(),
        lastName: formData.lastName.trim(),
        email: formData.email.trim().toLowerCase(),
        password: formData.password
      });

      if (response.success && response.data) {
        console.log('✅ Registration successful:', response.data);
        onLogin(response.data);
        navigate('/dashboard');
      } else {
        console.error('❌ Registration failed:', response.message);
        setError(response.message || 'Registration failed. Please try again.');
      }
    } catch (err) {
      console.error('❌ Registration error:', err);
      setError('An error occurred during registration. Please check your connection and try again.');
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
              theme === 'dark' ? 'bg-slate-800' : 'bg-white'
            }`}>
          <div className="text-center mb-8">
            <h1 className={`text-2xl font-bold mb-2 ${
              theme === 'dark' ? 'text-white' : 'text-gray-800'
            }`}>{t('createAccount')}</h1>
            
            <div className="mb-4">
              <Logo size="md" />
            </div>
            
            <p className={`text-sm ${
              theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
            }`}>
              {t('joinCareMateTrackHealth')}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className={`px-4 py-3 rounded-lg text-sm transition-colors duration-300 ${
                theme === 'dark' 
                  ? 'bg-red-900/50 border border-red-700 text-red-300' 
                  : 'bg-red-50 border border-red-200 text-red-600'
              }`}>
                {error}
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={`block text-sm font-medium mb-1 ${
                  theme === 'dark' ? 'text-slate-300' : 'text-gray-700'
                }`}>
                  {t('firstName')}*
                </label>
                <input
                  type="text"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleInputChange}
                  placeholder={t('firstNamePlaceholder')}
                  className={`w-full px-3 py-2 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 transition-colors text-sm ${
                    fieldErrors.firstName
                      ? 'ring-2 ring-red-500'
                      : theme === 'dark'
                      ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600'
                      : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white'
                  }`}
                  required
                />
                {fieldErrors.firstName && (
                  <p className="text-red-500 text-xs mt-1">{fieldErrors.firstName}</p>
                )}
              </div>
              <div>
                <label className={`block text-sm font-medium mb-1 ${
                  theme === 'dark' ? 'text-slate-300' : 'text-gray-700'
                }`}>
                  {t('lastName')}
                </label>
                <input
                  type="text"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleInputChange}
                  placeholder={t('lastNamePlaceholder')}
                  className={`w-full px-3 py-2 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 transition-colors text-sm ${
                    fieldErrors.lastName
                      ? 'ring-2 ring-red-500'
                      : theme === 'dark'
                      ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600'
                      : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white'
                  }`}
                />
                {fieldErrors.lastName && (
                  <p className="text-red-500 text-xs mt-1">{fieldErrors.lastName}</p>
                )}
              </div>
            </div>

            <div>
              <label className={`block text-sm font-medium mb-1 ${
                theme === 'dark' ? 'text-slate-300' : 'text-gray-700'
              }`}>
                {t('emailAddress')}*
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                  placeholder={t('emailPlaceholder')}
                className={`w-full px-3 py-2 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 transition-colors text-sm ${
                  fieldErrors.email
                    ? 'ring-2 ring-red-500'
                    : theme === 'dark'
                    ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600'
                    : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white'
                }`}
                required
              />
              {fieldErrors.email && (
                <p className="text-red-500 text-xs mt-1">{fieldErrors.email}</p>
              )}
            </div>

            <div>
              <label className={`block text-sm font-medium mb-1 ${
                theme === 'dark' ? 'text-slate-300' : 'text-gray-700'
              }`}>
                {t('password')}*
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  value={formData.password}
                  onChange={handleInputChange}
                  placeholder={t('passwordPlaceholder')}
                  className={`w-full px-3 py-2 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 transition-colors text-sm pr-10 ${
                    fieldErrors.password
                      ? 'ring-2 ring-red-500'
                      : theme === 'dark'
                      ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600'
                      : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white'
                  }`}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 transition-colors ${
                    theme === 'dark' 
                      ? 'text-slate-400 hover:text-slate-300' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {fieldErrors.password && (
                <p className="text-red-500 text-xs mt-1">{fieldErrors.password}</p>
              )}
            </div>

            <div>
              <label className={`block text-sm font-medium mb-1 ${
                theme === 'dark' ? 'text-slate-300' : 'text-gray-700'
              }`}>
                {t('confirmPassword')}*
              </label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  placeholder={t('confirmPasswordPlaceholder')}
                  className={`w-full px-3 py-2 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 transition-colors text-sm pr-10 ${
                    fieldErrors.confirmPassword
                      ? 'ring-2 ring-red-500'
                      : theme === 'dark'
                      ? 'bg-slate-700 text-white placeholder-slate-400 focus:bg-slate-600'
                      : 'bg-gray-100 text-gray-900 placeholder-gray-500 focus:bg-white'
                  }`}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 transition-colors ${
                    theme === 'dark' 
                      ? 'text-slate-400 hover:text-slate-300' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {fieldErrors.confirmPassword && (
                <p className="text-red-500 text-xs mt-1">{fieldErrors.confirmPassword}</p>
              )}
            </div>

            <div className="space-y-2 text-xs">
              <label className="flex items-start">
                <input
                  type="checkbox"
                  checked={agreements.terms}
                  onChange={() => handleAgreementChange('terms')}
                  className="mt-1 mr-2 rounded"
                />
                <span className={`${
                  theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
                }`}>{t('iAgreeToTerms')}</span>
              </label>
              <label className="flex items-start">
                <input
                  type="checkbox"
                  checked={agreements.privacy}
                  onChange={() => handleAgreementChange('privacy')}
                  className="mt-1 mr-2 rounded"
                />
                <span className={`${
                  theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
                }`}>
                  {t('iAgreeToPrivacy')}
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gray-600 hover:bg-[#183172] text-white py-3 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? t('creatingAccount') : t('createAccount')}
            </button>

            <div className={`text-center text-sm ${
              theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
            }`}>
              {t('alreadyHaveAccount')}{' '}
              <button
                type="button"
                onClick={() => navigate('/login')}
                className={`underline transition-colors ${
                  theme === 'dark' 
                    ? 'text-blue-400 hover:text-blue-300' 
                    : 'text-[#183172] hover:text-[#183172]/80'
                }`}
              >
                {t('signIn')}
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

export default SignUpPage;