import React from 'react';
import { useNavigate } from 'react-router-dom';
import { MessageCircle, Clock, Stethoscope, ArrowRight, FileText } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import { User } from '../types';
import { useLanguage } from '../contexts/LanguageContext';
import { useTheme } from '../contexts/ThemeContext';

interface DashboardPageProps {
  user: User | null;
  isGuest: boolean;
  onLogout: () => void;
  onSignIn: () => void;
}

const DashboardPage: React.FC<DashboardPageProps> = ({ user, isGuest, onLogout, onSignIn }) => {
  const navigate = useNavigate();
  const { t } = useLanguage();
  const { theme } = useTheme();
  return (
    <Layout>
      <Header onLogout={onLogout} user={user} isGuest={isGuest} onSignIn={onSignIn} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-4xl mx-auto">

          {/* Guest User Notice */}
          {isGuest && (
            <div className={`mb-8 rounded-xl p-4 transition-colors duration-300 ${
              theme === 'dark' 
                ? 'bg-yellow-500/20 border border-yellow-500/30' 
                : 'bg-yellow-500/20 border border-yellow-500/30'
            }`}>
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-yellow-500/20 rounded-full flex items-center justify-center">
                  <span className="text-yellow-500 text-sm">👤</span>
                </div>
                <div>
                  <h3 className={`font-semibold ${
                    theme === 'dark' ? 'text-white' : 'text-white'
                  }`}>{t('guestUser')}</h3>
                  <p className={`text-sm ${
                    theme === 'dark' ? 'text-white/80' : 'text-white/80'
                  }`}>
                    {t('guestChatHistoryNote')} <button onClick={() => navigate('/login')} className="text-yellow-300 hover:text-yellow-200 underline">{t('signIn')}</button> {t('toSaveYourConversations')}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Welcome Section */}
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              {t('yourHealthAssistant')}
            </h2>
            <p className="text-white/90 text-lg">
              {t('shareSymptoms')}
            </p>
          </div>

          {/* Main Consultation Section */}
          <div 
            onClick={() => navigate('/assistant')}
            className={`rounded-3xl p-6 shadow-2xl mb-8 transition-all duration-300 hover:shadow-xl cursor-pointer ${
              theme === 'dark' 
                ? 'bg-gradient-to-r from-slate-800/40 to-slate-900/40 backdrop-blur-sm border border-white/10 hover:from-blue-500/20 hover:to-purple-600/20' 
                : 'bg-gradient-to-r from-blue-500 to-purple-600'
            }`}
          >
            <div className="text-center text-white">
              <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 ${
                theme === 'dark' ? 'bg-gradient-to-r from-blue-500 to-purple-600' : 'bg-white/20'
              }`}>
                <Stethoscope className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-3xl font-bold mb-4">{t('getHealthGuidance')}</h2>
              <p className="text-white/90 text-lg mb-8 max-w-2xl mx-auto">
                {t('shareSymptoms')}
              </p>
              <button
                onClick={() => navigate('/assistant')}
                className={`px-12 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 shadow-lg flex items-center mx-auto ${
                  theme === 'dark' 
                    ? 'bg-white text-slate-800 hover:bg-white/90' 
                    : 'bg-white text-blue-600 hover:bg-white/90'
                }`}
              >
                <MessageCircle className="w-6 h-6 mr-3" />
                {t('askAssistant')}
                <ArrowRight className="w-6 h-6 ml-3" />
              </button>
            </div>
          </div>

          {/* Features Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {/* <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 text-white">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mr-4">
                  <Mic className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold">{t('voiceInput')}</h3>
              </div>
              <p className="text-white/80">
                {t('tapAndSpeak')}
              </p>
            </div> */}

            {/* <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 text-white">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mr-4">
                  <FileText className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-bold">{t('textInput')}</h3>
              </div>
              <p className="text-white/80">
                {t('typeAndDescribe')}
              </p>
            </div> */}
          </div>

          {/* Quick Actions */}
          {/* <div className="grid grid-cols-2 gap-6"> */}
                      {/* <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-3xl p-8 shadow-2xl mb-8">

            <button
              onClick={() => navigate('/assistant')}
              className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 text-center text-white hover:bg-white/20 transition-all transform hover:scale-105"
            >
              <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <Heart className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-bold text-lg mb-2">{t('getHealthGuidance')}</h3>
              <p className="text-sm text-white/80">{t('shareSymptoms')}</p>
            </button>
           </div> */}
          
          
          {!isGuest ? (
            <div 
              onClick={() => navigate('/history')}
              className={`rounded-3xl p-6 shadow-2xl mb-8 transition-all duration-300 hover:shadow-xl cursor-pointer ${
                theme === 'dark' 
                  ? 'bg-gradient-to-r from-slate-800/40 to-slate-900/40 backdrop-blur-sm border border-white/10 hover:from-blue-500/20 hover:to-purple-600/20' 
                  : 'bg-gradient-to-r from-blue-500/35 to-purple-600/35 backdrop-blur-sm border border-white/20'
              }`}
            >
              <div className="text-center text-white">
                <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 ${
                  theme === 'dark' ? 'bg-gradient-to-r from-blue-500 to-purple-600' : 'bg-white/20'
                }`}>
                  <Clock className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-3xl font-bold mb-4">{t('viewHistory')}</h2>
                <p className="text-white/90 text-lg mb-8 max-w-2xl mx-auto">
                  {t('pastAssessments')}
                </p>
                <button
                  onClick={() => navigate('/history')}
                  className={`px-12 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 shadow-lg flex items-center mx-auto ${
                    theme === 'dark' 
                      ? 'bg-white text-slate-800 hover:bg-white/90' 
                      : 'bg-white text-blue-600 hover:bg-white/90'
                  }`}
                >
                  <FileText className="w-6 h-6 mr-3" />
                  {t('viewHistory')}
                  <ArrowRight className="w-6 h-6 ml-3" />
                </button>
              </div>
            </div>
          ) : (
            <div 
              onClick={() => navigate('/login')}
              className={`rounded-3xl p-6 shadow-2xl mb-8 transition-all duration-300 hover:shadow-xl cursor-pointer ${
                theme === 'dark' 
                  ? 'bg-gradient-to-r from-slate-800/40 to-slate-900/40 backdrop-blur-sm border border-white/10 hover:from-blue-500/20 hover:to-purple-600/20' 
                  : 'bg-gradient-to-r from-blue-500/35 to-purple-600/35 backdrop-blur-sm border border-white/20'
              }`}
            >
              <div className="text-center text-white">
                <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 ${
                  theme === 'dark' ? 'bg-gradient-to-r from-blue-500 to-purple-600' : 'bg-white/20'
                }`}>
                  <Clock className="w-10 h-10 text-white" />
                </div>
                <h2 className="text-3xl font-bold mb-4">{t('signInToViewHistory')}</h2>
                <p className="text-white/90 text-lg mb-8 max-w-2xl mx-auto">
                  {t('saveYourConvoToAccessAnyTime')}
                </p>
                <button
                  onClick={onSignIn}
                  className={`px-12 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 shadow-lg flex items-center mx-auto ${
                    theme === 'dark' 
                      ? 'bg-white text-slate-800 hover:bg-white/90' 
                      : 'bg-white text-blue-600 hover:bg-white/90'
                  }`}
                >
                  <FileText className="w-6 h-6 mr-3" />
                  {t('signInToViewHistory')}
                  <ArrowRight className="w-6 h-6 ml-3" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>


      
    </Layout>
  );
};

export default DashboardPage; 