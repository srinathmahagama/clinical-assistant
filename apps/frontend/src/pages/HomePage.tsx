import React from 'react';
import { useNavigate } from 'react-router-dom';
import { User, Mic, FileText, MessageCircle, Clock } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import { useLanguage } from '../contexts/LanguageContext';

interface HomePageProps {
  onNavigate: (page: string) => void;
}

const HomePage: React.FC<HomePageProps> = ({ onNavigate: _ }) => {
  const navigate = useNavigate();
  const { t } = useLanguage();
  return (
    <Layout backgroundType="auth">
      <Header 
        showLanguage={true} 
        title="CareMate" 
        showSignIn={true} 
        onSignIn={() => navigate('/login')} 
      />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-4xl mx-auto">
          {/* Main Title */}
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-[#183172] mb-4">
              {t('welcomeToCareMate')}
            </h1>
            <p className="text-gray-600 text-lg">
              {t('shareSymptoms')}
            </p>
          </div>

          {/* Main Actions */}
          <div className="space-y-6">
            {/* Create Account Section */}
            <div className="bg-gray-100 rounded-2xl p-6 shadow-lg">
              <div className="text-center">
                <div className="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-3">
                  <User className="w-5 h-5 text-gray-600" />
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-2">{t('createYourAccount')}</h2>
                <p className="text-gray-600 mb-4 text-sm">
                  {t('saveYourHealthAssessments')}
                </p>
                <div className="flex gap-4 justify-center">
                  <button
                    onClick={() => navigate('/signup')}
                    className="bg-gray-600 hover:bg-[#183172] text-white px-20 py-3 rounded-full font-medium transition-colors"
                  >
                    {t('createAccount')}
                  </button>
                  <button
                    onClick={() => navigate('/login')}
                    className="bg-gray-600 hover:bg-[#183172] text-white px-20 py-3 rounded-full font-medium transition-colors"
                  >
                    {t('signIn')}
                  </button>
                </div>
              </div>
            </div>

            {/* Voice Report Section */}
            <div className="bg-gray-100 rounded-2xl p-6 shadow-lg">
              <div className="text-center">
                <div className="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Mic className="w-5 h-5 text-gray-600" />
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-2">{t('tellUsHowYouFeel')}</h2>
                <p className="text-gray-600 mb-4 text-sm">
                  {t('tapAndSpeak')}
                </p>
                <button
                  onClick={() => navigate('/voice-input')}
                  className="bg-gray-600 hover:bg-[#183172] text-white px-48 py-3 rounded-full font-medium transition-colors"
                >
                  {t('startVoiceReport')}
                </button>
              </div>
            </div>

            {/* Text Report Section */}
            <div className="bg-gray-100 rounded-2xl p-6 shadow-lg">
              <div className="text-center">
                <div className="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-3">
                  <FileText className="w-5 h-5 text-gray-600" />
                </div>
                <h2 className="text-xl font-bold text-gray-800 mb-2">{t('typeYourSymptoms')}</h2>
                <p className="text-gray-600 mb-4 text-sm">
                  {t('typeAndDescribe')}
                </p>
                <button
                  onClick={() => navigate('/text-input')}
                  className="bg-gray-600 hover:bg-[#183172] text-white px-48 py-3 rounded-full font-medium transition-colors"
                >
                  {t('startTextReport')}
                </button>
              </div>
            </div>

            {/* Bottom Actions */}
            <div className="grid grid-cols-2 gap-6">
              <button
                onClick={() => navigate('/assistant')}
                className="bg-gray-100 rounded-2xl p-6 shadow-lg text-center hover:bg-gray-200 transition-colors"
              >
                <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <MessageCircle className="w-6 h-6 text-gray-600" />
                </div>
                <h3 className="font-bold text-gray-800 mb-2">{t('askAssistant')}</h3>
                <p className="text-sm text-gray-600">{t('getHealthGuidance')}</p>
              </button>
              
              <button
                onClick={() => navigate('/history')}
                className="bg-gray-100 rounded-2xl p-6 shadow-lg text-center hover:bg-gray-200 transition-colors"
              >
                <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Clock className="w-6 h-6 text-gray-600" />
                </div>
                <h3 className="font-bold text-gray-800 mb-2">{t('viewHistory')}</h3>
                <p className="text-sm text-gray-600">{t('pastAssessments')}</p>
              </button>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default HomePage;