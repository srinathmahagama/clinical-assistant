import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Mic, FileText, MessageCircle, Clock } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import { User } from '../types';
import { useLanguage } from '../contexts/LanguageContext';

interface DashboardPageProps {
  user: User;
  onNavigate: (page: string) => void;
  onLogout: () => void;
}

const DashboardPage: React.FC<DashboardPageProps> = ({ user, onNavigate, onLogout }) => {
  const navigate = useNavigate();
  const { t } = useLanguage();
  return (
    <Layout>
      <Header onLogout={onLogout} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-4xl mx-auto">
          {/* Greeting */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              {t('hi')} {user.firstName}.....
            </h1>
          </div>

          {/* Welcome Section */}
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              {t('welcomeToHealthAssistant')}
            </h2>
            <p className="text-white/90 text-lg">
              {t('shareSymptoms')}
            </p>
          </div>

          {/* Main Actions */}
          <div className="space-y-8">
            {/* Voice Report Section */}
            <div className="bg-gray-100 rounded-2xl p-8 shadow-lg">
              <div className="text-center">
                <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Mic className="w-6 h-6 text-gray-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">{t('tellUsHowYouFeel')}</h2>
                <p className="text-gray-600 mb-6">
                  {t('tapAndSpeak')}
                </p>
                <button
                  onClick={() => navigate('/voice-input')}
                  className="bg-[#183172] hover:bg-[#183172]/80 text-white px-48 py-3 rounded-full font-medium transition-colors"
                >
                  {t('startVoiceReport')}
                </button>
              </div>
            </div>

            {/* Text Report Section */}
            <div className="bg-gray-100 rounded-2xl p-8 shadow-lg">
              <div className="text-center">
                <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <FileText className="w-6 h-6 text-gray-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">{t('typeYourSymptoms')}</h2>
                <p className="text-gray-600 mb-6">
                  {t('typeAndDescribe')}
                </p>
                <button
                  onClick={() => navigate('/text-input')}
                  className="bg-[#183172] hover:bg-[#183172]/80 text-white px-48 py-3 rounded-full font-medium transition-colors"
                >
                  {t('startTextReport')}
                </button>
              </div>
            </div>

            {/* Bottom Actions */}
            <div className="grid grid-cols-2 gap-6">
              <button
                onClick={() => navigate('/assistant')}
                className="bg-gray-100 rounded-2xl p-6 text-center shadow-lg hover:bg-gray-200 transition-colors"
              >
                <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <MessageCircle className="w-6 h-6 text-gray-600" />
                </div>
                <h3 className="font-bold text-gray-800 mb-2">{t('askAssistant')}</h3>
                <p className="text-sm text-gray-600">{t('getHealthGuidance')}</p>
              </button>
              
              <button
                onClick={() => navigate('/history')}
                className="bg-gray-100 rounded-2xl p-6 text-center shadow-lg hover:bg-gray-200 transition-colors"
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

export default DashboardPage;