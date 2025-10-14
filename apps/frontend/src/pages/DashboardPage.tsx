import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MessageCircle, Clock, Stethoscope, ArrowRight, FileText, Heart, Users, Shield, Play, Volume2 } from 'lucide-react';
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
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [activeFeature, setActiveFeature] = useState(0);
  const [showWelcome, setShowWelcome] = useState(true);

  // Cultural animations and interactive elements
  const culturalElements = [
    {
      icon: "🌿",
      title: "Boodja Ngangk",
      description: "Country Healing",
      noongarDesc: "Ngangk boodja boola djerap"
    },
    {
      icon: "👨‍👩‍👧‍👦",
      title: "Moort Kwop",
      description: "Family Wellbeing",
      noongarDesc: "Moort kwop koorliny"
    },
    {
      icon: "🦘",
      title: "Koorliny Boodja",
      description: "Walking Country",
      noongarDesc: "Koorliny boodja ngaangk"
    },
    {
      icon: "📚",
      title: "Katitjin Ngangk",
      description: "Healing Knowledge",
      noongarDesc: "Katitjin ngaangk djerap"
    }
  ];

  const playWelcomeAudio = () => {
    setIsPlayingAudio(true);
    // Simulate audio playback
    setTimeout(() => setIsPlayingAudio(false), 3000);
  };

  useEffect(() => {
    // Welcome animation timeout
    const timer = setTimeout(() => {
      setShowWelcome(false);
    }, 4000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <Layout>
      <Header onLogout={onLogout} user={user} isGuest={isGuest} onSignIn={onSignIn} />
      
      {/* Animated Background */}
      <div className="fixed inset-0 -z-10">
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1519066629447-267fffa62d4b?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80')`,
          }}
        />
        <div className={`absolute inset-0 ${
          theme === "dark"
            ? "bg-gradient-to-br from-emerald-900/80 via-slate-900/70 to-amber-900/60"
            : "bg-gradient-to-br from-emerald-600/60 via-blue-500/50 to-amber-400/50"
        }`} />
        
        {/* Animated Cultural Patterns */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-10 left-10 w-20 h-20 bg-amber-300 rounded-full animate-pulse"></div>
          <div className="absolute bottom-20 right-20 w-16 h-16 bg-emerald-300 rounded-full animate-bounce"></div>
          <div className="absolute top-1/2 left-1/4 w-12 h-12 bg-blue-300 rounded-full animate-ping"></div>
        </div>
      </div>

      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-6xl mx-auto">

          {/* Welcome Animation */}
          {showWelcome && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
              <div className="text-center animate-bounce">
                <div className="text-6xl mb-4">👋</div>
                <h2 className="text-4xl font-bold text-white mb-2">Kaya! Welcome</h2>
                <p className="text-xl text-amber-300">Ngangk Moort Boodja</p>
              </div>
            </div>
          )}

          {/* Guest User Notice - Cultural Style */}
          {isGuest && (
            <div className={`mb-8 rounded-2xl p-6 transition-all duration-500 animate-fade-in ${
              theme === 'dark' 
                ? 'bg-amber-500/20 border border-amber-500/30 backdrop-blur-sm' 
                : 'bg-amber-500/25 border border-amber-500/40 backdrop-blur-sm'
            }`}>
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-amber-500/20 rounded-full flex items-center justify-center animate-pulse">
                  <span className="text-amber-500 text-lg">👤</span>
                </div>
                <div className="flex-1">
                  <h3 className={`font-bold text-lg ${
                    theme === 'dark' ? 'text-amber-200' : 'text-amber-800'
                  }`}>{t('guestUser')}</h3>
                  <p className={`text-sm ${
                    theme === 'dark' ? 'text-amber-100/80' : 'text-amber-700'
                  }`}>
                    {t('guestChatHistoryNote')} <button 
                      onClick={() => navigate('/login')} 
                      className="text-amber-300 hover:text-amber-200 underline font-semibold transition-colors"
                    >
                      {t('signIn')}
                    </button> {t('toSaveYourConversations')}
                  </p>
                </div>
                <button
                  onClick={playWelcomeAudio}
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors bg-amber-500/20 text-amber-200 hover:bg-amber-500/30"
                >
                  <Volume2 className="w-4 h-4" />
                  <span>Noongar Audio</span>
                </button>
              </div>
            </div>
          )}

          {/* Welcome Section with Cultural Greeting */}
          <div className="text-center mb-12 animate-slide-down">
            <div className="inline-block mb-4">
              <span className="text-4xl">👋</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Kaya! {t('yourHealthAssistant')}
            </h2>
            <p className="text-amber-300 text-xl mb-2">
              Ngangk Moort - Healing Together
            </p>
            <p className="text-white/90 text-lg max-w-2xl mx-auto">
              {t('shareSymptoms')}
            </p>
          </div>

          {/* Cultural Features Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8 animate-slide-up">
            {culturalElements.map((element, index) => (
              <div
                key={index}
                className="backdrop-blur-sm rounded-2xl p-4 border bg-white/10 border-white/20 text-center hover:transform hover:scale-105 transition-all duration-300 cursor-pointer hover:border-amber-400/50"
                onMouseEnter={() => setActiveFeature(index)}
              >
                <div className="text-3xl mb-2 animate-bounce">{element.emoji}</div>
                <h3 className="text-sm font-bold text-white mb-1">
                  {element.title}
                </h3>
                <p className="text-white/80 text-xs mb-1">
                  {element.description}
                </p>
                <p className="text-amber-300 text-xs font-semibold">
                  {element.noongarDesc}
                </p>
              </div>
            ))}
          </div>

          {/* Main Consultation Section - Cultural Design */}
          <div 
            onClick={() => navigate('/assistant')}
            className={`rounded-3xl p-8 shadow-2xl mb-8 transition-all duration-500 hover:shadow-xl cursor-pointer group animate-fade-in ${
              theme === 'dark' 
                ? 'bg-gradient-to-r from-emerald-900/40 to-amber-900/40 backdrop-blur-sm border border-amber-400/30 hover:from-emerald-800/50 hover:to-amber-800/50' 
                : 'bg-gradient-to-r from-emerald-500/60 to-amber-400/60 backdrop-blur-sm border border-amber-300'
            }`}
          >
            <div className="text-center text-white">
              <div className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 ${
                theme === 'dark' ? 'bg-gradient-to-r from-amber-500 to-emerald-600' : 'bg-white/20'
              }`}>
                <Stethoscope className="w-12 h-12 text-white" />
              </div>
              <h2 className="text-3xl font-bold mb-4 animate-pulse">Koorliny Ngangk - Start Healing Journey</h2>
              <p className="text-amber-200 text-xl mb-2">
                Ngangk boola kwop - Healing brings wellness
              </p>
              <p className="text-white/90 text-lg mb-8 max-w-2xl mx-auto">
                {t('shareSymptoms')}
              </p>
              <button
                onClick={() => navigate('/assistant')}
                className={`px-12 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 shadow-lg flex items-center mx-auto group-hover:shadow-2xl ${
                  theme === 'dark' 
                    ? 'bg-amber-500 text-white hover:bg-amber-600' 
                    : 'bg-amber-500 text-white hover:bg-amber-600'
                }`}
              >
                <MessageCircle className="w-6 h-6 mr-3" />
                {t('askAssistant')}
                <ArrowRight className="w-6 h-6 ml-3 group-hover:translate-x-2 transition-transform" />
              </button>
            </div>
          </div>

          {/* History Section with Cultural Context */}
          {!isGuest ? (
            <div 
              onClick={() => navigate('/history')}
              className={`rounded-3xl p-8 shadow-2xl mb-8 transition-all duration-500 hover:shadow-xl cursor-pointer group animate-fade-in ${
                theme === 'dark' 
                  ? 'bg-gradient-to-r from-slate-800/40 to-slate-900/40 backdrop-blur-sm border border-emerald-400/30 hover:from-emerald-900/30 hover:to-slate-800/50' 
                  : 'bg-gradient-to-r from-emerald-500/35 to-blue-500/35 backdrop-blur-sm border border-emerald-300'
              }`}
            >
              <div className="text-center text-white">
                <div className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 ${
                  theme === 'dark' ? 'bg-gradient-to-r from-emerald-500 to-blue-600' : 'bg-white/20'
                }`}>
                  <Clock className="w-12 h-12 text-white" />
                </div>
                <h2 className="text-3xl font-bold mb-4">Djinang Ngangk - Your Healing Story</h2>
                <p className="text-emerald-200 text-xl mb-2">
                  Katitjin koorliny - Learning from your journey
                </p>
                <p className="text-white/90 text-lg mb-8 max-w-2xl mx-auto">
                  {t('pastAssessments')}
                </p>
                <button
                  onClick={() => navigate('/history')}
                  className={`px-12 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 shadow-lg flex items-center mx-auto group-hover:shadow-2xl ${
                    theme === 'dark' 
                      ? 'bg-emerald-500 text-white hover:bg-emerald-600' 
                      : 'bg-emerald-500 text-white hover:bg-emerald-600'
                  }`}
                >
                  <FileText className="w-6 h-6 mr-3" />
                  {t('viewHistory')}
                  <ArrowRight className="w-6 h-6 ml-3 group-hover:translate-x-2 transition-transform" />
                </button>
              </div>
            </div>
          ) : (
            <div 
              onClick={() => navigate('/login')}
              className={`rounded-3xl p-8 shadow-2xl mb-8 transition-all duration-500 hover:shadow-xl cursor-pointer group animate-fade-in ${
                theme === 'dark' 
                  ? 'bg-gradient-to-r from-amber-900/30 to-orange-900/30 backdrop-blur-sm border border-amber-400/30 hover:from-amber-800/40 hover:to-orange-800/40' 
                  : 'bg-gradient-to-r from-amber-500/35 to-orange-500/35 backdrop-blur-sm border border-amber-300'
              }`}
            >
              <div className="text-center text-white">
                <div className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 ${
                  theme === 'dark' ? 'bg-gradient-to-r from-amber-500 to-orange-600' : 'bg-white/20'
                }`}>
                  <Heart className="w-12 h-12 text-white" />
                </div>
                <h2 className="text-3xl font-bold mb-4">Boola Moort - Join Our Healing Family</h2>
                <p className="text-amber-200 text-xl mb-2">
                  Ngalak koorliny kwop - Let's walk well together
                </p>
                <p className="text-white/90 text-lg mb-8 max-w-2xl mx-auto">
                  {t('saveYourConvoToAccessAnyTime')}
                </p>
                <button
                  onClick={onSignIn}
                  className={`px-12 py-4 rounded-full font-bold text-lg transition-all transform hover:scale-105 shadow-lg flex items-center mx-auto group-hover:shadow-2xl ${
                    theme === 'dark' 
                      ? 'bg-amber-500 text-white hover:bg-amber-600' 
                      : 'bg-amber-500 text-white hover:bg-amber-600'
                  }`}
                >
                  <Users className="w-6 h-6 mr-3" />
                  {t('signInToViewHistory')}
                  <ArrowRight className="w-6 h-6 ml-3 group-hover:translate-x-2 transition-transform" />
                </button>
              </div>
            </div>
          )}

          {/* Cultural Wisdom Section */}
          <div className={`rounded-3xl p-8 text-center backdrop-blur-sm border ${
            theme === 'dark' 
              ? 'bg-emerald-900/20 border-emerald-400/30' 
              : 'bg-emerald-500/20 border-emerald-400/40'
          }`}>
            <h3 className="text-2xl font-bold text-white mb-4">
              Noongar Katitjin - Cultural Wisdom
            </h3>
            <p className="text-white/90 text-lg mb-4">
              "Koorliny boodja boola ngaangk wer katitjin"
            </p>
            <p className="text-amber-300">
              Walking country brings healing and knowledge
            </p>
          </div>
        </div>
      </div>

      {/* Add custom animations to CSS */}
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideDown {
          from { opacity: 0; transform: translateY(-50px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(50px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.8s ease-out;
        }
        .animate-slide-down {
          animation: slideDown 0.8s ease-out;
        }
        .animate-slide-up {
          animation: slideUp 0.8s ease-out;
        }
      `}</style>
    </Layout>
  );
};

export default DashboardPage;