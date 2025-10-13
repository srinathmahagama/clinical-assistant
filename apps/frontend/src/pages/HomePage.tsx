import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Heart,
  Stethoscope,
  Users,
  Shield,
  ArrowRight,
  UserPlus,
  User,
} from "lucide-react";
import { useTheme } from "../contexts/ThemeContext";
import { useLanguage } from "../contexts/LanguageContext";
import { authService } from "../services/api";
import LanguageDropdown from "../components/LanguageDropdown/LanguageDropdown";
import ThemeToggle from "../components/ThemeToggle/ThemeToggle";
import Logo from "../components/Logo/Logo";

interface HomePageProps {
  onNavigate: (page: string) => void;
  onLogout: () => void;
  onContinueAsGuest: () => void;
  user?: any;
  isGuest?: boolean;
}

const HomePage: React.FC<HomePageProps> = ({
  onNavigate,
  onLogout,
  onContinueAsGuest,
  user,
  isGuest,
}) => {
  const navigate = useNavigate();
  const { theme } = useTheme();
  const { setLanguage, t } = useLanguage();
  const [showWelcomeModal, setShowWelcomeModal] = useState(!user);

  const handleSignIn = () => {
    setShowWelcomeModal(false);
    navigate("/login");
  };

  const handleCreateAccount = () => {
    setShowWelcomeModal(false);
    navigate("/signup");
  };

  const languageEnglish = () => {
    setShowWelcomeModal(false);
    setLanguage("en");
  };

  const languageNoongar = () => {
    setShowWelcomeModal(false);
    setLanguage("noongar");
  };

  const handleContinueAsGuest = () => {
    setShowWelcomeModal(false);
    onContinueAsGuest();
  };

  const handleCloseModal = () => {
    setShowWelcomeModal(false);
  };

  const features = [
    {
      icon: <Stethoscope className="w-8 h-8" />,
      title: t("aiHealthAssessment"),
      description: t("aiHealthAssessmentDesc"),
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: t("expertGuidance"),
      description: t("expertGuidanceDesc"),
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: t("privacyProtected"),
      description: t("privacyProtectedDesc"),
    },
  ];

  const stats = [
    { number: "50K+", label: t("healthAssessments") },
    { number: "98%", label: t("accuracyRate") },
    { number: "24/7", label: t("availableSupport") },
    { number: "100+", label: t("healthConditions") },
  ];

  return (
    <div className="min-h-screen">
      {/* Background with lady doctor image */}
      <div className="relative min-h-screen">
        {/* Background Image */}
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80')`,
          }}
        />

        {/* Theme-specific Gradient Overlay */}
        <div
          className={`absolute inset-0 ${
            theme === "dark"
              ? "bg-gradient-to-br from-black/85 via-gray-900/80 to-black/90"
              : "bg-gradient-to-br from-purple-900/80 via-indigo-900/70 to-pink-900/80"
          }`}
        />

        {/* Content */}
        <div className="relative z-10">
          {/* Header */}
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Logo size="sm" className="w-10 h-10" />
                <h1 className="text-2xl font-bold text-white">CareMate</h1>
              </div>
              <div className="flex items-center space-x-4">
                <ThemeToggle />
                <LanguageDropdown />
                <button
                  onClick={() => navigate("/login")}
                  className="text-white/90 hover:text-white transition-colors"
                >
                  {t("signIn")}
                </button>
                <button
                  onClick={() => navigate("/signup")}
                  className="px-4 py-2 rounded-lg transition-colors bg-white/20 text-white hover:bg-white/30"
                >
                  {t("signUp")}
                </button>
              </div>
            </div>
          </div>

          {/* Hero Section */}
          <div className="max-w-7xl mx-auto px-6 py-7">
            <div className="text-center mb-16">
              <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
                {t("yourHealthOurPriority")}
              </h1>
              {/* <p className="text-xl text-white/90 mb-8 max-w-3xl mx-auto">
    {t('experienceFutureHealthcare')}
  </p> */}

              <div className="backdrop-blur-sm rounded-2xl p-12 border bg-white/15 border-white/30 max-w-3xl mx-auto">
                {/* Button Stack */}
                <h2 className="text-3xl font-bold text-white mb-4">
                  {t("readyToTakeControl")}
                </h2>
                <p className="mb-8 max-w-2xl mx-auto text-white/80">
                  {t("joinThousandsUsers")}
                </p>
                <div className="flex flex-col items-center gap-6">
                  <button
                    onClick={() => navigate("/login")}
                    className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-10 py-4 rounded-xl font-semibold text-lg hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg w-72"
                  >
                    {t("loginToAccount")}
                  </button>

                  <button
                    onClick={() => navigate("/signup")}
                    className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-10 py-4 rounded-xl font-semibold text-lg hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg w-72"
                  >
                    {t("registerNow")}
                  </button>

                  <button
                    onClick={handleContinueAsGuest}
                    className="bg-white/20 text-white px-10 py-4 rounded-xl font-semibold text-lg hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg w-72"
                  >
                    {t("tryAsGuest")}
                  </button>
                </div>
              </div>
            </div>

            {/* {about} */}
            {/* <div className="max-w-4xl mx-auto mb-20 px-6">
  <div className="backdrop-blur-md bg-white/10 border border-white/20 rounded-2xl p-10 shadow-xl text-center">
    <p className="text-lg leading-relaxed text-white/90 whitespace-pre-line">
      {t('aboutText')}
    </p>
  </div>
</div> */}

            {/* Stats */}
            {/* <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-20">
              {stats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="text-3xl md:text-4xl font-bold text-white mb-2">
                    {stat.number}
                  </div>
                  <div className="text-white/80">
                    {stat.label}
                  </div>
                </div>
              ))}
            </div> */}

            {/* Features */}
            <div className="grid md:grid-cols-3 gap-8 mb-20">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className="backdrop-blur-sm rounded-xl p-8 border bg-white/15 border-white/30"
                >
                  <div className="text-blue-400 mb-4">{feature.icon}</div>
                  <h3 className="text-xl font-semibold text-white mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-white/80">{feature.description}</p>
                </div>
              ))}
            </div>

            {/* CTA Section */}
            {/* <div className="text-center backdrop-blur-sm rounded-2xl p-12 border bg-white/15 border-white/30">
              <h2 className="text-3xl font-bold text-white mb-4">
                {t('readyToTakeControl')}
              </h2>
              <p className="mb-8 max-w-2xl mx-auto text-white/80">
                {t('joinThousandsUsers')}
              </p>
              <button
                onClick={() => navigate('/signup')}
                className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-8 py-4 rounded-xl font-semibold text-lg hover:from-blue-600 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg"
              >
                {t('startHealthJourneyToday')}
              </button>
              
            </div> */}
          </div>
        </div>
      </div>

      {/* Welcome Modal */}
      {showWelcomeModal && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div
            className={`rounded-2xl max-w-md w-full p-8 relative ${
              theme === "dark" ? "bg-slate-800" : "bg-white"
            }`}
          >
            {/* Close Button */}
            <button
              onClick={handleCloseModal}
              className={`absolute top-4 right-4 transition-colors ${
                theme === "dark"
                  ? "text-gray-400 hover:text-gray-200"
                  : "text-gray-400 hover:text-gray-600"
              }`}
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>

            {/* Header */}
            <div className="flex items-center mb-6">
              <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mr-4">
                <User className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2
                  className={`text-2xl font-bold ${
                    theme === "dark" ? "text-white" : "text-gray-800"
                  }`}
                >
                  Welcome to Caremate
                </h2>
                <h2
                  className={`text-2xl font-bold ${
                    theme === "dark" ? "text-white" : "text-gray-800"
                  }`}
                >
                  Kaya CareMate
                </h2>

                {/* <p className={`${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'
                  }`}>{t('yourHealthAssistantReady')}</p> */}
              </div>
            </div>

            {/* Description */}
            {/* <p className={`mb-8 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-600'
                }`}>
                {t('signInSaveHistory')}
              </p> */}

            {/* Action Buttons */}
            <div className="space-y-4">
              {/* <button
                onClick={handleSignIn}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-6 rounded-xl font-semibold hover:from-blue-600 hover:to-purple-700 transition-all flex items-center justify-center"
              >
                <ArrowRight className="w-5 h-5 mr-2" />
                {t('signIn')}
              </button> */}

              <button
                onClick={languageEnglish}
                className="w-full bg-white border-2 border-blue-500 text-blue-500 py-3 px-6 rounded-xl font-semibold hover:bg-blue-50 transition-all flex items-center justify-center"
              >
                {/* <UserPlus className="w-5 h-5 mr-2" /> */}
                English
              </button>

              <button
                onClick={languageNoongar}
                className={`w-full py-3 px-6 rounded-xl font-semibold transition-all ${
                  theme === "dark"
                    ? "bg-gray-700 text-gray-200 hover:bg-gray-600"
                    : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                }`}
              >
                Noongar
              </button>
            </div>

            {/* Footer */}
            {/* <div className={`mt-6 text-center text-sm ${theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
              }`}>
              <p className="mb-2">{t('guestChatHistoryNote')}</p>
              <div className="space-y-1">
                <a href="#" className="text-blue-500 hover:underline">{t('forgetPassword')}</a>
                <br />
                <span>{t('createNewAccountIfDontHave')} </span>
                <a href="#" className="text-blue-500 hover:underline">{t('signUp')}</a>
              </div>
            </div> */}
          </div>
        </div>
      )}
    </div>
  );
};

export default HomePage;
