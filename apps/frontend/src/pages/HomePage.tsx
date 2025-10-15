import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Heart,
  Users,
  Shield,
  Play,
  Pause,
  Volume2,
  BookOpen,
  Mic,
  Camera,
  X,
} from "lucide-react";
import { useTheme } from "../contexts/ThemeContext";
import { useLanguage } from "../contexts/LanguageContext";
import LanguageDropdown from "../components/LanguageDropdown/LanguageDropdown";
import ThemeToggle from "../components/ThemeToggle/ThemeToggle";
import Logo from "../components/Logo/Logo";

// Cultural Popup Component
const CulturalPopup = ({ element, isOpen, onClose, currentLanguage }) => {
  const popupRef = useRef(null);

  // Close popup when clicking outside or pressing Escape
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (popupRef.current && !popupRef.current.contains(event.target)) {
        onClose();
      }
    };

    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const content = element.popupContent[currentLanguage] || element.popupContent.english;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div
        ref={popupRef}
        className="bg-white dark:bg-slate-800 rounded-3xl max-w-md w-full p-6 relative border-2 border-amber-400 shadow-2xl"
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
        >
          <X className="w-6 h-6" />
        </button>

        {/* Header */}
        <div className="text-center mb-4">
          <div className="text-4xl mb-3">{element.emoji}</div>
          <h3 className="text-2xl font-bold text-gray-800 dark:text-white">
            {content.title}
          </h3>
          <p className="text-gray-600 dark:text-gray-300">
            {element.description}
          </p>
        </div>

        {/* Content */}
        <div className="mb-4">
          <p className="text-gray-700 dark:text-gray-200 text-lg leading-relaxed">
            {content.content}
          </p>
          {content.translation && (
            <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-700">
              <p className="text-sm text-amber-800 dark:text-amber-200">
                <strong>Translation:</strong> {content.translation}
              </p>
            </div>
          )}
        </div>

        {/* Language Indicator */}
        <div className="text-center">
          <span className="inline-block px-3 py-1 bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 rounded-full text-sm font-medium">
            {currentLanguage === 'noongar' ? 'Noongar' : 'English'}
          </span>
        </div>
      </div>
    </div>
  );
};

// Cultural Elements Data
const culturalElements = [
  {
    title: "Boodja Ngangk",
    description: "Country healing ways",
    emoji: "🌿",
    popupContent: {
      noongar: {
        title: "Boodja Ngangk",
        content: "Boodja ngaangk boola djerap. Ngalak katitjin koorliny boodja, boola moort kwop. Djena ngaangk wer djinang warr-abooliny boodja.",
        translation: "Country healing is strong. We learn walking on country, for family wellbeing. See healing and know smart country ways."
      },
      english: {
        title: "Country Healing Ways",
        content: "The land holds ancient healing knowledge. Walking on country connects us to traditional healing practices that have cared for our people for thousands of generations.",
        translation: ""
      }
    }
  },
  {
    title: "Moort Kwop",
    description: "Family wellbeing",
    emoji: "👨‍👩‍👧‍👦",
    popupContent: {
      noongar: {
        title: "Moort Kwop",
        content: "Moort kwop boola djerap koorliny. Ngalak ngaangk wer katitjin moort, boola kwop wer djerap boodja. Yira moort ngaangk yira koorliny.",
        translation: "Family wellbeing makes strong walking. We heal and learn family, for good and strong country. Your family heals your walking."
      },
      english: {
        title: "Family Wellbeing",
        content: "Our strength comes from our families. When we care for each other and share knowledge across generations, we create lasting health and happiness for all our relations.",
        translation: ""
      }
    }
  },
  {
    title: "Koorliny Boodja",
    description: "Walking on country",
    emoji: "🦘",
    popupContent: {
      noongar: {
        title: "Koorliny Boodja",
        content: "Koorliny boodja boola ngaangk wer katitjin. Djena boodja wer warrima boodja, boola kwop koorliny. Yira koorliny yira ngaangk.",
        translation: "Walking country brings healing and learning. See country and listen to country, for good walking. Your walking is your healing."
      },
      english: {
        title: "Walking on Country",
        content: "Walking on country is medicine for mind, body and spirit. It connects us to our ancestors, our stories, and the healing rhythms of the land that sustain us.",
        translation: ""
      }
    }
  },
  {
    title: "Katitjin Ngangk",
    description: "Healing knowledge",
    emoji: "📚",
    popupContent: {
      noongar: {
        title: "Katitjin Ngangk",
        content: "Katitjin ngaangk boola djerap wer kwop. Ngalak djinang ngaangk wer warr-abooliny katitjin, boola moort wer boodja kwop. Yira katitjin yira djerap.",
        translation: "Healing knowledge is strong and good. We know healing and smart knowledge, for family and country wellbeing. Your knowledge is your strength."
      },
      english: {
        title: "Healing Knowledge",
        content: "Our healing knowledge comes from thousands of years of living in harmony with country. This wisdom guides us in caring for our people and maintaining balance in all aspects of life.",
        translation: ""
      }
    }
  }
];

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
  const [showWelcomeModal, setShowWelcomeModal] = useState(!user && !isGuest);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [activeFeature, setActiveFeature] = useState(0);
  const [selectedLanguage, setSelectedLanguage] = useState("");
  const [activePopup, setActivePopup] = useState(null);

  const handleStartInEnglish = () => {
    setShowWelcomeModal(false);
    setLanguage("en");
    setSelectedLanguage("en");
  };

  const handleStartInNoongar = () => {
    setShowWelcomeModal(false);
    setLanguage("noongar");
    setSelectedLanguage("noongar");
  };

  const handleEnterApp = () => {
    onContinueAsGuest();
  };

  const handleCloseModal = () => {
    setShowWelcomeModal(false);
  };

  const playWelcomeAudio = () => {
    setIsPlayingAudio(true);
    setTimeout(() => setIsPlayingAudio(false), 3000);
  };

  const handleCulturalElementClick = (index) => {
    setActivePopup(index);
  };

  const handleClosePopup = () => {
    setActivePopup(null);
  };

  const features = [
    {
      icon: <Heart className="w-10 h-10" />,
      title: "Ngangk Moort Boodja",
      description: "Warr-abooliny djinang ngaangk, boola koorliny kwobidak",
      noongarDesc: "Ngangk moort boodja warr-abooliny djinang ngaangk"
    },
    {
      icon: <Users className="w-10 h-10" />,
      title: "Moort Ngangk",
      description: "Koorliny ngaangk moort, boola kwop wer djerap",
      noongarDesc: "Moort ngaangk koorliny boola kwop wer djerap"
    },
    {
      icon: <Shield className="w-10 h-10" />,
      title: "Koorliny Kwop",
      description: "Yira koorliny kwop, katatjininy wer djerap",
      noongarDesc: "Yira koorliny kwop katatjininy wer djerap"
    },
  ];

  const interactiveActions = [
    {
      icon: <Mic className="w-6 h-6" />,
      label: "Warrima Noongar",
      action: () => console.log("Voice input activated")
    },
    {
      icon: <Camera className="w-6 h-6" />,
      label: "Djena Boodja",
      action: () => console.log("Camera activated")
    },
    {
      icon: <BookOpen className="w-6 h-6" />,
      label: "Katitjin Ngangk",
      action: () => navigate("/knowledge")
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Background with Australian landscape */}
      <div className="relative min-h-screen">
        {/* Background Image - Australian landscape */}
        <div
          className="absolute inset-0 bg-cover bg-center bg-no-repeat"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1519066629447-267fffa62d4b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80')`,
          }}
        />

        {/* Cultural Pattern Overlay */}
        <div
          className={`absolute inset-0 ${
            theme === "dark"
              ? "bg-gradient-to-br from-emerald-900/70 via-slate-900/80 to-amber-900/60"
              : "bg-gradient-to-br from-emerald-600/50 via-blue-500/40 to-amber-400/50"
          }`}
        />

        {/* Content */}
        <div className="relative z-10">
          {/* Header */}
          <div className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Logo size="sm" className="w-12 h-12" />
                <div>
                  <h1 className="text-2xl font-bold text-white">CareMate</h1>
                  <p className="text-white/80 text-sm">Ngangk Moort</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <ThemeToggle />
                <LanguageDropdown />
                {selectedLanguage && (
                  <button
                    onClick={handleEnterApp}
                    className="bg-gradient-to-r from-amber-500 to-emerald-600 text-white px-6 py-2 rounded-xl font-semibold hover:from-amber-600 hover:to-emerald-700 transition-all"
                  >
                    Enter App
                  </button>
                )}
                <button
                  onClick={playWelcomeAudio}
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors bg-white/20 text-white hover:bg-white/30"
                >
                  {isPlayingAudio ? (
                    <Pause className="w-4 h-4" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  <Volume2 className="w-4 h-4" />
                  <span>Noongar Audio</span>
                </button>
              </div>
            </div>
          </div>

          {/* Hero Section */}
          <div className="max-w-7xl mx-auto px-6 py-12">
            <div className="text-center mb-16">
              {/* Welcome in Noongar */}
              <div className="mb-8">
                <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                  {selectedLanguage === "noongar" ? "Kaya! Welcome to CareMate" : "Welcome to CareMate"}
                </h2>
                <p className="text-2xl text-amber-300 mb-4">
                  Ngangk Moort - Healing Together
                </p>
                {selectedLanguage && (
                  <div className="bg-white/20 backdrop-blur-sm rounded-2xl p-4 max-w-md mx-auto">
                    <p className="text-white text-lg">
                      {selectedLanguage === "noongar" 
                        ? "Ngalak koorliny kwop - Let's walk well together" 
                        : "Explore our healing features below"}
                    </p>
                  </div>
                )}
              </div>

              {/* Main Interactive Card */}
              <div className="backdrop-blur-sm rounded-3xl p-12 border-2 bg-white/10 border-amber-400/30 max-w-4xl mx-auto shadow-2xl">
                <h2 className="text-3xl font-bold text-white mb-6">
                  Koorliny Kwop - Walk Well
                </h2>
                <p className="text-xl mb-8 text-white/90 max-w-2xl mx-auto">
                  Your health journey on Country. Safe, cultural, and strong.
                </p>

                {/* Interactive Action Buttons */}
                <div className="flex flex-wrap justify-center gap-6 mb-8">
                  {interactiveActions.map((action, index) => (
                    <button
                      key={index}
                      onClick={action.action}
                      className="flex flex-col items-center space-y-2 p-4 rounded-2xl bg-white/20 hover:bg-white/30 transition-all transform hover:scale-105 min-w-[120px]"
                    >
                      <div className="text-amber-300">{action.icon}</div>
                      <span className="text-white font-semibold text-sm">
                        {action.label}
                      </span>
                    </button>
                  ))}
                </div>

                {/* Main Access Button - Only show if no language selected */}
                {!selectedLanguage ? (
                  <button
                    onClick={() => setShowWelcomeModal(true)}
                    className="bg-gradient-to-r from-amber-500 to-emerald-600 text-white px-12 py-4 rounded-2xl font-bold text-xl hover:from-amber-600 hover:to-emerald-700 transition-all transform hover:scale-105 shadow-2xl border-2 border-amber-300/50"
                  >
                    Start Your Healing Journey - Koorliny Ngangk
                  </button>
                ) : (
                  <div className="space-y-4">
                    <button
                      onClick={handleEnterApp}
                      className="bg-gradient-to-r from-amber-500 to-emerald-600 text-white px-12 py-4 rounded-2xl font-bold text-xl hover:from-amber-600 hover:to-emerald-700 transition-all transform hover:scale-105 shadow-2xl border-2 border-amber-300/50"
                    >
                      {selectedLanguage === "noongar" 
                        ? "Koorliny App - Enter App" 
                        : "Enter CareMate App"}
                    </button>
                    <p className="text-white/80 text-sm">
                      {selectedLanguage === "noongar"
                        ? "Boola djinang ngaangk - Explore healing knowledge first"
                        : "Explore our features above before entering the app"}
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Cultural Features Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
              {culturalElements.map((element, index) => (
                <div
                  key={index}
                  onClick={() => handleCulturalElementClick(index)}
                  className="backdrop-blur-sm rounded-2xl p-6 border bg-white/10 border-white/20 text-center hover:transform hover:scale-105 transition-all cursor-pointer hover:border-amber-400/50"
                >
                  <div className="text-3xl mb-3">{element.emoji}</div>
                  <h3 className="text-lg font-bold text-white mb-2">
                    {element.title}
                  </h3>
                  <p className="text-white/80 text-sm">
                    {element.description}
                  </p>
                  <div className="mt-2">
                    <span className="text-amber-300 text-xs">Click to learn more</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Features with Cultural Context */}
            <div className="grid md:grid-cols-3 gap-8 mb-20">
              {features.map((feature, index) => (
                <div
                  key={index}
                  className="backdrop-blur-sm rounded-2xl p-8 border bg-white/15 border-amber-400/30 hover:border-amber-400/50 transition-all cursor-pointer"
                  onMouseEnter={() => setActiveFeature(index)}
                >
                  <div className="text-amber-300 mb-4 flex justify-center">
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold text-white mb-3 text-center">
                    {feature.title}
                  </h3>
                  <p className="text-white/80 text-center mb-3">
                    {feature.description}
                  </p>
                  <p className="text-amber-300 text-sm text-center font-semibold">
                    {feature.noongarDesc}
                  </p>
                </div>
              ))}
            </div>

            {/* Community Section */}
            <div className="text-center backdrop-blur-sm rounded-3xl p-12 border-2 bg-emerald-900/30 border-emerald-400/30 mb-16">
              <h2 className="text-3xl font-bold text-white mb-4">
                Moort Ngangk - Community Healing
              </h2>
              <p className="text-xl mb-8 text-white/90 max-w-2xl mx-auto">
                Together we walk strong. Share stories, learn healing ways, keep culture alive.
              </p>
              <div className="flex justify-center gap-6">
                <button
                  onClick={() => selectedLanguage ? navigate("/community") : setShowWelcomeModal(true)}
                  className="bg-white/20 text-white px-8 py-3 rounded-xl font-semibold hover:bg-white/30 transition-all"
                >
                  Join Community - Boola Moort
                </button>
                <button
                  onClick={() => selectedLanguage ? navigate("/stories") : setShowWelcomeModal(true)}
                  className="bg-amber-500 text-white px-8 py-3 rounded-xl font-semibold hover:bg-amber-600 transition-all"
                >
                  Share Your Story - Djinang Kwobidak
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Welcome Modal - Only show if no language selected */}
      {showWelcomeModal && !selectedLanguage && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div
            className={`rounded-3xl max-w-md w-full p-8 relative border-2 ${
              theme === "dark" 
                ? "bg-slate-800 border-amber-500/30" 
                : "bg-white border-amber-400"
            }`}
          >
            {/* Close Button */}
            <button
              onClick={handleCloseModal}
              className={`absolute top-4 right-4 transition-colors ${
                theme === "dark"
                  ? "text-amber-300 hover:text-amber-200"
                  : "text-amber-600 hover:text-amber-800"
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

            {/* Cultural Welcome Header */}
            <div className="text-center mb-6">
              <div className="w-16 h-16 bg-gradient-to-r from-amber-500 to-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🌿</span>
              </div>
              <h2
                className={`text-3xl font-bold ${
                  theme === "dark" ? "text-amber-300" : "text-emerald-800"
                }`}
              >
                Kaya! Welcome
              </h2>
              <p className={`text-lg ${theme === "dark" ? "text-amber-200" : "text-emerald-600"}`}>
                Ngangk Moort Boodja
              </p>
            </div>

            {/* Welcome Message */}
            <div className={`text-center mb-8 p-4 rounded-2xl ${
              theme === "dark" ? "bg-slate-700/50" : "bg-emerald-50"
            }`}>
              <p className={`${theme === "dark" ? "text-gray-200" : "text-gray-700"}`}>
                We acknowledge the Traditional Owners of the land and their continuing connection to country, culture and community.
              </p>
            </div>

            {/* Language Selection */}
            <div className="space-y-4">
              <button
                onClick={handleStartInNoongar}
                className="w-full bg-gradient-to-r from-amber-500 to-emerald-600 text-white py-4 px-6 rounded-2xl font-bold text-lg hover:from-amber-600 hover:to-emerald-700 transition-all transform hover:scale-105 shadow-lg"
              >
                Start in Noongar - Koorliny Noongar
              </button>

              <button
                onClick={handleStartInEnglish}
                className={`w-full py-4 px-6 rounded-2xl font-bold text-lg transition-all border-2 ${
                  theme === "dark"
                    ? "bg-slate-700 border-amber-500/30 text-amber-200 hover:bg-slate-600 hover:border-amber-400/50"
                    : "bg-white border-emerald-500 text-emerald-700 hover:bg-emerald-50 hover:border-emerald-600"
                }`}
              >
                Start in English
              </button>
            </div>

            {/* Cultural Note */}
            <div className={`mt-6 text-center text-sm ${
              theme === "dark" ? "text-amber-200/80" : "text-emerald-600"
            }`}>
              <p>This platform respects and celebrates Noongar culture and language</p>
            </div>
          </div>
        </div>
      )}

      {/* Cultural Popups */}
      {culturalElements.map((element, index) => (
        <CulturalPopup
          key={index}
          element={element}
          isOpen={activePopup === index}
          onClose={handleClosePopup}
          currentLanguage={selectedLanguage || "english"}
        />
      ))}
    </div>
  );
};

export default HomePage;