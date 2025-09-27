import React from 'react';
import Footer from '../Footer/Footer';

interface LayoutProps {
  children: React.ReactNode;
  showLanguageButton?: boolean;
  backgroundType?: 'auth' | 'dashboard';
}

const Layout: React.FC<LayoutProps> = ({ children, showLanguageButton = true, backgroundType = 'dashboard' }) => {
  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background Image */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: backgroundType === 'auth' 
            ? `url('https://images.pexels.com/photos/4173251/pexels-photo-4173251.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2')`
            : `url('https://images.pexels.com/photos/4386466/pexels-photo-4386466.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2')`
        }}
      >
        {/* Gradient Overlay */}
        <div className={`absolute inset-0 ${
          backgroundType === 'auth' 
            ? 'bg-gradient-to-br from-blue-500/70 via-indigo-600/70 to-purple-700/70'
            : 'bg-gradient-to-br from-blue-400/80 via-purple-500/80 to-pink-400/80'
        }`}></div>
      </div>

      {/* Language Button */}
      {showLanguageButton && (
        <div className="absolute top-4 right-4 z-10">
          <button className="bg-white/90 hover:bg-white text-gray-700 px-4 py-2 rounded-full text-sm font-medium transition-colors">
            Language
          </button>
        </div>
      )}

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        <div className="flex-1">
          {children}
        </div>
        <Footer />
      </div>
    </div>
  );
};

export default Layout;