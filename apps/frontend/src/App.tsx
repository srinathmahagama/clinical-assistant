import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { LanguageProvider } from './contexts/LanguageContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { AudioProvider } from './contexts/AudioContext';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import SignUpPage from './pages/SignUpPage';
import DashboardPage from './pages/DashboardPage';
import AssistantPage from './pages/AssistantPage';
import HistoryPage from './pages/HistoryPage';
import SignInPopup from './components/SignInPopup/SignInPopup';
import ProtectedRoute from './components/ProtectedRoute/ProtectedRoute';
import { User } from './types';
import { authService } from './services/api';

// Component to handle popup visibility based on location
const AppContent: React.FC<{
  user: User | null;
  isGuest: boolean;
  showSignInPopup: boolean;
  setShowSignInPopup: (show: boolean) => void;
  handleLogin: (userData: User) => void;
  handleLogout: () => void;
  handleContinueAsGuest: () => void;
  handleNavigate: (data?: any) => void;
}> = ({ user, isGuest, showSignInPopup, setShowSignInPopup, handleLogin, handleLogout, handleContinueAsGuest, handleNavigate }) => {
  const location = useLocation();
  const navigate = useNavigate();
  
  // Enhanced logout handler that redirects to home
  const handleLogoutWithRedirect = async () => {
    await handleLogout();
    navigate('/home');
  };
  
  // Enhanced guest handler that redirects to dashboard
  const handleContinueAsGuestWithRedirect = async () => {
    await handleContinueAsGuest();
    navigate('/dashboard');
  };
  
  return (
    <>
      <Routes>
        <Route path="/" element={<Navigate to={user ? "/dashboard" : "/home"} replace />} />
        <Route path="/home" element={<HomePage onNavigate={handleNavigate} onLogout={handleLogoutWithRedirect} onContinueAsGuest={handleContinueAsGuestWithRedirect} user={user} isGuest={isGuest} />} />
        <Route path="/login" element={<LoginPage onNavigate={handleNavigate} onLogin={handleLogin} />} />
        <Route path="/signup" element={<SignUpPage onNavigate={handleNavigate} onLogin={handleLogin} />} />
        <Route 
          path="/dashboard" 
          element={
            <ProtectedRoute user={user} isGuest={isGuest}>
              <DashboardPage 
                user={user} 
                isGuest={isGuest}
                onLogout={handleLogoutWithRedirect}
                onSignIn={() => setShowSignInPopup(true)}
              />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/assistant" 
          element={
            <ProtectedRoute user={user} isGuest={isGuest}>
              <AssistantPage 
                user={user}
                isGuest={isGuest}
                onNavigate={handleNavigate} 
                onLogout={handleLogoutWithRedirect}
                onSignIn={() => setShowSignInPopup(true)}
              />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/history" 
          element={
            <ProtectedRoute user={user} isGuest={isGuest}>
              <HistoryPage 
                onNavigate={handleNavigate} 
                onLogout={handleLogoutWithRedirect}
                user={user}
                isGuest={isGuest}
              />
            </ProtectedRoute>
          } 
        />
      </Routes>
      
      {/* Sign-in Popup - Only show when not on home page, login page, or signup page, and when user is not logged in */}
      {location.pathname !== '/home' && location.pathname !== '/login' && location.pathname !== '/signup' && !user && (
        <SignInPopup
          isOpen={showSignInPopup}
          onClose={() => setShowSignInPopup(false)}
          onSignIn={() => {
            setShowSignInPopup(false);
            window.location.href = '/login';
          }}
          onSignUp={() => {
            setShowSignInPopup(false);
            window.location.href = '/signup';
          }}
          onContinueAsGuest={handleContinueAsGuestWithRedirect}
        />
      )}
    </>
  );
};

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isGuest, setIsGuest] = useState(false);
  const [pageData, setPageData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showSignInPopup, setShowSignInPopup] = useState(false);

  useEffect(() => {
    // Check if user is already logged in
    const checkAuth = async () => {
      try {
        // First check localStorage for quick access
        const userStr = localStorage.getItem('user');
        const token = localStorage.getItem('authToken');
        
        if (userStr && token) {
          const localUser = JSON.parse(userStr);
          setUser(localUser);
          setIsGuest(localUser.isGuest || false);
          
          // Verify with backend
          const response = await authService.getCurrentUser();
          if (response.success && response.data) {
            setUser(response.data);
            setIsGuest(response.data.isGuest || false);
          } else {
            // Token expired or invalid, clear local storage
            localStorage.removeItem('user');
            localStorage.removeItem('authToken');
            setUser(null);
            setIsGuest(false);
            setShowSignInPopup(true);
          }
        } else {
          // No local user data, show sign-in popup
          setShowSignInPopup(true);
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        setShowSignInPopup(true);
      } finally {
        setIsLoading(false);
      }
    };
    
    checkAuth();
  }, []);

  const handleLogin = (userData: User) => {
    setUser(userData);
    setIsGuest(userData.isGuest || false);
    setShowSignInPopup(false);
  };

  const handleLogout = async () => {
    await authService.logout();
    setUser(null);
    setIsGuest(false);
    setShowSignInPopup(false);
  };

  const handleContinueAsGuest = async () => {
    try {
      const response = await authService.createGuest();
      if (response.success && response.data) {
        setUser(response.data);
        setIsGuest(true);
        setShowSignInPopup(false);
      } else {
        console.error('Failed to create guest session:', response.message);
        // Fallback to local guest user
        const guestUser = {
          id: `guest-${Date.now()}`,
          firstName: 'Guest',
          lastName: 'User',
          email: `guest-${Date.now()}@temp.com`,
          createdAt: new Date().toISOString(),
          isEmailVerified: false
        };
        setUser(guestUser);
        setIsGuest(true);
        setShowSignInPopup(false);
      }
    } catch (error) {
      console.error('Error creating guest session:', error);
      // Fallback to local guest user
      const guestUser = {
        id: `guest-${Date.now()}`,
        firstName: 'Guest',
        lastName: 'User',
        email: `guest-${Date.now()}@temp.com`,
        createdAt: new Date().toISOString(),
        isEmailVerified: false
      };
      setUser(guestUser);
      setIsGuest(true);
      setShowSignInPopup(false);
    }
  };

  const handleNavigate = (data?: any) => {
    setPageData(data);
  };

  // Show loading spinner while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-400 via-pink-500 to-red-500">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-white/30 border-t-white rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg font-medium">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <ThemeProvider>
      <LanguageProvider>
        <AudioProvider>
          <Router>
            <div className="App">
              <AppContent
                user={user}
                isGuest={isGuest}
                showSignInPopup={showSignInPopup}
                setShowSignInPopup={setShowSignInPopup}
                handleLogin={handleLogin}
                handleLogout={handleLogout}
                handleContinueAsGuest={handleContinueAsGuest}
                handleNavigate={handleNavigate}
              />
            </div>
          </Router>
        </AudioProvider>
      </LanguageProvider>
    </ThemeProvider>
  );
}

export default App;