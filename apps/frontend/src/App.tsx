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
  handleRestoreGuestSession: () => void;
}> = ({ user, isGuest, showSignInPopup, setShowSignInPopup, handleLogin, handleLogout, handleContinueAsGuest, handleNavigate, handleRestoreGuestSession }) => {
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

  // Check for guest session on mount and route changes
  useEffect(() => {
    const checkGuestSession = () => {
      const guestSession = localStorage.getItem('guestSession');
      const sessionTimestamp = localStorage.getItem('guestSessionTimestamp');
      
      if (guestSession === 'true' && sessionTimestamp && !user && !isGuest) {
        const sessionAge = Date.now() - parseInt(sessionTimestamp);
        const sessionMaxAge = 24 * 60 * 60 * 1000; // 24 hours
        
        if (sessionAge < sessionMaxAge) {
          // Valid guest session exists - restore it
          handleRestoreGuestSession();
        } else {
          // Session expired
          localStorage.removeItem('guestSession');
          localStorage.removeItem('guestSessionTimestamp');
          if (location.pathname !== '/home') {
            setShowSignInPopup(true);
          }
        }
      }
    };

    checkGuestSession();
  }, [location.pathname, user, isGuest, handleRestoreGuestSession]);

  return (
    <>
      <Routes>
        {/* FIX: Always redirect to home page first, regardless of auth status */}
        <Route path="/" element={<Navigate to="/home" replace />} />
        
        <Route 
          path="/home" 
          element={
            <HomePage 
              onNavigate={handleNavigate} 
              onLogout={handleLogoutWithRedirect} 
              onContinueAsGuest={handleContinueAsGuestWithRedirect} 
              user={user} 
              isGuest={isGuest} 
            />
          } 
        />
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
                onRestoreGuestSession={handleRestoreGuestSession}
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
      {location.pathname !== '/home' && location.pathname !== '/login' && location.pathname !== '/signup' && !user && !isGuest && (
        <SignInPopup
          isOpen={showSignInPopup}
          onClose={() => {
            setShowSignInPopup(false);
            // If user closes popup and no session exists, redirect to home
            if (!user && !isGuest) {
              navigate('/home');
            }
          }}
          onSignIn={() => {
            setShowSignInPopup(false);
            navigate('/login');
          }}
          onSignUp={() => {
            setShowSignInPopup(false);
            navigate('/signup');
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
    // Check if user is already logged in or has guest session
    const checkAuth = async () => {
      try {
        // First check for guest session
        const guestSession = localStorage.getItem('guestSession');
        const sessionTimestamp = localStorage.getItem('guestSessionTimestamp');
        
        if (guestSession === 'true' && sessionTimestamp) {
          const sessionAge = Date.now() - parseInt(sessionTimestamp);
          const sessionMaxAge = 24 * 60 * 60 * 1000; // 24 hours
          
          if (sessionAge < sessionMaxAge) {
            // Valid guest session exists
            const guestUser = {
              id: `guest-${sessionTimestamp}`,
              firstName: 'Guest',
              lastName: 'User',
              email: `guest@temp.com`,
              createdAt: new Date(parseInt(sessionTimestamp)).toISOString(),
              isEmailVerified: false,
              isGuest: true
            };
            setUser(guestUser);
            setIsGuest(true);
            setIsLoading(false);
            return;
          } else {
            // Guest session expired
            localStorage.removeItem('guestSession');
            localStorage.removeItem('guestSessionTimestamp');
          }
        }

        // Check for regular authenticated user
        const userStr = localStorage.getItem('user');
        const token = localStorage.getItem('authToken');
        
        if (userStr && token) {
          const localUser = JSON.parse(userStr);
          
          // Verify with backend
          const response = await authService.getCurrentUser();
          if (response.success && response.data) {
            setUser(response.data);
            setIsGuest(response.data.isGuest || false);
          } else {
            // Token expired or invalid, clear local storage
            localStorage.removeItem('user');
            localStorage.removeItem('authToken');
            // Don't show popup immediately, let user see home page first
          }
        }
        // If no session found, just continue to home page without showing popup
      } catch (error) {
        console.error('Auth check failed:', error);
        // Continue to home page even if auth check fails
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
    
    // Clear guest session if exists
    if (userData.isGuest) {
      localStorage.setItem('guestSession', 'true');
      localStorage.setItem('guestSessionTimestamp', Date.now().toString());
    } else {
      localStorage.removeItem('guestSession');
      localStorage.removeItem('guestSessionTimestamp');
    }
  };

  const handleLogout = async () => {
    try {
      await authService.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setUser(null);
      setIsGuest(false);
      setShowSignInPopup(false);
      // Clear all session storage
      localStorage.removeItem('user');
      localStorage.removeItem('authToken');
      localStorage.removeItem('guestSession');
      localStorage.removeItem('guestSessionTimestamp');
    }
  };

  const handleContinueAsGuest = async () => {
    try {
      const response = await authService.createGuest();
      if (response.success && response.data) {
        const guestUser = {
          ...response.data,
          isGuest: true
        };
        setUser(guestUser);
        setIsGuest(true);
        setShowSignInPopup(false);
        
        // Save guest session to localStorage
        localStorage.setItem('guestSession', 'true');
        localStorage.setItem('guestSessionTimestamp', Date.now().toString());
      } else {
        console.error('Failed to create guest session:', response.message);
        // Fallback to local guest user
        await handleCreateLocalGuest();
      }
    } catch (error) {
      console.error('Error creating guest session:', error);
      // Fallback to local guest user
      await handleCreateLocalGuest();
    }
  };

  const handleCreateLocalGuest = async () => {
    const guestUser = {
      id: `guest-${Date.now()}`,
      firstName: 'Guest',
      lastName: 'User',
      email: `guest-${Date.now()}@temp.com`,
      createdAt: new Date().toISOString(),
      isEmailVerified: false,
      isGuest: true
    };
    setUser(guestUser);
    setIsGuest(true);
    setShowSignInPopup(false);
    
    // Save guest session to localStorage
    localStorage.setItem('guestSession', 'true');
    localStorage.setItem('guestSessionTimestamp', Date.now().toString());
  };

  const handleRestoreGuestSession = () => {
    const guestUser = {
      id: `guest-${Date.now()}`,
      firstName: 'Guest',
      lastName: 'User',
      email: `guest@temp.com`,
      createdAt: new Date().toISOString(),
      isEmailVerified: false,
      isGuest: true
    };
    setUser(guestUser);
    setIsGuest(true);
    setShowSignInPopup(false);
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
                handleRestoreGuestSession={handleRestoreGuestSession}
              />
            </div>
          </Router>
        </AudioProvider>
      </LanguageProvider>
    </ThemeProvider>
  );
}

export default App;