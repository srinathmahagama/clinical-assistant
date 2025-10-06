import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { LanguageProvider } from './contexts/LanguageContext';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import SignUpPage from './pages/SignUpPage';
import DashboardPage from './pages/DashboardPage';
import VoiceInputPage from './pages/VoiceInputPage';
import VoiceResultsPage from './pages/VoiceResultsPage';
import TextInputPage from './pages/TextInputPage';
import TextResultsPage from './pages/TextResultsPage';
import AssistantPage from './pages/AssistantPage';
import HistoryPage from './pages/HistoryPage';
import { User } from './types';
import { authService } from './services/api';

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [pageData, setPageData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is already logged in
    const currentUser = authService.getCurrentUser();
    if (currentUser) {
      setUser(currentUser);
    }
    setIsLoading(false);
  }, []);

  const handleLogin = (userData: User) => {
    setUser(userData);
  };

  const handleLogout = () => {
    authService.logout();
    setUser(null);
    // Redirect to home page after logout
    window.location.href = '/';
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
    <LanguageProvider>
      <Router>
        <div className="App">
          <Routes>
          <Route path="/" element={<HomePage onNavigate={handleNavigate} />} />
          <Route path="/login" element={<LoginPage onNavigate={handleNavigate} onLogin={handleLogin} />} />
          <Route path="/signup" element={<SignUpPage onNavigate={handleNavigate} onLogin={handleLogin} />} />
          <Route 
            path="/dashboard" 
            element={
              user ? (
                <DashboardPage user={user} onNavigate={handleNavigate} onLogout={handleLogout} />
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          <Route 
            path="/voice-input" 
            element={
              user ? (
                <VoiceInputPage onNavigate={handleNavigate} onLogout={handleLogout} />
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          <Route 
            path="/voice-results" 
            element={
              user ? (
                <VoiceResultsPage onNavigate={handleNavigate} onLogout={handleLogout} transcript={pageData?.transcript} />
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          <Route 
            path="/text-input" 
            element={
              user ? (
                <TextInputPage onNavigate={handleNavigate} onLogout={handleLogout} />
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          <Route 
            path="/text-results" 
            element={
              user ? (
                <TextResultsPage onNavigate={handleNavigate} onLogout={handleLogout} />
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          <Route 
            path="/assistant" 
            element={
              user ? (
                <AssistantPage onNavigate={handleNavigate} onLogout={handleLogout} />
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          <Route 
            path="/history" 
            element={
              user ? (
                <HistoryPage onNavigate={handleNavigate} onLogout={handleLogout} />
              ) : (
                <Navigate to="/login" replace />
              )
            } 
          />
          </Routes>
        </div>
      </Router>
    </LanguageProvider>
  );
}

export default App;