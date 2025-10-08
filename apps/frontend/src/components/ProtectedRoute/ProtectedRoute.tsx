import React from 'react';
import { Navigate } from 'react-router-dom';
import { User } from '../../types';

interface ProtectedRouteProps {
  user: User | null;
  isGuest?: boolean;
  children: React.ReactNode;
  redirectTo?: string;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ 
  user, 
  isGuest = false, 
  children, 
  redirectTo = '/login' 
}) => {
  // Check if user is authenticated (either registered user or guest)
  if (!user) {
    return <Navigate to={redirectTo} replace />;
  }

  // User is authenticated, render the protected content
  return <>{children}</>;
};

export default ProtectedRoute;
