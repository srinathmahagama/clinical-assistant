import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  className?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'md', 
  text = 'Processing...', 
  className = '' 
}) => {
  const sizeClasses = {
    sm: 'w-6 h-6',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  return (
    <div className={`flex flex-col items-center justify-center space-y-4 ${className}`}>
      <div className="relative">
        {/* Outer ring */}
        <div className={`${sizeClasses[size]} border-4 border-blue-200 rounded-full animate-spin`}></div>
        {/* Inner ring */}
        <div className={`absolute top-0 left-0 ${sizeClasses[size]} border-4 border-transparent border-t-blue-600 rounded-full animate-spin`}></div>
      </div>
      {text && (
        <p className="text-white font-medium text-center">{text}</p>
      )}
    </div>
  );
};

export default LoadingSpinner;
