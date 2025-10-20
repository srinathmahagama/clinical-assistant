import React from 'react';
import { X, Pill, Heart } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface RecommendationModalProps {
  isOpen: boolean;
  onClose: () => void;
  disease: string;
  recommendations: string[];
  severity?: string;
}

const RecommendationModal: React.FC<RecommendationModalProps> = ({
  isOpen,
  onClose,
  disease,
  recommendations,
  severity
}) => {
  const { theme } = useTheme();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className={`relative w-full max-w-md mx-4 rounded-2xl shadow-2xl ${
        theme === 'dark' 
          ? 'bg-slate-800 border border-amber-600' 
          : 'bg-white border border-amber-300'
      }`}>
        {/* Header */}
        <div className={`p-6 rounded-t-2xl ${
          theme === 'dark' 
            ? 'bg-gradient-to-r from-amber-700 to-emerald-700' 
            : 'bg-gradient-to-r from-amber-400 to-emerald-500'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Pill className="w-6 h-6 text-white" />
              <h2 className="text-xl font-bold text-white">Health Recommendations</h2>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded-full hover:bg-white/20 transition-colors"
            >
              <X className="w-5 h-5 text-white" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className={`p-6 space-y-4 ${
          theme === 'dark' ? 'text-slate-100' : 'text-gray-800'
        }`}>
          <div className="text-center">
            <h3 className="text-lg font-semibold mb-2">For: {disease}</h3>
            {severity && <p className="text-sm text-amber-500">Severity: {severity}</p>}
          </div>

          <div>
            <h4 className={`font-semibold mb-3 flex items-center space-x-2 ${
              theme === 'dark' ? 'text-amber-300' : 'text-amber-600'
            }`}>
              <Heart className="w-4 h-4" />
              <span>Recommended Actions:</span>
            </h4>
            <ul className="space-y-2">
              {recommendations.map((recommendation, index) => (
                <li key={index} className="flex items-start space-x-3 p-3 rounded-lg bg-amber-50 dark:bg-slate-700">
                  <div className="w-2 h-2 rounded-full mt-2 bg-amber-500 dark:bg-amber-400" />
                  <span className="text-sm">{recommendation}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="p-4 border-t bg-gray-50 dark:bg-slate-700">
          <button
            onClick={onClose}
            className="w-full py-2 px-4 bg-amber-500 hover:bg-amber-600 text-white rounded-lg font-semibold"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default RecommendationModal;