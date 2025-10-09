import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Check, RotateCcw } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import LoadingSpinner from '../components/LoadingSpinner/LoadingSpinner';
import { useLanguage } from '../contexts/LanguageContext';

interface TextInputPageProps {
  onNavigate: (page: string, data?: any) => void;
  onLogout: () => void;
}

const TextInputPage: React.FC<TextInputPageProps> = ({ onNavigate, onLogout }) => {
  const navigate = useNavigate();
  const { t, language } = useLanguage();
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [additionalInfo, setAdditionalInfo] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [manualSymptom, setManualSymptom] = useState('');

  const symptoms = [
    'Fever',
    'Cough',
    'Headache',
    'Runny Nose',
    'Stomach Pain',
    'Nausea',
    'Fatigue',
    'Joint Pain',
    'Chest Pain',
    'Shortness of Breath',
    'Dizziness',
    'Sore Throat'
  ];

  const toggleSymptom = (symptom: string) => {
    setSelectedSymptoms(prev => 
      prev.includes(symptom) 
        ? prev.filter(s => s !== symptom)
        : [...prev, symptom]
    );
  };

  const addManualSymptom = () => {
    if (manualSymptom.trim() && !selectedSymptoms.includes(manualSymptom.trim())) {
      setSelectedSymptoms(prev => [...prev, manualSymptom.trim()]);
      setManualSymptom('');
    }
  };

  const removeSymptom = (symptom: string) => {
    setSelectedSymptoms(prev => prev.filter(s => s !== symptom));
  };

  const handleSubmit = async () => {
    if (selectedSymptoms.length === 0) {
      alert('Please select at least one symptom');
      return;
    }

    setIsLoading(true);
    
    try {
      // Send symptoms to Flask backend for assessment creation
      const response = await fetch('http://localhost:5000/create-assessment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          transcript: selectedSymptoms.join(', ') + (additionalInfo ? ' ' + additionalInfo : ''),
          inputType: 'text',
          symptoms: selectedSymptoms,
          additionalInfo: additionalInfo
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const assessment = data.assessment;
        
        // Navigate with processed data from backend
        navigate('/text-results', { 
          state: { 
            assessment: assessment,
            inputType: 'text'
          } 
        });
      } else {
        // Fallback to original behavior if backend fails
        navigate('/dashboard', { 
          state: { 
            selectedSymptoms,
            additionalInfo,
            inputType: 'text'
          } 
        });
      }
    } catch (error) {
      console.error('Failed to create assessment:', error);
      // Fallback to original behavior if backend fails
      navigate('/dashboard', { 
        state: { 
          selectedSymptoms,
          additionalInfo,
          inputType: 'text'
        } 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedSymptoms([]);
    setAdditionalInfo('');
    setManualSymptom('');
  };

  return (
    <Layout>
      <Header onLogout={onLogout} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-2xl mx-auto pt-8">
          <BackButton to="/dashboard" />
          
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-4">{t('textInput')}</h1>
          </div>

          <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-8">
            {isLoading ? (
              <div className="py-12">
                <LoadingSpinner size="lg" text="Creating your health assessment..." />
                <p className="text-gray-600 mt-4 text-sm">
                  Please wait while we process your health assessment...
                </p>
              </div>
            ) : (
              <>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">{t('selectSymptoms')}</h2>
                <p className="text-gray-600 mb-6">
                  {t('selectAllThatApply')}
                </p>

                {/* Symptoms Grid - Restored to 3 columns */}
                <div className="grid grid-cols-3 gap-3 mb-6">
                  {symptoms.map((symptom) => (
                    <button
                      key={symptom}
                      onClick={() => toggleSymptom(symptom)}
                      className={`p-3 rounded-lg border-2 transition-all duration-200 ${
                        selectedSymptoms.includes(symptom)
                          ? 'border-[#183172] bg-[#183172] text-white'
                          : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
                      }`}
                    >
                      <div className="flex items-center justify-center">
                        <span className="text-sm font-medium">{symptom}</span>
                        {selectedSymptoms.includes(symptom) && (
                          <Check className="w-4 h-4 ml-2" />
                        )}
                      </div>
                    </button>
                  ))}
                </div>

                {/* Manual Symptom Input */}
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Add Other Symptoms
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={manualSymptom}
                      onChange={(e) => setManualSymptom(e.target.value)}
                      placeholder="Type a symptom..."
                      className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#183172] focus:border-transparent"
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          addManualSymptom();
                        }
                      }}
                    />
                    <button
                      onClick={addManualSymptom}
                      disabled={!manualSymptom.trim()}
                      className="px-4 py-3 bg-[#183172] text-white rounded-lg hover:bg-[#183172]/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Add
                    </button>
                  </div>
                </div>

                {/* Additional Information */}
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {t('additionalInfo')}
                  </label>
                  <textarea
                    value={additionalInfo}
                    onChange={(e) => setAdditionalInfo(e.target.value)}
                    placeholder={t('additionalInfoPlaceholder')}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#183172] focus:border-transparent resize-none"
                    rows={3}
                  />
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 justify-center">
                  <button
                    onClick={resetForm}
                    className="flex items-center gap-2 px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                  >
                    <RotateCcw className="w-4 h-4" />
                    {t('reset')}
                  </button>
                  <button
                    onClick={handleSubmit}
                    disabled={selectedSymptoms.length === 0}
                    className="flex items-center gap-2 px-6 py-3 bg-[#183172] text-white rounded-lg hover:bg-[#183172]/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Check className="w-4 h-4" />
                    {t('createAssessment')}
                  </button>
                </div>

                {/* Selected Symptoms Summary */}
                {selectedSymptoms.length > 0 && (
                  <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                    <h3 className="text-sm font-medium text-blue-800 mb-2">
                      Selected Symptoms ({selectedSymptoms.length}):
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedSymptoms.map((symptom) => (
                        <span
                          key={symptom}
                          className="px-3 py-1 bg-blue-100 text-blue-800 text-xs rounded-full flex items-center gap-1"
                        >
                          {symptom}
                          <button
                            onClick={() => removeSymptom(symptom)}
                            className="ml-1 text-blue-600 hover:text-blue-800 font-bold"
                          >
                            ×
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default TextInputPage;