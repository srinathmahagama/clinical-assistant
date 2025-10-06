import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Calendar, Eye, Phone, MessageCircle, Save, CheckCircle } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import { assessmentService } from '../services/api';

interface VoiceResultsPageProps {
  onNavigate: (page: string) => void;
  onLogout: () => void;
  transcript?: string;
}

const VoiceResultsPage: React.FC<VoiceResultsPageProps> = ({ onNavigate, onLogout, transcript }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const transcriptData = transcript || location.state?.transcript;
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [saveError, setSaveError] = useState('');

  const handleSaveAssessment = async () => {
    if (isSaved) return;
    
    setIsSaving(true);
    setSaveError('');
    
    try {
      // Create a new assessment from the current transcript
      const symptoms = transcriptData ? transcriptData.split(' ').filter(word => 
        ['headache', 'fever', 'cough', 'pain', 'nausea', 'dizziness', 'fatigue', 'weakness', 'sore', 'throat', 'runny', 'nose'].includes(word.toLowerCase())
      ) : ['General symptoms'];
      
      const response = await assessmentService.createAssessment('voice', transcriptData, symptoms, transcriptData);
      if (response.success && response.data) {
        // Now save the created assessment
        const saveResponse = await assessmentService.saveAssessment(response.data.id);
        if (saveResponse.success) {
          setIsSaved(true);
        } else {
          setSaveError(saveResponse.message || 'Failed to save assessment');
        }
      } else {
        setSaveError(response.message || 'Failed to create assessment');
      }
    } catch (error) {
      console.error('Save assessment error:', error);
      setSaveError('Failed to save assessment. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Layout>
      <Header onLogout={onLogout} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-2xl mx-auto pt-8">
          <BackButton to="/voice-input" />
          
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-4">Voice Results</h1>
          </div>

          <div className="space-y-6">
            {/* Transcript Display */}
            {transcriptData && (
              <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-6">
                <h2 className="text-xl font-bold text-gray-800 mb-4">What You Said</h2>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-gray-700 italic">"{transcriptData}"</p>
                </div>
              </div>
            )}

            {/* Assessment Results */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-6">
              <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Your Health Assessment</h2>
                <p className="text-gray-600">Based on the symptoms you shared</p>
              </div>

              {/* Severity Level */}
              <div className="bg-white rounded-xl p-6 mb-6 border-l-4 border-orange-400">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-gray-800">Moderate - See Healthcare Provider</h3>
                  <span className="text-sm text-gray-500">Low Confidence</span>
                </div>
                <p className="text-gray-700 mb-4">
                  We recommend speaking with a healthcare provider to better understand your symptoms.
                </p>
                <div>
                  <p className="font-medium text-gray-800 mb-1">Primary Concerns:</p>
                  <p className="text-gray-600">unclear symptoms</p>
                </div>
              </div>

              {/* Ask Assistant */}
              <div className="bg-blue-50 rounded-xl p-4 mb-6 flex items-center justify-between">
                <div className="flex items-center">
                  <MessageCircle className="w-8 h-8 text-blue-600 mr-3" />
                  <div>
                    <p className="font-medium text-gray-800">Have Questions About Your Results?</p>
                    <p className="text-sm text-gray-600">Ask our health assistant for more information and guidance</p>
                  </div>
                </div>
                <button
                  onClick={() => navigate('/assistant')}
                  className="bg-gray-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-[#183172] transition-colors"
                >
                  Ask Assistant
                </button>
              </div>

              {/* Recommendations */}
              <div className="bg-white rounded-xl p-6">
                <h3 className="font-bold text-gray-800 mb-4">What You Should Do</h3>
                <p className="text-gray-600 mb-4">Follow these recommendations in order of priority</p>
                
                <div className="space-y-4">
                  <div className="flex items-start">
                    <div className="bg-blue-100 rounded-full p-2 mr-4 mt-1">
                      <Calendar className="w-4 h-4 text-blue-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-800">PRIORITY 1</p>
                      <p className="text-gray-700">Contact your healthcare provider for evaluation</p>
                      <p className="text-sm text-gray-500">Within 1-2 days</p>
                    </div>
                  </div>

                  <div className="flex items-start">
                    <div className="bg-green-100 rounded-full p-2 mr-4 mt-1">
                      <Eye className="w-4 h-4 text-green-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-800">PRIORITY 2</p>
                      <p className="text-gray-700">Keep track of your symptoms and how you're feeling</p>
                      <p className="text-sm text-gray-500">Ongoing</p>
                    </div>
                  </div>

                  <div className="flex items-start">
                    <div className="bg-red-100 rounded-full p-2 mr-4 mt-1">
                      <Phone className="w-4 h-4 text-red-600" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-800">PRIORITY 3</p>
                      <p className="text-gray-700">Seek immediate care if symptoms become severe</p>
                      <p className="text-sm text-gray-500">If symptoms worsen</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Important Reminders */}
              <div className="bg-purple-50 rounded-xl p-4 mt-6">
                <h4 className="font-bold text-purple-800 mb-2">Important Reminders</h4>
                <ul className="text-sm text-purple-700 space-y-1">
                  <li>• This assessment is for guidance only and does not replace professional medical advice</li>
                  <li>• If your symptoms worsen or you develop new concerning symptoms, seek medical care immediately</li>
                  <li>• Always consult with a healthcare provider for proper diagnosis and treatment</li>
                </ul>
              </div>

              {/* Save Error Message */}
              {saveError && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-600 text-sm">{saveError}</p>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4 mt-6">
                <button 
                  onClick={handleSaveAssessment}
                  disabled={isSaving || isSaved}
                  className={`flex-1 py-3 rounded-lg font-medium transition-colors flex items-center justify-center gap-2 ${
                    isSaved 
                      ? 'bg-green-100 text-green-800 cursor-not-allowed border-2 border-green-200'
                      : isSaving
                        ? 'bg-gray-400 text-white cursor-not-allowed'
                        : 'bg-gray-600 text-white hover:bg-[#183172]'
                  }`}
                >
                  {isSaving ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Saving...
                    </>
                  ) : isSaved ? (
                    <>
                      <CheckCircle className="w-4 h-4" />
                      Assessment Saved
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      Save this Assessment
                    </>
                  )}
                </button>
                <button
                  onClick={() => onNavigate('dashboard')}
                  className="flex-1 bg-gray-200 text-gray-800 py-3 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                >
                  New Assessment
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default VoiceResultsPage;