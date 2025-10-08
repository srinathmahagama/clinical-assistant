import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Calendar, Eye, Phone, MessageCircle, Save, CheckCircle } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import { assessmentService } from '../services/api';

interface TextResultsPageProps {
  onNavigate: (page: string) => void;
  onLogout: () => void;
}

const TextResultsPage: React.FC<TextResultsPageProps> = ({ onNavigate, onLogout }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const assessment = location.state?.assessment;
  const [isSaving, setIsSaving] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [saveError, setSaveError] = useState('');

  const handleSaveAssessment = async () => {
    if (isSaved) return;
    
    setIsSaving(true);
    setSaveError('');
    
    try {
      // Save the assessment
      const saveResponse = await assessmentService.saveAssessment(assessment.id);
      if (saveResponse.success) {
        setIsSaved(true);
      } else {
        setSaveError(saveResponse.message || 'Failed to save assessment');
      }
    } catch (error) {
      console.error('Save assessment error:', error);
      setSaveError('Failed to save assessment. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  if (!assessment) {
    return (
      <Layout>
        <Header onLogout={onLogout} />
        <div className="min-h-screen p-4 pt-20">
          <div className="max-w-2xl mx-auto pt-8">
            <BackButton to="/text-input" />
            <div className="text-center">
              <h1 className="text-3xl font-bold text-white mb-4">No Assessment Found</h1>
              <p className="text-white/90 mb-8">Please go back and create an assessment.</p>
              <button
                onClick={() => navigate('/text-input')}
                className="bg-[#183172] text-white px-6 py-3 rounded-lg font-medium hover:bg-[#183172]/80 transition-colors"
              >
                Create Assessment
              </button>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <Header onLogout={onLogout} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-2xl mx-auto pt-8">
          <BackButton to="/text-input" />
          
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-4">Text Assessment Results</h1>
          </div>

          <div className="space-y-6">
            {/* Assessment Results */}
            <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-6">
              <div className="text-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800 mb-2">Your Health Assessment</h2>
                <p className="text-gray-600">Based on the symptoms you selected</p>
              </div>

              {/* Severity Level */}
              <div className={`bg-white rounded-xl p-6 mb-6 border-l-4 ${
                assessment.severity === 'High' ? 'border-red-400' : 
                assessment.severity === 'Moderate' ? 'border-orange-400' : 'border-green-400'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-gray-800">
                    {assessment.severity} - {assessment.severity === 'High' ? 'Seek Immediate Care' : 
                     assessment.severity === 'Moderate' ? 'See Healthcare Provider' : 'Monitor Symptoms'}
                  </h3>
                  <span className="text-sm text-gray-500">AI Generated</span>
                </div>
                <p className="text-gray-700 mb-4">
                  {assessment.severity === 'High' ? 
                    'Your symptoms suggest a high severity condition. Please seek immediate medical attention.' :
                    assessment.severity === 'Moderate' ?
                    'We recommend speaking with a healthcare provider to better understand your symptoms.' :
                    'Your symptoms appear to be mild. Monitor them closely and seek care if they worsen.'
                  }
                </p>
                <div>
                  <p className="font-medium text-gray-800 mb-1">Primary Concerns:</p>
                  <p className="text-gray-600">{assessment.primaryConcerns?.join(', ') || 'General symptoms'}</p>
                </div>
              </div>

              {/* Symptoms */}
              <div className="bg-gradient-to-r from-red-50 to-pink-50 rounded-xl p-6 mb-6 border border-red-100">
                <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                  <span className="w-8 h-8 bg-red-100 text-red-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">
                    🏥
                  </span>
                  Reported Symptoms
                </h3>
                <div className="flex flex-wrap gap-3">
                  {assessment.symptoms.map((symptom, index) => (
                    <span
                      key={index}
                      className="bg-white text-red-700 px-4 py-2 rounded-full text-sm font-medium shadow-sm border border-red-200"
                    >
                      {symptom}
                    </span>
                  ))}
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
                  {assessment.recommendations.map((recommendation, index) => (
                    <div key={index} className="flex items-start">
                      <div className={`rounded-full p-2 mr-4 mt-1 ${
                        index === 0 ? 'bg-blue-100' : 
                        index === 1 ? 'bg-green-100' : 'bg-red-100'
                      }`}>
                        {index === 0 ? <Calendar className="w-4 h-4 text-blue-600" /> :
                         index === 1 ? <Eye className="w-4 h-4 text-green-600" /> :
                         <Phone className="w-4 h-4 text-red-600" />}
                      </div>
                      <div>
                        <p className="font-medium text-gray-800">PRIORITY {index + 1}</p>
                        <p className="text-gray-700">{recommendation}</p>
                        <p className="text-sm text-gray-500">
                          {index === 0 ? 'Within 1-2 days' : 
                           index === 1 ? 'Ongoing' : 'If symptoms worsen'}
                        </p>
                      </div>
                    </div>
                  ))}
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
                  onClick={() => navigate('/dashboard')}
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

export default TextResultsPage;
