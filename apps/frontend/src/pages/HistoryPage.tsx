import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Eye, Download, Trash2, Clock, Save, FileText, X, CheckCircle } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import { Assessment } from '../types';
import { assessmentService } from '../services/api';

interface HistoryPageProps {
  onNavigate: (page: string) => void;
  onLogout: () => void;
}

const HistoryPage: React.FC<HistoryPageProps> = ({ onNavigate, onLogout }) => {
  const navigate = useNavigate();
  const [assessments, setAssessments] = useState<Assessment[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [selectedAssessment, setSelectedAssessment] = useState<Assessment | null>(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [savedAssessments, setSavedAssessments] = useState<string[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  useEffect(() => {
    loadAssessments();
    loadSavedAssessments();
  }, []);

  const loadAssessments = async () => {
    try {
      const response = await assessmentService.getAssessments();
      if (response.success && response.data) {
        setAssessments(response.data);
      }
    } catch (error) {
      console.error('Failed to load assessments:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSavedAssessments = () => {
    const saved = JSON.parse(localStorage.getItem('savedAssessments') || '[]');
    setSavedAssessments(saved);
  };

  const handleSaveAssessment = async (id: string) => {
    try {
      const response = await assessmentService.saveAssessment(id);
      if (response.success) {
        const newSaved = [...savedAssessments, id];
        setSavedAssessments(newSaved);
        localStorage.setItem('savedAssessments', JSON.stringify(newSaved));
      }
    } catch (error) {
      console.error('Failed to save assessment:', error);
    }
  };

  const handleDeleteAssessment = async (id: string) => {
    setDeletingId(id);
    try {
      const response = await assessmentService.deleteAssessment(id);
      if (response.success) {
        setAssessments(prev => prev.filter(a => a.id !== id));
        // Remove from saved assessments if it was saved
        const newSaved = savedAssessments.filter(savedId => savedId !== id);
        setSavedAssessments(newSaved);
        localStorage.setItem('savedAssessments', JSON.stringify(newSaved));
      }
    } catch (error) {
      console.error('Failed to delete assessment:', error);
    } finally {
      setDeletingId(null);
    }
  };

  const handleDownloadAssessment = async (id: string, format: 'pdf' | 'json' = 'pdf') => {
    setDownloadingId(id);
    try {
      const response = await assessmentService.downloadAssessment(id, format);
      if (response.success && response.data) {
        // Create download link
        const link = document.createElement('a');
        link.href = response.data.downloadUrl;
        link.download = `assessment-${id}.${format === 'json' ? 'json' : 'txt'}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Update download count
        const currentCount = parseInt(localStorage.getItem(`downloadCount_${id}`) || '0');
        localStorage.setItem(`downloadCount_${id}`, (currentCount + 1).toString());
      }
    } catch (error) {
      console.error('Failed to download assessment:', error);
    } finally {
      setDownloadingId(null);
    }
  };

  const handleViewDetails = async (assessment: Assessment) => {
    setSelectedAssessment(assessment);
    setShowDetailsModal(true);
  };

  const filteredAssessments = assessments.filter(assessment =>
    assessment.symptoms.some(symptom => 
      symptom.toLowerCase().includes(searchTerm.toLowerCase())
    ) || assessment.date.includes(searchTerm)
  );

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'High': return 'text-red-600 bg-red-50';
      case 'Moderate': return 'text-orange-600 bg-orange-50';
      case 'Low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <Layout showLanguageButton={false}>
      <Header onLogout={onLogout} showLanguage={false} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-4xl mx-auto">
          {/* Page Header */}
          <div className="mb-8">
            <BackButton to="/dashboard" text="Home" />
            <div className="text-center mt-4">
              <h1 className="text-3xl font-bold text-white mb-2">Health History</h1>
              <p className="text-white/80">Your past health assessments</p>
            </div>
          </div>
          {/* Search */}
          <div className="mb-8">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search your assessment"
                className="w-full pl-12 pr-4 py-4 bg-white rounded-lg border-0 focus:ring-2 focus:ring-blue-500 focus:outline-none text-gray-700 placeholder-gray-500 shadow-lg"
              />
            </div>
          </div>

            {/* Loading */}
            {isLoading && (
              <div className="text-center py-8">
                <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto"></div>
                <p className="text-white mt-2">Loading your history...</p>
              </div>
            )}

          {/* Assessment List */}
          <div className="space-y-6">
            {filteredAssessments.map((assessment) => (
              <div key={assessment.id} className="bg-white rounded-xl p-6 shadow-lg">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center">
                    <Clock className="w-5 h-5 text-gray-500 mr-3" />
                    <div>
                      <p className="font-semibold text-gray-800 text-lg">{assessment.date}</p>
                      <p className="text-sm text-gray-600">{assessment.time}</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button 
                      onClick={() => handleViewDetails(assessment)}
                      className="p-2 text-gray-500 hover:text-blue-600 transition-colors"
                      title="View Details"
                    >
                      <Eye className="w-5 h-5" />
                    </button>
                    
                    <button 
                      onClick={() => handleSaveAssessment(assessment.id)}
                      disabled={savedAssessments.includes(assessment.id)}
                      className={`p-2 transition-colors ${
                        savedAssessments.includes(assessment.id)
                          ? 'text-green-600 cursor-not-allowed'
                          : 'text-gray-500 hover:text-green-600'
                      }`}
                      title={savedAssessments.includes(assessment.id) ? "Already Saved" : "Save Assessment"}
                    >
                      {savedAssessments.includes(assessment.id) ? 
                        <CheckCircle className="w-5 h-5" /> : 
                        <Save className="w-5 h-5" />
                      }
                    </button>
                    
                    <div className="relative group">
                      <button 
                        onClick={() => handleDownloadAssessment(assessment.id)}
                        disabled={downloadingId === assessment.id}
                        className="p-2 text-gray-500 hover:text-green-600 transition-colors disabled:opacity-50"
                        title="Download Assessment"
                      >
                        {downloadingId === assessment.id ? (
                          <div className="w-5 h-5 border-2 border-green-600 border-t-transparent rounded-full animate-spin"></div>
                        ) : (
                          <Download className="w-5 h-5" />
                        )}
                      </button>
                      
                      {/* Download format dropdown */}
                      <div className="absolute right-0 top-full mt-1 bg-white rounded-lg shadow-lg border border-gray-200 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                        <button
                          onClick={() => handleDownloadAssessment(assessment.id, 'pdf')}
                          className="w-full px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center"
                        >
                          <FileText className="w-4 h-4 mr-2" />
                          Download as PDF
                        </button>
                        <button
                          onClick={() => handleDownloadAssessment(assessment.id, 'json')}
                          className="w-full px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center"
                        >
                          <FileText className="w-4 h-4 mr-2" />
                          Download as JSON
                        </button>
                      </div>
                    </div>
                    
                    <button 
                      onClick={() => handleDeleteAssessment(assessment.id)}
                      disabled={deletingId === assessment.id}
                      className="p-2 text-gray-500 hover:text-red-600 transition-colors disabled:opacity-50"
                      title="Delete Assessment"
                    >
                      {deletingId === assessment.id ? (
                        <div className="w-5 h-5 border-2 border-red-600 border-t-transparent rounded-full animate-spin"></div>
                      ) : (
                        <Trash2 className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                </div>

                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-2 font-medium">Symptoms:</p>
                  <p className="text-gray-800 text-lg">{assessment.symptoms.join(', ')}</p>
                </div>

                <div className="mb-4">
                  <p className="text-sm text-gray-600 mb-2 font-medium">Primary Concerns:</p>
                  <p className="text-gray-800 text-lg">{assessment.primaryConcerns.join(', ')}</p>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="text-sm text-gray-600 mr-3 font-medium">Severity:</span>
                    <span className={`px-4 py-2 rounded-lg text-sm font-semibold ${getSeverityColor(assessment.severity)}`}>
                      {assessment.severity}
                    </span>
                  </div>
                  <button 
                    onClick={() => handleViewDetails(assessment)}
                    className="text-[#183172] hover:text-[#183172]/80 text-sm font-semibold flex items-center transition-colors"
                  >
                    View Details →
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* Empty State */}
          {!isLoading && filteredAssessments.length === 0 && (
            <div className="text-center py-12">
              <Clock className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-xl font-medium text-white mb-2">
                {searchTerm ? 'No matching assessments' : 'No assessments yet'}
              </h3>
              <p className="text-white/80 mb-6">
                {searchTerm 
                  ? 'Try adjusting your search terms'
                  : 'Start your first health assessment to see your history here'
                }
              </p>
              {!searchTerm && (
                <button
                  onClick={() => navigate('/dashboard')}
                  className="bg-gray-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-[#183172] transition-colors"
                >
                  Start Assessment
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Assessment Details Modal */}
      {showDetailsModal && selectedAssessment && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
            <div className="p-8">
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-8 pb-4 border-b border-gray-200">
                <div>
                  <h2 className="text-3xl font-bold text-gray-800">Assessment Details</h2>
                  <p className="text-gray-600 mt-1">Complete health assessment information</p>
                </div>
                <button
                  onClick={() => setShowDetailsModal(false)}
                  className="p-3 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-full transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Assessment Content */}
              <div className="space-y-8">
                {/* Basic Info */}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-100">
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                    <Clock className="w-5 h-5 mr-2 text-blue-600" />
                    Assessment Information
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <p className="text-sm text-gray-600 mb-1">Assessment Date</p>
                      <p className="text-lg font-semibold text-gray-800">{selectedAssessment.date}</p>
                    </div>
                    <div className="bg-white rounded-lg p-4 shadow-sm">
                      <p className="text-sm text-gray-600 mb-1">Assessment Time</p>
                      <p className="text-lg font-semibold text-gray-800">{selectedAssessment.time}</p>
                    </div>
                  </div>
                </div>

                {/* Symptoms */}
                <div className="bg-gradient-to-r from-red-50 to-pink-50 rounded-xl p-6 border border-red-100">
                  <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                    <span className="w-8 h-8 bg-red-100 text-red-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">
                      🏥
                    </span>
                    Reported Symptoms
                  </h3>
                  <div className="flex flex-wrap gap-3">
                    {selectedAssessment.symptoms.map((symptom, index) => (
                      <span
                        key={index}
                        className="bg-white text-red-700 px-4 py-2 rounded-full text-sm font-medium shadow-sm border border-red-200"
                      >
                        {symptom}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Severity and Primary Concerns */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Severity */}
                  <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl p-6 border border-yellow-100">
                    <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                      <span className="w-8 h-8 bg-yellow-100 text-yellow-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">
                        ⚠️
                      </span>
                      Severity Level
                    </h3>
                    <div className="flex items-center">
                      <span className={`px-6 py-3 rounded-xl text-lg font-bold ${getSeverityColor(selectedAssessment.severity)}`}>
                        {selectedAssessment.severity}
                      </span>
                    </div>
                  </div>

                  {/* Primary Concerns */}
                  <div className="bg-gradient-to-r from-orange-50 to-red-50 rounded-xl p-6 border border-orange-100">
                    <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                      <span className="w-8 h-8 bg-orange-100 text-orange-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">
                        🎯
                      </span>
                      Primary Concerns
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedAssessment.primaryConcerns.map((concern, index) => (
                        <span
                          key={index}
                          className="bg-white text-orange-700 px-4 py-2 rounded-full text-sm font-medium shadow-sm border border-orange-200"
                        >
                          {concern}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-6 border border-green-100">
                  <h3 className="text-xl font-bold text-gray-800 mb-6 flex items-center">
                    <span className="w-8 h-8 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">
                      💡
                    </span>
                    Health Recommendations
                  </h3>
                  <div className="space-y-4">
                    {selectedAssessment.recommendations.map((recommendation, index) => (
                      <div key={index} className="flex items-start bg-white rounded-lg p-4 shadow-sm border border-green-200">
                        <span className="bg-green-100 text-green-800 rounded-full w-8 h-8 flex items-center justify-center text-sm font-bold mr-4 mt-0.5 flex-shrink-0">
                          {index + 1}
                        </span>
                        <p className="text-gray-700 font-medium leading-relaxed">{recommendation}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Modal Actions */}
              <div className="flex flex-col sm:flex-row gap-4 mt-8 pt-6 border-t border-gray-200">
                <button
                  onClick={() => handleSaveAssessment(selectedAssessment.id)}
                  disabled={savedAssessments.includes(selectedAssessment.id)}
                  className={`flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
                    savedAssessments.includes(selectedAssessment.id)
                      ? 'bg-green-100 text-green-800 cursor-not-allowed border-2 border-green-200'
                      : 'bg-green-600 text-white hover:bg-green-700 hover:shadow-lg transform hover:-translate-y-0.5'
                  }`}
                >
                  {savedAssessments.includes(selectedAssessment.id) ? (
                    <>
                      <CheckCircle className="w-5 h-5" />
                      Assessment Saved
                    </>
                  ) : (
                    <>
                      <Save className="w-5 h-5" />
                      Save Assessment
                    </>
                  )}
                </button>
                
                <button
                  onClick={() => handleDownloadAssessment(selectedAssessment.id)}
                  className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 hover:shadow-lg transform hover:-translate-y-0.5 transition-all"
                >
                  <Download className="w-5 h-5" />
                  Download Report
                </button>
                
                <button
                  onClick={() => setShowDetailsModal(false)}
                  className="flex items-center justify-center gap-2 px-6 py-3 bg-gray-200 text-gray-800 rounded-xl font-semibold hover:bg-gray-300 hover:shadow-lg transform hover:-translate-y-0.5 transition-all"
                >
                  <X className="w-5 h-5" />
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </Layout>
  );
};

export default HistoryPage;