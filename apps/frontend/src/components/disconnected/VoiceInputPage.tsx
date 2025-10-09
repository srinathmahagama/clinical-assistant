import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mic, MicOff, RotateCcw, Check } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import LoadingSpinner from '../components/LoadingSpinner/LoadingSpinner';
import { useLanguage } from '../contexts/LanguageContext';

interface VoiceInputPageProps {
  onNavigate: (page: string, data?: any) => void;
  onLogout: () => void;
}

const VoiceInputPage: React.FC<VoiceInputPageProps> = ({ onNavigate, onLogout }) => {
  const navigate = useNavigate();
  const { t, language } = useLanguage();
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [showTranscript, setShowTranscript] = useState(false);
  const [recordingError, setRecordingError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = async () => {
    try {
      setRecordingError('');
      setTranscript('');
      setShowTranscript(false);
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      streamRef.current = stream;
      
      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      // Handle data available
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      // Handle recording stop
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await processAudio(audioBlob);
      };
      
      // Start recording
      mediaRecorder.start(1000); // Collect data every second
      setIsRecording(true);
      
    } catch (error) {
      console.error('Recording failed:', error);
      setRecordingError('Failed to access microphone. Please check your permissions and try again.');
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
    
    // Stop all tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    setIsRecording(false);
  };

  const processAudio = async (audioBlob: Blob) => {
    setIsProcessing(true);
    
    try {
      // Send audio to Flask backend for detection only
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      
      const response = await fetch('http://localhost:5000/detect-audio', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const data = await response.json();
        setTranscript(data.transcript || 'Audio processed successfully');
        setShowTranscript(true);
      } else {
        throw new Error('Failed to process audio');
      }
    } catch (error) {
      console.error('Audio processing failed:', error);
      setRecordingError('Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const tryAgain = () => {
    setTranscript('');
    setShowTranscript(false);
    setRecordingError('');
  };

  const useTranscript = async () => {
    setIsLoading(true);
    
    try {
      // Send transcript to Flask backend for assessment creation
      const response = await fetch('http://localhost:5000/create-assessment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          transcript: transcript,
          inputType: 'voice'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const assessment = data.assessment;
        
        // Navigate with processed data from backend
        navigate('/voice-results', { 
          state: { 
            transcript,
            assessment: assessment
          } 
        });
      } else {
        // Fallback to original behavior if backend fails
        navigate('/voice-results', { state: { transcript } });
      }
    } catch (error) {
      console.error('Failed to create assessment:', error);
      // Fallback to original behavior if backend fails
      navigate('/voice-results', { state: { transcript } });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout>
      <Header onLogout={onLogout} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-2xl mx-auto pt-8">
          <BackButton to="/dashboard" />
          
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-4">{t('voiceInput')}</h1>
          </div>

          <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-8 text-center">
            {isLoading ? (
              <div className="py-12">
                <LoadingSpinner size="lg" text="Creating your health assessment..." />
                <p className="text-gray-600 mt-4 text-sm">
                  Please wait while we process your health assessment...
                </p>
              </div>
            ) : (
              <>
                <h2 className="text-2xl font-bold text-gray-800 mb-2">{t('voiceRecording')}</h2>
                <p className="text-gray-600 mb-8">
                  {t('clickToStart')}
                </p>

                {/* Recording Button */}
                <div className="mb-8 flex flex-col items-center">
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isProcessing}
                    className={`w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${
                      isRecording 
                        ? 'bg-red-500 hover:bg-red-600 animate-pulse' 
                        : isProcessing
                          ? 'bg-yellow-500 cursor-not-allowed'
                          : 'bg-[#183172] hover:bg-[#183172]/80'
                    }`}
                  >
                    {isProcessing ? (
                      <div className="w-8 h-8 border-4 border-white border-t-transparent rounded-full animate-spin"></div>
                    ) : isRecording ? (
                      <Mic className="w-12 h-12 text-white" />
                    ) : (
                      <MicOff className="w-12 h-12 text-white" />
                    )}
                  </button>
                  
                  {isProcessing && (
                    <p className="text-yellow-600 text-sm mt-2">Processing audio...</p>
                  )}
                </div>

                {/* Recording Status */}
                {isRecording && (
                  <div className="mb-6 text-center">
                    <p className="text-red-600 font-medium">{t('recording')}</p>
                    <p className="text-gray-600 text-sm mt-2">Speak clearly about your symptoms</p>
                  </div>
                )}

                {/* Error Message */}
                {recordingError && (
                  <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-red-600 text-sm">{recordingError}</p>
                  </div>
                )}

                {/* Transcript Display with Action Buttons */}
                {showTranscript && transcript && (
                  <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                    <h3 className="font-medium text-gray-800 mb-2">{t('whatWeHeard')}</h3>
                    <div className="bg-white p-3 rounded border mb-4">
                      <p className="text-gray-700 text-sm italic">"{transcript}"</p>
                    </div>
                    
                    {/* Action Buttons */}
                    <div className="flex gap-3 justify-center">
                      <button
                        onClick={tryAgain}
                        className="flex items-center gap-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                      >
                        <RotateCcw className="w-4 h-4" />
                        {t('tryAgain')}
                      </button>
                      <button
                        onClick={useTranscript}
                        className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-[#183172] transition-colors"
                      >
                        <Check className="w-4 h-4" />
                        {t('useThis')}
                      </button>
                    </div>
                  </div>
                )}

                {/* Instructions */}
                <div className="text-gray-600 text-sm">
                  <p className="mb-2">Speak clearly about how you're feeling.</p>
                  <p>Include when symptoms started and how severe they are.</p>
                  <div className="mt-3 p-3 bg-blue-50 rounded-lg">
                    <p className="text-blue-800 text-xs font-medium mb-1">💡 Tips for better voice detection:</p>
                    <ul className="text-blue-700 text-xs space-y-1">
                      <li>• Speak clearly and at a normal pace</li>
                      <li>• Reduce background noise</li>
                      <li>• Hold the microphone close to your mouth</li>
                      <li>• Speak for at least 3-5 seconds</li>
                      <li>• Use Chrome browser for best results</li>
                    </ul>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default VoiceInputPage;