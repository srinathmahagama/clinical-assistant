import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Send, Mic, MicOff, Bot } from 'lucide-react';
import Layout from '../components/Layout/Layout';
import Header from '../components/Header/Header';
import BackButton from '../components/BackButton/BackButton';
import { useLanguage } from '../contexts/LanguageContext';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string;
}

interface AssistantPageProps {
  onNavigate: (page: string) => void;
  onLogout: () => void;
}

const AssistantPage: React.FC<AssistantPageProps> = ({ onNavigate, onLogout }) => {
  const navigate = useNavigate();
  const { language } = useLanguage();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
      isUser: false,
      timestamp: '10:00 p.m'
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingError, setRecordingError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    // Stop microphone if currently recording
    if (isRecording && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText,
      isUser: true,
      timestamp: new Date().toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
      })
    };

    setMessages(prev => [...prev, userMessage]);
    const messageText = inputText;
    setInputText('');
    setIsLoading(true);

    try {
      // Send message to Flask backend
      const response = await fetch('http://localhost:5000/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText })
      });
      
      if (response.ok) {
        const data = await response.json();
        const aiMessage: Message = {
          id: data.response.id,
          text: data.response.text,
          isUser: false,
          timestamp: data.response.timestamp,
          createdAt: data.response.createdAt
        };
        
        setMessages(prev => [...prev, aiMessage]);
      } else {
        // Fallback to mock response if backend fails
        const responses = [
          "Based on your symptoms, I recommend monitoring them closely. If they persist or worsen, please consult with a healthcare provider.",
          "That's a good question about your health. While I can provide general guidance, it's always best to discuss specific concerns with your doctor.",
          "I understand your concern. Let me help you understand what these symptoms might indicate and when you should seek medical attention.",
          "Thank you for sharing that information. Based on what you've told me, here are some general recommendations..."
        ];

        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          text: responses[Math.floor(Math.random() * responses.length)],
          isUser: false,
          timestamp: new Date().toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
          })
        };

        setMessages(prev => [...prev, aiMessage]);
      }
    } catch (error) {
      console.error('Chat message failed:', error);
      // Fallback to mock response if backend fails
      const responses = [
        "Based on your symptoms, I recommend monitoring them closely. If they persist or worsen, please consult with a healthcare provider.",
        "That's a good question about your health. While I can provide general guidance, it's always best to discuss specific concerns with your doctor.",
        "I understand your concern. Let me help you understand what these symptoms might indicate and when you should seek medical attention.",
        "Thank you for sharing that information. Based on what you've told me, here are some general recommendations..."
      ];

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: responses[Math.floor(Math.random() * responses.length)],
        isUser: false,
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour: 'numeric', 
          minute: '2-digit',
          hour12: true 
        })
      };

      setMessages(prev => [...prev, aiMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const startVoiceRecording = async () => {
    try {
      setRecordingError('');
      
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

  const stopVoiceRecording = () => {
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
        const transcript = data.transcript || 'Audio processed successfully';
        // Add transcript to input field (don't send automatically)
        setInputText(prev => prev + transcript + ' ');
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

  return (
    <Layout showLanguageButton={false}>
      <Header onLogout={onLogout} showLanguage={false} />
      <div className="min-h-screen p-4 pt-20">
        <div className="max-w-4xl mx-auto">
          <BackButton to="/dashboard" text="Home" />
          <div className="flex items-center justify-center">
        {/* Chat Container */}
        <div className="w-full max-w-2xl bg-white rounded-lg shadow-lg overflow-hidden" style={{ height: '600px' }}>
          {/* Chat Header */}
          <div className="bg-gray-200 p-4 border-b border-gray-300">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-[#183172] rounded-full flex items-center justify-center mr-3">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-lg font-bold text-gray-800">Health Assistant</h2>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4" style={{ height: '400px' }}>
            <div className="space-y-2">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-xs px-3 py-2 rounded-lg ${
                    message.isUser
                      ? 'bg-[#183172] text-white'
                      : 'bg-gray-200 text-gray-800'
                  }`}>
                    {!message.isUser && (
                      <div className="flex items-center mb-2">
                        <div className="w-6 h-6 bg-[#183172] rounded-full flex items-center justify-center mr-2">
                          <Bot className="w-4 h-4 text-white" />
                        </div>
                        <span className="text-xs text-gray-500">{message.timestamp}</span>
                      </div>
                    )}
                    <p className="text-sm">{message.text}</p>
                    {message.isUser && (
                      <div className="text-xs text-blue-100 mt-1 text-right">
                        {message.timestamp}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-200 text-gray-800 px-4 py-3 rounded-lg">
                    <div className="flex items-center">
                      <div className="w-6 h-6 bg-[#183172] rounded-full flex items-center justify-center mr-2">
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Input */}
          <div className="bg-gray-200 p-3 border-t border-gray-300" style={{ height: '80px' }}>
            {/* Error Message */}
            {recordingError && (
              <div className="mb-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-600 text-sm">{recordingError}</p>
              </div>
            )}
            
            <div className="flex items-center space-x-2">
              <div className="flex-1 relative">
                <input
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about your health............"
                  className="w-full px-3 py-2 bg-gray-100 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors pr-12 text-sm"
                />
                <button 
                  onClick={isRecording ? stopVoiceRecording : startVoiceRecording}
                  disabled={isProcessing}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 transition-colors ${
                    isRecording 
                      ? 'text-red-500 hover:text-red-700' 
                      : isProcessing
                        ? 'text-yellow-500 cursor-not-allowed'
                        : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {isProcessing ? (
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                  ) : isRecording ? (
                    <Mic className="w-4 h-4" />
                  ) : (
                    <MicOff className="w-4 h-4" />
                  )}
                </button>
              </div>
              <button
                onClick={sendMessage}
                disabled={!inputText.trim() || isLoading}
                className="bg-gray-400 text-white p-2 rounded-lg hover:bg-gray-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            
            {/* Recording Status */}
            {isRecording && (
              <div className="mt-2 text-center">
                <p className="text-red-600 text-sm font-medium">Recording... Speak clearly</p>
              </div>
            )}
            
            {isProcessing && (
              <div className="mt-2 text-center">
                <p className="text-yellow-600 text-sm font-medium">Processing audio...</p>
              </div>
            )}
            
          </div>
        </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default AssistantPage;