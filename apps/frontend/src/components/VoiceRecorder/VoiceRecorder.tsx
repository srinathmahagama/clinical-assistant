import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Send, X, Square, Play, Pause } from 'lucide-react';

interface VoiceRecorderProps {
  onSendVoiceMessage: (audioBlob: Blob, duration: number) => void;
  onCancel: () => void;
  onTranscribedText?: (text: string) => void; // NEW: Add this prop
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  onSendVoiceMessage,
  onCancel,
  onTranscribedText // NEW: Add this prop
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string>('');
  
  // NEW: Add speech-to-text state
  const [transcribedText, setTranscribedText] = useState<string>('');
  const [isTranscribing, setIsTranscribing] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  
  // NEW: Add speech recognition ref
  const recognitionRef = useRef<any>(null);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const startTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    setRecordingTime(0);
    timerRef.current = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  // NEW: Speech recognition functions
  const startSpeechRecognition = () => {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      console.log('Speech recognition not supported');
      return;
    }

    try {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        console.log('🎤 Speech recognition started');
        setIsTranscribing(true);
      };

      recognition.onresult = (event: any) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }

        // Update transcribed text in real-time
        if (interimTranscript || finalTranscript) {
          const newText = finalTranscript || interimTranscript;
          setTranscribedText(newText);
          
          // Send to parent component if provided
          if (onTranscribedText) {
            onTranscribedText(newText);
          }
        }
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        // Don't show error to user - just log it
      };

      recognition.onend = () => {
        console.log('🎤 Speech recognition ended');
        setIsTranscribing(false);
      };

      recognition.start();
      recognitionRef.current = recognition;

    } catch (error) {
      console.error('Failed to start speech recognition:', error);
    }
  };

  const stopSpeechRecognition = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsTranscribing(false);
    }
  };

  const cleanup = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    // NEW: Clean up speech recognition
    stopSpeechRecognition();
    setTranscribedText('');
  };

  const handleStartRecording = async () => {
    try {
      setError('');
      setTranscribedText(''); // NEW: Reset transcribed text
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          setAudioBlob(blob);
          const url = URL.createObjectURL(blob);
          setAudioUrl(url);
        
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        stopTimer();
        // NEW: Stop speech recognition when recording stops
        stopSpeechRecognition();
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      startTimer();
      
      // NEW: Start speech recognition when recording starts
      startSpeechRecognition();
      
    } catch (error) {
      console.error('Failed to start recording:', error);
      setError('Failed to access microphone. Please check permissions.');
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handlePlayPause = () => {
    if (!audioRef.current || !audioUrl) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSend = () => {
    if (audioBlob && recordingTime > 0) {
      onSendVoiceMessage(audioBlob, recordingTime);
      cleanup();
      setAudioBlob(null);
      setAudioUrl('');
      setRecordingTime(0);
      setIsPlaying(false);
      setError('');
      setTranscribedText(''); // NEW: Reset transcribed text
    }
  };

  const handleCancel = () => {
    cleanup();
    setAudioBlob(null);
    setAudioUrl('');
    setRecordingTime(0);
    setIsPlaying(false);
    setError('');
    setTranscribedText(''); // NEW: Reset transcribed text
    onCancel();
  };

  // NEW: Add type definitions for speech recognition
  useEffect(() => {
    // Add global type definitions if needed
    if (typeof window !== 'undefined') {
      (window as any).SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    }
  }, []);

  // Error state - UPDATED to show transcription status
  if (error) {
    return (
      <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex-1">
          <p className="text-red-600 text-sm">{error}</p>
          </div>
            <button
          onClick={handleCancel}
          className="p-2 text-red-600 hover:bg-red-100 rounded-full"
          >
            <X className="w-4 h-4" />
          </button>
      </div>
    );
  }

  // Initial state - show start recording button
  if (!isRecording && !audioBlob) {
    return (
      <button
        onClick={handleStartRecording}
        className="p-3 rounded-full bg-green-500 text-white hover:bg-green-600 transition-all"
        title="Record voice message"
      >
          <Mic className="w-5 h-5" />
      </button>
    );
  }

  // Recording state - UPDATED to show transcription
  if (isRecording) {
    return (
      <div className="flex flex-col space-y-2 p-3 bg-red-50 border border-red-200 rounded-xl min-w-64">
        {/* Recording controls */}
        <div className="flex items-center space-x-3">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
          
          <button
            onClick={handleStopRecording}
            className="p-2 rounded-full bg-red-500 text-white hover:bg-red-600"
          >
            <Square className="w-4 h-4" />
          </button>
          
          <span className="text-red-600 font-bold text-sm">
            {formatTime(recordingTime)}
          </span>
          
          <button
            onClick={handleCancel}
            className="p-2 rounded-full text-red-600 hover:bg-red-100"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* NEW: Transcription display */}
        {transcribedText && (
          <div className="mt-2 p-2 bg-white rounded border border-gray-200">
            <div className="flex items-center space-x-2 mb-1">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-xs text-gray-600 font-medium">Transcribing...</span>
            </div>
            <p className="text-sm text-gray-800 break-words">{transcribedText}</p>
          </div>
        )}

        {/* NEW: Transcription status */}
        {isTranscribing && !transcribedText && (
          <div className="flex items-center space-x-2 text-xs text-gray-600">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span>Listening for speech...</span>
          </div>
        )}
      </div>
    );
  }

  // Audio preview state (after recording) - UPDATED to show final transcription
  if (audioBlob && audioUrl) {
    return (
      <div className="flex flex-col space-y-2 p-3 bg-green-50 border border-green-200 rounded-xl min-w-64">
        {/* Audio controls */}
        <div className="flex items-center space-x-3">
          <audio
            ref={audioRef}
            src={audioUrl}
            onEnded={() => setIsPlaying(false)}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          />
          
          <button
            onClick={handlePlayPause}
            className="p-2 rounded-full bg-blue-500 text-white hover:bg-blue-600"
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
          
          <span className="text-green-600 font-bold text-sm">
            {formatTime(recordingTime)}
          </span>
          
          <button
            onClick={handleSend}
            className="p-2 rounded-full bg-green-500 text-white hover:bg-green-600"
          >
            <Send className="w-4 h-4" />
          </button>
          
          <button
            onClick={handleCancel}
            className="p-2 rounded-full bg-gray-500 text-white hover:bg-gray-600"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* NEW: Final transcription display */}
        {transcribedText && (
          <div className="mt-2 p-2 bg-white rounded border border-green-200">
            <div className="flex items-center space-x-2 mb-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-xs text-gray-600 font-medium">Transcribed Text:</span>
            </div>
            <p className="text-sm text-gray-800 break-words">{transcribedText}</p>
          </div>
        )}
      </div>
    );
  }

  return null;
};

export default VoiceRecorder;