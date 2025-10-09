import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Send, X, Square, Play, Pause } from 'lucide-react';

interface VoiceRecorderProps {
  onSendVoiceMessage: (audioBlob: Blob, duration: number) => void;
  onCancel: () => void;
  isRecording: boolean;
  onStartRecording: () => void;
  onStopRecording: () => void;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  onSendVoiceMessage,
  onCancel,
  isRecording,
  onStartRecording,
  onStopRecording
}) => {
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState<string>('');
  const [showPermissionHelp, setShowPermissionHelp] = useState(false);
  const [audioLevels, setAudioLevels] = useState<number[]>([]);
  
  // Force clear error function
  const clearError = () => {
    setError('');
    setShowPermissionHelp(false);
    console.log('Error state cleared');
  };
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Cleanup effect and clear error on mount
  useEffect(() => {
    // Clear any existing error when component mounts
    setError('');
    setShowPermissionHelp(false);
    
    return () => {
      cleanup();
    };
  }, []);

  // Add keyboard shortcut to clear error
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && error) {
        clearError();
      }
    };

    if (error) {
      document.addEventListener('keydown', handleKeyPress);
      return () => {
        document.removeEventListener('keydown', handleKeyPress);
      };
    }
  }, [error]);

  const cleanup = () => {
    // Stop all tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    // Clear timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    
    // Stop audio level monitoring
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    // Close audio context
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    // Revoke audio URL
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    
    // Reset audio levels
    setAudioLevels([]);
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

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const startAudioLevelMonitoring = (stream: MediaStream) => {
    try {
      // Create audio context for level monitoring
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioContext;
      
      // Create analyser node
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;
      
      // Connect stream to analyser
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      // Start monitoring audio levels
      const monitorAudioLevels = () => {
        if (!analyserRef.current) return;
        
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);
        
        // Calculate average level
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i];
        }
        const averageLevel = sum / bufferLength;
        
        // Update audio levels for waveform visualization
        setAudioLevels(prev => {
          const newLevels = [...prev, averageLevel];
          // Keep only last 20 levels for performance
          return newLevels.slice(-20);
        });
        
        // Continue monitoring
        animationFrameRef.current = requestAnimationFrame(monitorAudioLevels);
      };
      
      monitorAudioLevels();
    } catch (error) {
      console.warn('Audio level monitoring failed:', error);
    }
  };

  const testMicrophoneAccess = async () => {
    try {
      console.log('Testing microphone access...');
      
      // Simple microphone test with minimal constraints
      const testStream = await navigator.mediaDevices.getUserMedia({ 
        audio: true 
      });
      
      console.log('Microphone test successful:', testStream);
      
      // Stop the test stream immediately
      testStream.getTracks().forEach(track => track.stop());
      
      return true;
    } catch (error) {
      console.error('Microphone test failed:', error);
      return false;
    }
  };

  const handleStartRecording = async () => {
    try {
      setIsInitializing(true);
      setError('');
      
      console.log('Starting voice recording...');
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser');
      }
      
      // Log browser and device info for debugging
      console.log('Browser info:', {
        userAgent: navigator.userAgent,
        platform: navigator.platform,
        mediaDevices: !!navigator.mediaDevices,
        getUserMedia: !!navigator.mediaDevices?.getUserMedia
      });
      
      // Test microphone access first
      const micTest = await testMicrophoneAccess();
      if (!micTest) {
        throw new Error('Microphone access test failed. Please check your microphone permissions.');
      }
      
      // Check current permission state first
      try {
        const permissionStatus = await navigator.permissions.query({ name: 'microphone' as PermissionName });
        console.log('Microphone permission status:', permissionStatus.state);
        
        if (permissionStatus.state === 'denied') {
          throw new Error('Microphone permission is denied. Please enable it in your browser settings.');
        }
      } catch (permError) {
        console.log('Could not check permission status:', permError);
        // Continue anyway, as some browsers don't support permission query
      }
      
      // Request microphone access with optimized settings for voice recording
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 48000, // Higher sample rate for better quality
          channelCount: 1, // Mono recording for voice
          // latency: 0.01 // Low latency
        } 
      });
      
      console.log('Microphone access granted, stream:', stream);
      streamRef.current = stream;
      
      // Start audio level monitoring for real-time waveform
      startAudioLevelMonitoring(stream);
      
      // Create MediaRecorder with better compatibility and quality
      let mimeType = 'audio/webm';
      let options = {};
      
      // Try different MIME types in order of preference for voice recording
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
        options = { mimeType, audioBitsPerSecond: 128000 }; // Higher bitrate for better quality
      } else if (MediaRecorder.isTypeSupported('audio/webm')) {
        mimeType = 'audio/webm';
        options = { mimeType, audioBitsPerSecond: 128000 };
      } else if (MediaRecorder.isTypeSupported('audio/mp4;codecs=mp4a.40.2')) {
        mimeType = 'audio/mp4;codecs=mp4a.40.2';
        options = { mimeType, audioBitsPerSecond: 128000 };
      } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
        mimeType = 'audio/mp4';
        options = { mimeType, audioBitsPerSecond: 128000 };
      } else if (MediaRecorder.isTypeSupported('audio/wav')) {
        mimeType = 'audio/wav';
        options = { mimeType };
      } else {
        // Fallback - let browser choose but with bitrate setting
        console.warn('No specific MIME type supported, using browser default');
        options = { audioBitsPerSecond: 128000 };
      }
      
      console.log('Using MIME type:', mimeType);
      
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      console.log('MediaRecorder created:', mediaRecorder);
      
      // Handle data available
      mediaRecorder.ondataavailable = (event) => {
        console.log('Data available:', event.data.size, 'bytes');
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      // Handle recording stop
      mediaRecorder.onstop = () => {
        console.log('Recording stopped, chunks:', audioChunksRef.current.length);
        if (audioChunksRef.current.length > 0) {
          const blob = new Blob(audioChunksRef.current, { type: mimeType });
          console.log('Audio blob created:', blob.size, 'bytes, type:', blob.type);
          setAudioBlob(blob);
          const url = URL.createObjectURL(blob);
          setAudioUrl(url);
        }
        
        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        
        stopTimer();
      };
      
      // Handle errors
      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        setError('Recording failed. Please try again.');
        cleanup();
        onCancel();
      };
      
      // Start recording with smaller time slices for better quality
      mediaRecorder.start(100); // Collect data every 100ms for better quality
      console.log('Recording started');
      setError(''); // Clear any previous errors
      setShowPermissionHelp(false);
      startTimer();
      onStartRecording();
      
    } catch (error) {
      console.error('Failed to start recording:', error);
      let errorMessage = 'Failed to start recording. ';
      
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          errorMessage = 'Microphone access denied. Please click the microphone icon in your browser\'s address bar and allow access, then try again.';
        } else if (error.name === 'NotFoundError') {
          errorMessage = 'No microphone found. Please connect a microphone and try again.';
        } else if (error.name === 'NotSupportedError') {
          errorMessage = 'Voice recording is not supported in this browser. Please try using Chrome, Firefox, or Edge.';
        } else if (error.message.includes('permission is denied')) {
          errorMessage = 'Microphone permission is denied. Please enable it in your browser settings and refresh the page.';
        } else {
          errorMessage += error.message;
        }
      }
      
      setError(errorMessage);
      cleanup();
      onCancel();
    } finally {
      setIsInitializing(false);
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      onStopRecording();
    }
  };

  const handlePlayPause = () => {
    if (!audioRef.current || !audioUrl) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
  };

  const handleSend = () => {
    console.log('Sending voice message:', audioBlob, 'duration:', recordingTime);
    if (audioBlob && recordingTime > 0) {
      onSendVoiceMessage(audioBlob, recordingTime);
      // Clean up
      cleanup();
      setAudioBlob(null);
      setAudioUrl('');
      setRecordingTime(0);
      setIsPlaying(false);
      setError('');
    } else {
      console.error('Cannot send: missing audioBlob or invalid duration');
    }
  };

  const handleCancel = () => {
    cleanup();
    setAudioBlob(null);
    setAudioUrl('');
    setRecordingTime(0);
    setIsPlaying(false);
    setError('');
    setShowPermissionHelp(false);
    onCancel();
  };

  // Generate waveform visualization based on real audio levels
  const generateWaveform = () => {
    const bars = 8;
    const waveform = [];
    
    for (let i = 0; i < bars; i++) {
      let height = 8; // Minimum height
      
      if (audioLevels.length > 0) {
        // Use real audio levels if available
        const levelIndex = Math.floor((i / bars) * audioLevels.length);
        const audioLevel = audioLevels[levelIndex] || 0;
        height = Math.max(8, (audioLevel / 255) * 30 + 8); // Scale to 8-38px
      } else {
        // Fallback to animated bars during recording
        height = Math.random() * 20 + 8;
      }
      
      waveform.push(
        <div
          key={i}
          className="w-1 bg-red-500 rounded-full animate-pulse"
          style={{ height: `${height}px` }}
        />
      );
    }
    return waveform;
  };

  // Show error state
  if (error) {
    return (
      <div className="space-y-2">
        <div className="flex items-center space-x-2 p-3 bg-red-50 border border-red-200 rounded-lg max-w-md">
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-1">
              <Mic className="w-4 h-4 text-red-600" />
              <span className="text-red-600 text-sm font-medium">Microphone Error</span>
            </div>
            <p className="text-red-600 text-xs">{error}</p>
          </div>
          <div className="flex items-center space-x-1">
            <button
              onClick={handleStartRecording}
              className="p-2 text-green-600 hover:bg-green-100 rounded-full transition-colors"
              title="Retry recording"
            >
              <Mic className="w-4 h-4" />
            </button>
          <button
            onClick={clearError}
            className="p-2 text-red-600 hover:bg-red-100 rounded-full transition-colors"
            title="Close error"
          >
            <X className="w-4 h-4" />
          </button>
          </div>
        </div>
        
        {/* Permission help */}
        {error.includes('denied') && (
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg max-w-md">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
                <span className="text-white text-xs">?</span>
              </div>
              <span className="text-blue-600 text-sm font-medium">How to fix:</span>
            </div>
            <div className="text-blue-600 text-xs space-y-1">
              <p>1. Look for a microphone icon in your browser's address bar</p>
              <p>2. Click it and select "Allow" for microphone access</p>
              <p>3. Refresh the page and try again</p>
              <p>4. If still not working, check your system microphone settings</p>
            </div>
            <div className="mt-2 pt-2 border-t border-blue-200">
            <button
              onClick={clearError}
              className="text-blue-600 text-xs hover:text-blue-800 underline"
            >
              Clear error and try again
            </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Initial state - show start recording button
  if (!isRecording && !audioBlob) {
    return (
      <button
        onClick={handleStartRecording}
        disabled={isInitializing}
        className="p-3 rounded-full bg-green-500 text-white hover:bg-green-600 transition-all transform hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        title="Record voice message"
      >
        {isInitializing ? (
          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
        ) : (
          <Mic className="w-5 h-5" />
        )}
      </button>
    );
  }

  // Recording state
  if (isRecording) {
    return (
      <div className="flex items-center space-x-3 p-3 bg-red-50 border border-red-200 rounded-xl min-w-[280px] animate-pulse">
        {/* Recording indicator */}
        <div className="w-3 h-3 bg-red-500 rounded-full animate-ping"></div>
        
        {/* Stop Button */}
        <button
          onClick={handleStopRecording}
          className="p-2 rounded-full bg-red-500 text-white hover:bg-red-600 transition-all"
          title="Stop Recording"
        >
          <Square className="w-4 h-4" />
        </button>
        
        {/* Waveform */}
        <div className="flex items-center space-x-1 flex-1 justify-center">
          {generateWaveform()}
        </div>
        
        {/* Timer */}
        <span className="text-red-600 font-bold text-sm min-w-[40px]">
          {formatTime(recordingTime)}
        </span>
        
        {/* Cancel Button */}
        <button
          onClick={handleCancel}
          className="p-2 rounded-full text-red-600 hover:bg-red-100 transition-colors"
          title="Cancel Recording"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  // Audio preview state (after recording)
  if (audioBlob && audioUrl) {
    return (
      <div className="flex items-center space-x-3 p-3 bg-green-50 border border-green-200 rounded-xl min-w-[280px]">
        {/* Hidden audio element */}
        <audio
          ref={audioRef}
          src={audioUrl}
          onEnded={() => setIsPlaying(false)}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />
        
        {/* Play/Pause Button */}
        <button
          onClick={handlePlayPause}
          className="p-2 rounded-full bg-blue-500 text-white hover:bg-blue-600 transition-all"
          title={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
        
        {/* Duration */}
        <span className="text-green-600 font-bold text-sm min-w-[40px]">
          {formatTime(recordingTime)}
        </span>
        
        {/* Send Button */}
        <button
          onClick={handleSend}
          className="p-2 rounded-full bg-green-500 text-white hover:bg-green-600 transition-all"
          title="Send voice message"
        >
          <Send className="w-4 h-4" />
        </button>
        
        {/* Cancel Button */}
        <button
          onClick={handleCancel}
          className="p-2 rounded-full bg-gray-500 text-white hover:bg-gray-600 transition-all"
          title="Cancel"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return null;
};

export default VoiceRecorder;