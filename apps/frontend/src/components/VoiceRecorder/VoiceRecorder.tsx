import React, { useState, useRef } from 'react';
import { Mic, MicOff, Send, X, Square, Play, Pause } from 'lucide-react';

interface VoiceRecorderProps {
  onSendVoiceMessage: (audioBlob: Blob, duration: number) => void;
  onCancel: () => void;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  onSendVoiceMessage,
  onCancel
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string>('');
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

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
  };

  const handleStartRecording = async () => {
    try {
      setError('');
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
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      startTimer();
      
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
    }
  };

  const handleCancel = () => {
    cleanup();
    setAudioBlob(null);
    setAudioUrl('');
    setRecordingTime(0);
    setIsPlaying(false);
    setError('');
    onCancel();
  };

  // Error state
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

  // Recording state
  if (isRecording) {
    return (
      <div className="flex items-center space-x-3 p-3 bg-red-50 border border-red-200 rounded-xl">
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
    );
  }

  // Audio preview state (after recording)
  if (audioBlob && audioUrl) {
    return (
      <div className="flex items-center space-x-3 p-3 bg-green-50 border border-green-200 rounded-xl">
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
    );
  }

  return null;
};

export default VoiceRecorder;
