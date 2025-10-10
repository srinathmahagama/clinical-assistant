import React, { useState, useRef } from 'react';
import { Play, Pause, Mic } from 'lucide-react';

interface VoiceMessageProps {
  audioUrl: string;
  duration: number;
  isUser: boolean;
  timestamp: string;
  transcript?: string;
}

const VoiceMessage: React.FC<VoiceMessageProps> = ({
  audioUrl,
  duration,
  isUser,
  timestamp,
  transcript
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const togglePlayPause = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleEnded = () => {
    setIsPlaying(false);
  };

  // Generate a simple waveform visualization
  const generateWaveform = () => {
    const bars = 12;
    const waveform = [];
    
    for (let i = 0; i < bars; i++) {
      const height = Math.random() * 20 + 8; // Random height between 8-28px
      waveform.push(
        <div
          key={i}
          className={`w-1 rounded-full ${
            isUser ? 'bg-white' : 'bg-gray-600'
          }`}
          style={{ height: `${height}px` }}
        />
      );
    }
    return waveform;
  };

  return (
    <div className={`flex items-center space-x-3 p-3 rounded-lg max-w-md ${
      isUser
        ? 'bg-blue-600 text-white'
        : 'bg-gray-200 text-gray-800'
    }`}>
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={audioUrl}
        onEnded={handleEnded}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
      />
      
      {/* Play/Pause button */}
      <button
        onClick={togglePlayPause}
        className={`p-2 rounded-full transition-colors ${
          isUser
            ? 'hover:bg-blue-700'
            : 'hover:bg-gray-300'
        }`}
      >
        {isPlaying ? (
          <Pause className="w-4 h-4" />
        ) : (
          <Play className="w-4 h-4" />
        )}
      </button>

      {/* Waveform visualization */}
      <div className="flex items-center space-x-1">
        {generateWaveform()}
      </div>

      {/* Duration */}
      <span className={`text-xs font-medium ${
        isUser ? 'text-white' : 'text-gray-600'
      }`}>
        {formatTime(duration)}
      </span>

      {/* Microphone icon */}
      <Mic className={`w-3 h-3 ${
        isUser ? 'text-white' : 'text-gray-500'
      }`} />
    </div>
  );
};

export default VoiceMessage;
