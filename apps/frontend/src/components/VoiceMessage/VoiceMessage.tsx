import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, FileText, Mic, X } from 'lucide-react';
import { useLanguage } from '../../contexts/LanguageContext';
import { useTheme } from '../../contexts/ThemeContext';
import MessageOptions from '../MessageOptions/MessageOptions';

interface VoiceMessageProps {
  audioUrl: string;
  duration: number;
  isUser: boolean;
  timestamp: string;
  transcript?: string;
  isTranscribed?: boolean;
  onTranscribe?: (() => void) | undefined;
  onDelete?: (() => void) | undefined;
}

const VoiceMessage: React.FC<VoiceMessageProps> = ({
  audioUrl,
  duration,
  isUser,
  timestamp,
  transcript,
  isTranscribed = false,
  onTranscribe,
  onDelete
}) => {
  const { t } = useLanguage();
  const { theme } = useTheme();
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [showTranscript, setShowTranscript] = useState(false);
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

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setCurrentTime(0);
    }
  };

  // Generate a simple waveform visualization based on duration
  const generateWaveform = () => {
    const bars = 20;
    const waveform = [];
    
    // Create a more realistic waveform based on duration
    // Longer messages get more varied waveforms
    const baseHeight = 5;
    const maxHeight = 25;
    const variation = Math.min(duration * 2, 15); // More variation for longer messages
    
    for (let i = 0; i < bars; i++) {
      // Create a more natural waveform pattern
      const position = i / bars;
      const sineWave = Math.sin(position * Math.PI * 4) * variation;
      const randomVariation = (Math.random() - 0.5) * variation * 0.5;
      const height = Math.max(baseHeight, Math.min(maxHeight, baseHeight + sineWave + randomVariation));
      
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
    <div className={`flex flex-col relative ${isUser ? 'items-end' : 'items-start'}`}>
      {/* Audio element */}
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onEnded={handleEnded}
        onLoadedMetadata={handleLoadedMetadata}
        preload="metadata"
      />

      {/* Main message container */}
      <div className={`flex items-center space-x-2 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        {/* Voice message bubble */}
        <div className={`flex items-center space-x-2 px-4 py-3 rounded-lg max-w-md relative group ${
          isUser
            ? theme === 'dark' 
              ? 'bg-blue-900/45 text-white'
              : 'bg-blue-800/45 text-white'
            : theme === 'dark'
              ? 'bg-slate-700 text-slate-100'
              : 'bg-gray-200 text-gray-800'
        }`}>
          {/* Play/Pause button */}
          <button
            onClick={togglePlayPause}
            className={`p-1 rounded-full transition-colors ${
              isUser
                ? 'hover:bg-blue-600'
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
          <span className={`text-xs ${
            isUser ? 'text-white' : 'text-gray-500'
          }`}>
            {formatTime(duration)}
          </span>

          {/* Microphone icon */}
          <Mic className={`w-3 h-3 ${
            isUser ? 'text-white' : 'text-gray-500'
          }`} />

          {/* Message options */}
          {onDelete && (
            <div className="opacity-0 group-hover:opacity-100 transition-opacity">
              <MessageOptions 
                onDelete={onDelete}
                isUser={isUser}
              />
            </div>
          )}
        </div>

        {/* Transcript button - for all voice messages */}
        <button
          onClick={() => {
            if (isTranscribed) {
              setShowTranscript(!showTranscript);
            } else if (onTranscribe) {
              onTranscribe();
            }
          }}
          className={`p-2 rounded-full transition-colors ${
            isTranscribed
              ? isUser 
                ? 'bg-blue-600 text-white hover:bg-blue-700' 
                : 'bg-blue-100 text-blue-600 hover:bg-blue-200'
              : isUser
                ? 'bg-gray-600 text-white hover:bg-gray-700'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
          title={isTranscribed ? t('viewTranscript') : t('transcript')}
        >
          <FileText className="w-4 h-4" />
        </button>
      </div>

      {/* Timestamp */}
      <div className={`text-xs mt-1 ${
        isUser ? 'text-white' : 'text-gray-500'
      }`}>
        {timestamp}
      </div>

      {/* Transcript popup - positioned near the voice message */}
      {showTranscript && transcript && (
        <div className={`absolute ${isUser ? 'right-0' : 'left-0'} top-0 mt-8 p-3 bg-white border border-gray-200 rounded-lg shadow-lg max-w-xs z-20`}>
          <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="text-sm text-gray-800 font-medium mb-1">{t('transcript')}:</p>
                    <p className="text-sm text-gray-700">{transcript}</p>
                  </div>
            <button
              onClick={() => setShowTranscript(false)}
              className="ml-2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
              title={t('closeTranscript')}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceMessage;
