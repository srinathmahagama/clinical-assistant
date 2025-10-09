import React, { createContext, useContext, useRef, useState, useCallback } from 'react';

interface AudioContextType {
  currentlyPlaying: string | null;
  playAudio: (messageId: string, audioElement: HTMLAudioElement | SpeechSynthesisUtterance) => void;
  stopAllAudio: () => void;
  isPlaying: (messageId: string) => boolean;
}

const AudioContext = createContext<AudioContextType | undefined>(undefined);

export const useAudio = () => {
  const context = useContext(AudioContext);
  if (!context) {
    throw new Error('useAudio must be used within an AudioProvider');
  }
  return context;
};

interface AudioProviderProps {
  children: React.ReactNode;
}

export const AudioProvider: React.FC<AudioProviderProps> = ({ children }) => {
  const [currentlyPlaying, setCurrentlyPlaying] = useState<string | null>(null);
  const audioRefs = useRef<Map<string, HTMLAudioElement | SpeechSynthesisUtterance>>(new Map());

  const playAudio = useCallback((messageId: string, audioElement: HTMLAudioElement | SpeechSynthesisUtterance) => {
    // Stop any currently playing audio
    stopAllAudio();
    
    // Store the new audio element
    audioRefs.current.set(messageId, audioElement);
    setCurrentlyPlaying(messageId);

    // Handle different audio types
    if (audioElement instanceof HTMLAudioElement) {
      audioElement.play().catch(console.error);
      audioElement.onended = () => {
        setCurrentlyPlaying(null);
        audioRefs.current.delete(messageId);
      };
    } else if (audioElement instanceof SpeechSynthesisUtterance) {
      audioElement.onend = () => {
        setCurrentlyPlaying(null);
        audioRefs.current.delete(messageId);
      };
      audioElement.onerror = () => {
        setCurrentlyPlaying(null);
        audioRefs.current.delete(messageId);
      };
      window.speechSynthesis.speak(audioElement);
    }
  }, []);

  const stopAllAudio = useCallback(() => {
    // Stop HTML audio elements
    audioRefs.current.forEach((audioElement) => {
      if (audioElement instanceof HTMLAudioElement) {
        audioElement.pause();
        audioElement.currentTime = 0;
      }
    });

    // Stop speech synthesis
    window.speechSynthesis.cancel();

    // Clear all references
    audioRefs.current.clear();
    setCurrentlyPlaying(null);
  }, []);

  const isPlaying = useCallback((messageId: string) => {
    return currentlyPlaying === messageId;
  }, [currentlyPlaying]);

  const value: AudioContextType = {
    currentlyPlaying,
    playAudio,
    stopAllAudio,
    isPlaying,
  };

  return (
    <AudioContext.Provider value={value}>
      {children}
    </AudioContext.Provider>
  );
};
