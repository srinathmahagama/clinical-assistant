export interface User {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
  createdAt?: string;
  updatedAt?: string;
  isEmailVerified?: boolean;
}

export interface Assessment {
  id: string;
  userId: string;
  date: string; // Format: YYYY/MM/DD
  time: string; // Format: H:MM a.m/p.m
  symptoms: string[];
  severity: 'Low' | 'Moderate' | 'High';
  recommendations: string[];
  primaryConcerns: string[];
  additionalInfo?: string;
  createdAt: string;
  updatedAt: string;
  isSaved?: boolean;
  downloadCount?: number;
}

export interface VoiceRecording {
  isRecording: boolean;
  transcript: string;
  duration: number;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
}

export interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: string; // Format: H:MM a.m/p.m
  createdAt: string;
}

export interface ChatSession {
  id: string;
  userId: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

// Future API endpoints structure
export interface ApiEndpoints {
  auth: {
    login: '/api/auth/login';
    register: '/api/auth/register';
    logout: '/api/auth/logout';
  };
  assessments: {
    create: '/api/assessments';
    getAll: '/api/assessments';
    getById: '/api/assessments/:id';
  };
  voice: {
    upload: '/api/voice/upload';
    transcribe: '/api/voice/transcribe';
  };
}

// Web Speech API types
declare global {
  interface Window {
    SpeechRecognition: typeof SpeechRecognition;
    webkitSpeechRecognition: typeof SpeechRecognition;
  }
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
  onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null;
  onstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  onend: ((this: SpeechRecognition, ev: Event) => any) | null;
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
  isFinal: boolean;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

declare var SpeechRecognition: {
  prototype: SpeechRecognition;
  new(): SpeechRecognition;
};