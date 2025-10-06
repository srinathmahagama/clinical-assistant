// ===================================================================
// VOICE DETECTION SERVICE - EXTERNAL API INTEGRATION
// ===================================================================
// This service works independently without backend
// Uses Web Speech API and external transcription services
// ===================================================================

export interface VoiceDetectionResult {
  transcript: string;
  confidence: number;
  isFinal: boolean;
}

export interface VoiceDetectionOptions {
  language?: string;
  continuous?: boolean;
  interimResults?: boolean;
  maxAlternatives?: number;
}

export interface ExternalTranscriptionResult {
  success: boolean;
  transcript?: string;
  confidence?: number;
  error?: string;
}

// ===================================================================
// WEB SPEECH API INTEGRATION
// ===================================================================
export class VoiceDetectionService {
  private recognition: any = null;
  private isSupported: boolean = false;
  private isRecording: boolean = false;
  private onResultCallback?: (result: VoiceDetectionResult) => void;
  private onErrorCallback?: (error: string) => void;
  private onEndCallback?: () => void;
  private lastSpeechTime: number = 0;
  private speechPauses: number[] = [];
  private currentTranscript: string = '';

  constructor() {
    this.checkSupport();
  }

  // Check if Web Speech API is supported
  private checkSupport(): void {
    this.isSupported = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
  }

  // Get support status
  public isVoiceSupported(): boolean {
    return this.isSupported;
  }

  // Initialize speech recognition
  public initialize(options: VoiceDetectionOptions = {}): boolean {
    if (!this.isSupported) {
      console.error('Voice detection not supported in this browser');
      return false;
    }

    try {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      this.recognition = new SpeechRecognition();
      
      // Configure recognition
      this.recognition.continuous = options.continuous || true;
      this.recognition.interimResults = options.interimResults || true;
      this.recognition.lang = options.language || 'en-US';
      this.recognition.maxAlternatives = options.maxAlternatives || 1;

      // Set up event handlers
      this.setupEventHandlers();
      
      return true;
    } catch (error) {
      console.error('Failed to initialize voice detection:', error);
      return false;
    }
  }

  // Update language without full reinitialization
  public updateLanguage(language: string): void {
    if (this.recognition && !this.isRecording) {
      try {
        this.recognition.lang = language;
      } catch (error) {
        console.warn('Failed to update voice detection language:', error);
        // Fallback to English if language update fails
        this.recognition.lang = 'en-US';
      }
    }
  }

  // Check if a specific language is supported
  public isLanguageSupported(language: string): boolean {
    // List of commonly supported languages by Web Speech API
    const supportedLanguages = [
      'en-US', 'en-GB', 'en-AU', 'en-CA',
      'es-ES', 'es-MX', 'es-AR',
      'fr-FR', 'fr-CA',
      'de-DE', 'de-AT',
      'it-IT', 'it-CH',
      'pt-BR', 'pt-PT',
      'nl-NL', 'nl-BE',
      'ru-RU',
      'ja-JP',
      'ko-KR',
      'zh-CN', 'zh-TW',
      'ar-SA', 'ar-EG',
      'hi-IN',
      'th-TH',
      'tr-TR',
      'sv-SE', 'no-NO', 'da-DK', 'fi-FI',
      'si-LK'
    ];
    
    return supportedLanguages.includes(language);
  }

  // Test if a language actually works by trying to set it
  public async testLanguageSupport(language: string): Promise<boolean> {
    if (!this.recognition) return false;
    
    try {
      // Simply try to set the language - if it throws an error, it's not supported
      this.recognition.lang = language;
      
      // If we get here without error, the language is supported
      return true;
    } catch (error) {
      console.warn(`Language ${language} not supported:`, error);
      return false;
    }
  }

  // Set up event handlers
  private setupEventHandlers(): void {
    if (!this.recognition) return;

    this.recognition.onresult = (event: any) => {
      let finalTranscript = '';
      let interimTranscript = '';
      let confidence = 0;

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const transcript = result[0].transcript;
        confidence = result[0].confidence || 0;

        if (result.isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      const transcript = finalTranscript || interimTranscript;
      if (transcript && this.onResultCallback) {
        // Store current transcript for timing analysis
        this.currentTranscript = transcript;
        
        this.onResultCallback({
          transcript: this.enhanceTranscript(transcript),
          confidence,
          isFinal: !!finalTranscript
        });
      }
    };

    this.recognition.onerror = (event: any) => {
      this.isRecording = false;
      const errorMessage = this.getErrorMessage(event.error);
      if (this.onErrorCallback) {
        this.onErrorCallback(errorMessage);
      }
    };

    this.recognition.onend = () => {
      this.isRecording = false;
      if (this.onEndCallback) {
        this.onEndCallback();
      }
    };
  }

  // Get user-friendly error messages
  private getErrorMessage(error: string): string {
    switch (error) {
      case 'no-speech':
        return 'No speech detected. Please speak clearly and try again.';
      case 'audio-capture':
        return 'Microphone not accessible. Please check your microphone permissions.';
      case 'not-allowed':
        return 'Microphone access denied. Please allow microphone access and try again.';
      case 'network':
        return 'Network error. Please check your internet connection and try again.';
      case 'service-not-allowed':
        return 'Speech recognition service not allowed. Please check your browser settings.';
      default:
        return `Speech recognition failed: ${error}. Please try again.`;
    }
  }

  // Enhance transcript with smart punctuation based on timing and phrases
  private enhanceTranscript(text: string): string {
    if (!text.trim()) return text;
    
    let enhancedText = text.trim();
    const currentTime = Date.now();
    
    // Track speech timing for pause detection
    if (this.lastSpeechTime > 0) {
      const pauseDuration = currentTime - this.lastSpeechTime;
      this.speechPauses.push(pauseDuration);
    }
    this.lastSpeechTime = currentTime;
    
    // Smart punctuation based on question phrases
    enhancedText = this.addQuestionMarks(enhancedText);
    
    // Add periods based on timing and natural pauses
    enhancedText = this.addPeriods(enhancedText);
    
    // Add commas for lists and conjunctions
    enhancedText = this.addCommas(enhancedText);
    
    // Clean up and format
    enhancedText = this.cleanupAndFormat(enhancedText);
    
    return enhancedText.trim();
  }

  // Detect and add question marks based on question phrases
  private addQuestionMarks(text: string): string {
    // Question words and phrases
    const questionPatterns = [
      // Direct question words
      /\b(what|how|when|where|why|which|who|whom|whose)\b/gi,
      // Question phrases
      /\b(should i|can i|could i|would i|do i|does it|is it|are you|was it|were you)\b/gi,
      // Medical question patterns
      /\b(what should|how should|when should|where should|why should)\b/gi,
      // Symptom questions
      /\b(is this|are these|does this|do these|should this|can this)\b/gi,
      // Time-based questions
      /\b(how long|how often|how much|how many)\b/gi
    ];

    // Check if text contains question patterns
    const hasQuestionPattern = questionPatterns.some(pattern => pattern.test(text));
    
    // Check for rising intonation indicators (common in questions)
    const hasRisingIntonation = /\b(should|can|could|would|do|does|is|are|was|were)\s+\w+\s*$/gi.test(text);
    
    // Check for question words at the beginning
    const startsWithQuestion = /^(what|how|when|where|why|which|who|should|can|could|would|do|does|is|are|was|were)/gi.test(text.trim());
    
    // Add question mark if it's likely a question
    if ((hasQuestionPattern || hasRisingIntonation || startsWithQuestion) && !text.includes('?')) {
      // Remove any trailing punctuation and add question mark
      text = text.replace(/[.!]+$/, '') + '?';
    }
    
    return text;
  }

  // Add periods based on timing and natural pauses
  private addPeriods(text: string): string {
    // Time references that typically end sentences
    const timeReferences = /\s+(yesterday|today|this morning|last week|last night|earlier|recently|now|then|afterwards)\s+/gi;
    text = text.replace(timeReferences, ' $1. ');
    
    // Action words that typically end sentences
    const actionWords = /\s+(started|began|happened|occurred|finished|ended|stopped|continued)\s+/gi;
    text = text.replace(actionWords, ' $1. ');
    
    // Medical/symptom completion phrases
    const symptomPhrases = /\s+(better|worse|same|improved|deteriorated|persisted|continued|stopped|started)\s+/gi;
    text = text.replace(symptomPhrases, ' $1. ');
    
    // Add periods based on pause timing (if we have timing data)
    if (this.speechPauses.length > 0) {
      const avgPause = this.speechPauses.reduce((a, b) => a + b, 0) / this.speechPauses.length;
      // If average pause is longer than 1.5 seconds, likely end of sentence
      if (avgPause > 1500 && !text.match(/[.!?]$/)) {
        text += '.';
      }
    }
    
    return text;
  }

  // Add commas for lists and conjunctions
  private addCommas(text: string): string {
    // List connectors
    text = text.replace(/\s+(and|with|plus|also|as well as)\s+/gi, ', $1 ');
    text = text.replace(/(\w+)\s+(and|with|plus|also)\s+(\w+)/gi, '$1, $2 $3');
    
    // Conjunctions
    text = text.replace(/\s+(but|however|though|although|so|therefore|because|since|while|whereas)\s+/gi, ', $1 ');
    
    // Symptom lists (common in medical descriptions)
    text = text.replace(/\b(headache|fever|cough|pain|nausea|dizziness|fatigue|weakness)\s+(and|with|plus)\s+/gi, '$1, $2 ');
    
    return text;
  }

  // Clean up and format the text
  private cleanupAndFormat(text: string): string {
    // Clean up multiple spaces
    text = text.replace(/\s+/g, ' ');
    
    // Clean up punctuation spacing
    text = text.replace(/\s*([,.!?])\s*/g, '$1 ');
    text = text.replace(/\s*([,.!?])\s*([,.!?])/g, '$1$2');
    
    // Ensure proper capitalization
    text = text.replace(/^([a-z])/, (match) => match.toUpperCase());
    text = text.replace(/([.!?])\s+([a-z])/g, (match, p1, p2) => p1 + ' ' + p2.toUpperCase());
    
    // Clean up multiple punctuation
    text = text.replace(/([.!?])\s*([.!?])+/g, '$1');
    
    return text;
  }

  // Start voice detection
  public startDetection(): boolean {
    if (!this.recognition || this.isRecording) {
      return false;
    }

    try {
      // Reset timing data for new session
      this.lastSpeechTime = 0;
      this.speechPauses = [];
      this.currentTranscript = '';
      
      this.recognition.start();
      this.isRecording = true;
      return true;
    } catch (error) {
      console.error('Failed to start voice detection:', error);
      this.isRecording = false;
      return false;
    }
  }

  // Stop voice detection
  public stopDetection(): void {
    if (this.recognition && this.isRecording) {
      this.recognition.stop();
      this.isRecording = false;
    }
  }

  // Abort voice detection
  public abortDetection(): void {
    if (this.recognition && this.isRecording) {
      this.recognition.abort();
      this.isRecording = false;
    }
  }

  // Check if currently recording
  public isCurrentlyRecording(): boolean {
    return this.isRecording;
  }

  // Set result callback
  public onResult(callback: (result: VoiceDetectionResult) => void): void {
    this.onResultCallback = callback;
  }

  // Set error callback
  public onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  // Set end callback
  public onEnd(callback: () => void): void {
    this.onEndCallback = callback;
  }

  // Clean up
  public destroy(): void {
    if (this.recognition) {
      this.recognition.abort();
      this.recognition = null;
    }
    this.isRecording = false;
    this.onResultCallback = undefined;
    this.onErrorCallback = undefined;
    this.onEndCallback = undefined;
  }
}

// ===================================================================
// EXTERNAL TRANSCRIPTION SERVICES
// ===================================================================

// Google Cloud Speech-to-Text API (requires API key)
export const googleTranscription = async (
  audioBlob: Blob,
  apiKey: string,
  language: string = 'en-US'
): Promise<ExternalTranscriptionResult> => {
  try {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'audio.wav');
    formData.append('language', language);

    const response = await fetch(
      `https://speech.googleapis.com/v1/speech:recognize?key=${apiKey}`,
      {
        method: 'POST',
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error(`Google API error: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.results && data.results.length > 0) {
      const result = data.results[0];
      return {
        success: true,
        transcript: result.alternatives[0].transcript,
        confidence: result.alternatives[0].confidence
      };
    }

    return {
      success: false,
      error: 'No transcription results'
    };
  } catch (error) {
    return {
      success: false,
      error: `Google transcription failed: ${error}`
    };
  }
};

// Azure Speech Services (requires subscription key)
export const azureTranscription = async (
  audioBlob: Blob,
  subscriptionKey: string,
  region: string,
  language: string = 'en-US'
): Promise<ExternalTranscriptionResult> => {
  try {
    const response = await fetch(
      `https://${region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=${language}`,
      {
        method: 'POST',
        headers: {
          'Ocp-Apim-Subscription-Key': subscriptionKey,
          'Content-Type': 'audio/wav',
        },
        body: audioBlob,
      }
    );

    if (!response.ok) {
      throw new Error(`Azure API error: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.RecognitionStatus === 'Success') {
      return {
        success: true,
        transcript: data.DisplayText,
        confidence: data.Confidence
      };
    }

    return {
      success: false,
      error: `Azure recognition failed: ${data.RecognitionStatus}`
    };
  } catch (error) {
    return {
      success: false,
      error: `Azure transcription failed: ${error}`
    };
  }
};

// AssemblyAI (requires API key)
export const assemblyAITranscription = async (
  audioBlob: Blob,
  apiKey: string
): Promise<ExternalTranscriptionResult> => {
  try {
    // First, upload the audio file
    const uploadResponse = await fetch('https://api.assemblyai.com/v2/upload', {
      method: 'POST',
      headers: {
        'authorization': apiKey,
      },
      body: audioBlob,
    });

    if (!uploadResponse.ok) {
      throw new Error(`AssemblyAI upload error: ${uploadResponse.status}`);
    }

    const uploadData = await uploadResponse.json();
    const audioUrl = uploadData.upload_url;

    // Then, start transcription
    const transcribeResponse = await fetch('https://api.assemblyai.com/v2/transcript', {
      method: 'POST',
      headers: {
        'authorization': apiKey,
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        audio_url: audioUrl,
        language_detection: true,
      }),
    });

    if (!transcribeResponse.ok) {
      throw new Error(`AssemblyAI transcription error: ${transcribeResponse.status}`);
    }

    const transcribeData = await transcribeResponse.json();
    const transcriptId = transcribeData.id;

    // Poll for completion
    let attempts = 0;
    const maxAttempts = 30; // 30 seconds timeout

    while (attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const statusResponse = await fetch(`https://api.assemblyai.com/v2/transcript/${transcriptId}`, {
        headers: {
          'authorization': apiKey,
        },
      });

      const statusData = await statusResponse.json();

      if (statusData.status === 'completed') {
        return {
          success: true,
          transcript: statusData.text,
          confidence: statusData.confidence
        };
      } else if (statusData.status === 'error') {
        return {
          success: false,
          error: `AssemblyAI transcription error: ${statusData.error}`
        };
      }

      attempts++;
    }

    return {
      success: false,
      error: 'AssemblyAI transcription timeout'
    };
  } catch (error) {
    return {
      success: false,
      error: `AssemblyAI transcription failed: ${error}`
    };
  }
};

// ===================================================================
// VOICE DETECTION UTILITIES
// ===================================================================

// Check browser compatibility
export const checkVoiceCompatibility = (): {
  supported: boolean;
  browser: string;
  features: string[];
} => {
  const features = [];
  let supported = false;

  // Check Web Speech API
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    features.push('Web Speech API');
    supported = true;
  }

  // Check MediaRecorder API
  if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function') {
    features.push('MediaRecorder API');
  }

  // Check AudioContext
  if (window.AudioContext || (window as any).webkitAudioContext) {
    features.push('AudioContext');
  }

  // Detect browser
  const userAgent = navigator.userAgent;
  let browser = 'Unknown';
  if (userAgent.includes('Chrome')) browser = 'Chrome';
  else if (userAgent.includes('Firefox')) browser = 'Firefox';
  else if (userAgent.includes('Safari')) browser = 'Safari';
  else if (userAgent.includes('Edge')) browser = 'Edge';

  return {
    supported,
    browser,
    features
  };
};

// Get supported languages
export const getSupportedLanguages = (): string[] => {
  return [
    'en-US', 'en-GB', 'en-AU', 'en-CA',
    'es-ES', 'es-MX', 'es-AR',
    'fr-FR', 'fr-CA',
    'de-DE', 'de-AT',
    'it-IT', 'pt-BR', 'pt-PT',
    'nl-NL', 'nl-BE',
    'ru-RU', 'ja-JP', 'ko-KR',
    'zh-CN', 'zh-TW', 'hi-IN',
    'ar-SA', 'th-TH', 'vi-VN'
  ];
};

// Export default instance
export const voiceDetection = new VoiceDetectionService();
export default voiceDetection;
