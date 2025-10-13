import { User, Assessment, ApiResponse, Message, ChatSession } from '../types';

// ===================================================================
// FLASK BACKEND INTEGRATION
// ===================================================================
// Backend endpoints are implemented in Flask
// Frontend dummy data removed for implemented endpoints
// ===================================================================

// API Base URL - Flask backend
const API_BASE_URL = 'http://localhost:5000';

// ===================================================================
// JWT AUTHENTICATION UTILITIES
// ===================================================================

// Get JWT token from localStorage
const getAuthToken = (): string | null => {
  return localStorage.getItem('authToken');
};

// Set JWT token in localStorage
const setAuthToken = (token: string): void => {
  localStorage.setItem('authToken', token);
};

// Remove JWT token from localStorage
const removeAuthToken = (): void => {
  localStorage.removeItem('authToken');
};

// Create headers with JWT token
const createAuthHeaders = (): HeadersInit => {
  const token = getAuthToken();
  return {
    'Content-Type': 'application/json',
    ...(token && { 'Authorization': `Bearer ${token}` })
  };
};

// Handle API response and token refresh
// const handleApiResponse = async (response: Response): Promise<any> => {
//   if (response.status === 401) {
//     // Token expired, try to refresh
//     const refreshResponse = await fetch(`${API_BASE_URL}/auth/refresh`, {
//       method: 'POST',
//       headers: createAuthHeaders()
//     });
    
//     if (refreshResponse.ok) {
//       const refreshData = await refreshResponse.json();
//       if (refreshData.success && refreshData.data?.token) {
//         setAuthToken(refreshData.data.token);
//         // Retry original request with new token
//         return fetch(response.url, {
//           method: 'GET', // Default method for retry
//           headers: createAuthHeaders()
//         }).then(res => res.json());
//       }
//     }
    
//     // Refresh failed, redirect to login
//     removeAuthToken();
//     localStorage.removeItem('user');
//     window.location.href = '/login';
//     throw new Error('Authentication failed');
//   }
  
//   return response.json();
// };

// ===================================================================
// DUMMY DATA - FOR NON-IMPLEMENTED ENDPOINTS ONLY
// ===================================================================
// const mockUsers: User[] = [
//   {
//     id: '1',
//     firstName: 'Nimash',
//     lastName: 'Silva',
//     email: 'nimash@example.com'
//   },
//   {
//     id: '2',
//     firstName: 'John',
//     lastName: 'Doe',
//     email: 'john@example.com'
//   }
// ];

const mockAssessments: Assessment[] = [
  {
    id: '1',
    userId: '1',
    date: '2025/01/16',
    time: '8:00 a.m',
    symptoms: ['Fever', 'Headache', 'Body Aches'],
    severity: 'Moderate',
    recommendations: [
      'Contact your healthcare provider for evaluation',
      'Keep track of your symptoms and how you\'re feeling',
      'Seek immediate care if symptoms become severe'
    ],
    primaryConcerns: ['fever management', 'headache relief'],
    additionalInfo: 'Started 2 days ago',
    createdAt: '2025-01-16T08:00:00Z',
    updatedAt: '2025-01-16T08:00:00Z',
    isSaved: true,
    downloadCount: 2
  }
];

// const mockChatHistory: Message[] = [
//   {
//     id: '1',
//     text: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
//     isUser: false,
//     timestamp: '10:00 p.m',
//     createdAt: '2025-01-16T22:00:00Z'
//   }
// ];

// const mockChatSessions: ChatSession[] = [
//   {
//     id: '1',
//     userId: '1',
//     title: 'Sample Chat',
//     messages: mockChatHistory,
//     createdAt: '2025-01-16T10:00:00Z',
//     updatedAt: '2025-01-16T10:00:00Z'
//   }
// ];

// ===================================================================
// HELPER FUNCTIONS
// ===================================================================

// Simulate API delay
const simulateApiDelay = (ms: number = 1000) => new Promise(resolve => setTimeout(resolve, ms));

// Generate random ID
// const generateId = () => Date.now().toString() + Math.random().toString(36).substr(2, 9);

// ===================================================================
// AUTHENTICATION SERVICE - FLASK BACKEND INTEGRATION
// ===================================================================
export const authService = {
  // POST /api/auth/login
  login: async (email: string, password: string): Promise<ApiResponse<User>> => {
    console.log('🔐 FLASK API: Login attempt for', email);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        const { user, access_token } = data.data;
        setAuthToken(access_token);
      localStorage.setItem('user', JSON.stringify(user));
      return { success: true, data: user };
    }
    
      return { success: false, message: data.message || 'Login failed' };
    } catch (error) {
      console.error('Login failed:', error);
      return { success: false, message: 'Network error during login' };
    }
  },

  // POST /api/auth/register
  register: async (userData: Omit<User, 'id'> & { password: string }): Promise<ApiResponse<User>> => {
    console.log('📝 FLASK API: Register attempt for', userData.email);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData)
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        const { user, access_token } = data.data;
        setAuthToken(access_token);
        localStorage.setItem('user', JSON.stringify(user));
        return { success: true, data: user };
      }
      
      return { success: false, message: data.message || 'Registration failed' };
    } catch (error) {
      console.error('Registration failed:', error);
      return { success: false, message: 'Network error during registration' };
    }
  },

  // POST /api/auth/logout
  logout: async (): Promise<ApiResponse<null>> => {
    console.log('🚪 FLASK API: Logout');
    
    try {
      await fetch(`${API_BASE_URL}/api/auth/logout`, {
        method: 'POST',
        headers: createAuthHeaders()
      });
      
      // const data = await response.json();
      
      removeAuthToken();
      localStorage.removeItem('user');
      return { success: true };
    } catch (error) {
      console.error('Logout failed:', error);
      // Still clear local storage even if API call fails
    removeAuthToken();
    localStorage.removeItem('user');
    return { success: true };
    }
  },

  // GET /api/auth/me
  getCurrentUser: async (): Promise<ApiResponse<User>> => {
    console.log('👤 FLASK API: Get current user');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
        method: 'GET',
        headers: createAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        localStorage.setItem('user', JSON.stringify(data.data));
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Not authenticated' };
    } catch (error) {
      console.error('Get current user failed:', error);
      // Fallback to local storage
    const userStr = localStorage.getItem('user');
    const user = userStr ? JSON.parse(userStr) : null;
    return user ? { success: true, data: user } : { success: false, message: 'Not authenticated' };
    }
  },

  // POST /api/auth/refresh
  refreshToken: async (): Promise<ApiResponse<{ token: string }>> => {
    console.log('🔄 FLASK API: Refresh token');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
        method: 'POST',
        headers: createAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        setAuthToken(data.data.token);
        return { success: true, data: { token: data.data.token } };
      }
      
      return { success: false, message: data.message || 'Token refresh failed' };
    } catch (error) {
      console.error('Token refresh failed:', error);
      return { success: false, message: 'Network error during token refresh' };
    }
  },

  // POST /api/auth/forgot-password
  forgotPassword: async (email: string): Promise<ApiResponse<{ message: string }>> => {
    console.log('🔑 FLASK API: Forgot password for', email);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/forgot-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });
      
      const data = await response.json();
      return { success: data.success, message: data.message || 'Password reset email sent' };
    } catch (error) {
      console.error('Forgot password failed:', error);
      return { success: false, message: 'Network error during password reset' };
    }
  },

  // POST /api/auth/reset-password
  resetPassword: async (token: string, newPassword: string): Promise<ApiResponse<{ message: string }>> => {
    console.log('🔄 FLASK API: Reset password with token');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, newPassword })
      });
      
      const data = await response.json();
      return { success: data.success, message: data.message || 'Password reset successfully' };
    } catch (error) {
      console.error('Reset password failed:', error);
      return { success: false, message: 'Network error during password reset' };
    }
  },

  // POST /api/auth/verify-email
  verifyEmail: async (token: string): Promise<ApiResponse<{ message: string }>> => {
    console.log('📧 FLASK API: Verify email with token');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/verify-email`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token })
      });
      
      const data = await response.json();
      return { success: data.success, message: data.message || 'Email verified successfully' };
    } catch (error) {
      console.error('Email verification failed:', error);
      return { success: false, message: 'Network error during email verification' };
    }
  },

  // POST /api/auth/guest - Create guest user session
  createGuest: async (): Promise<ApiResponse<User>> => {
    console.log('👤 FLASK API: Create guest user');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/guest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        const { user, access_token } = data.data;
        setAuthToken(access_token);
        localStorage.setItem('user', JSON.stringify(user));
        return { success: true, data: user };
      }
      
      return { success: false, message: data.message || 'Failed to create guest session' };
    } catch (error) {
      console.error('Create guest failed:', error);
      return { success: false, message: 'Network error during guest creation' };
    }
  }
};

// ===================================================================
// ASSESSMENT SERVICE - FLASK BACKEND INTEGRATION
// ===================================================================
export const assessmentService = {
  // POST /assess-symptoms - Create assessment from symptoms (FLASK BACKEND)
  createAssessment: async (inputType: 'voice' | 'text', _transcript: string, symptoms: string[], additionalInfo?: string): Promise<ApiResponse<Assessment>> => {
    console.log('📊 FLASK API: Create assessment with input type:', inputType, 'symptoms:', symptoms);
    
    try {
      const response = await fetch(`${API_BASE_URL}/assess-symptoms`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symptoms, 
          additionalInfo: additionalInfo || '' 
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const assessment = data.assessment;
      
      // Convert Flask response to our Assessment type
      const newAssessment: Assessment = {
        id: assessment.id,
        userId: '1', // TODO: Get from current user context
        date: assessment.date,
        time: assessment.time,
        symptoms: assessment.symptoms,
        severity: assessment.severity,
        recommendations: assessment.recommendations,
        primaryConcerns: assessment.primaryConcerns,
        additionalInfo: assessment.additionalInfo,
        createdAt: assessment.createdAt,
        updatedAt: assessment.createdAt
      };
      
      mockAssessments.unshift(newAssessment);
      return { success: true, data: newAssessment };
    } catch (error) {
      console.error('Assessment creation failed:', error);
      return { success: false, message: 'Failed to create assessment' };
    }
  },

  // GET /api/assessments - Get all assessments (DUMMY - NOT IN BACKEND YET)
  getAssessments: async (): Promise<ApiResponse<Assessment[]>> => {
    console.log('📋 DUMMY API: Get all assessments');
    
    await simulateApiDelay(800);
    return { success: true, data: [...mockAssessments] };
  },

  // GET /api/assessments/:id - Get assessment by ID (DUMMY - NOT IN BACKEND YET)
  getAssessmentById: async (id: string): Promise<ApiResponse<Assessment>> => {
    console.log('🔍 DUMMY API: Get assessment by ID:', id);
    
    await simulateApiDelay(500);
    const assessment = mockAssessments.find(a => a.id === id);
    return assessment 
      ? { success: true, data: assessment }
      : { success: false, message: 'Assessment not found' };
  },

  // PUT /api/assessments/:id - Update assessment (DUMMY - NOT IN BACKEND YET)
  updateAssessment: async (id: string, data: Partial<Assessment>): Promise<ApiResponse<Assessment>> => {
    console.log('✏️ DUMMY API: Update assessment:', id, data);
    
    await simulateApiDelay(1000);
    const index = mockAssessments.findIndex(a => a.id === id);
    if (index !== -1) {
      mockAssessments[index] = { ...mockAssessments[index], ...data };
      return { success: true, data: mockAssessments[index] };
    }
    return { success: false, message: 'Assessment not found' };
  },

  // DELETE /api/assessments/:id - Delete assessment (DUMMY - NOT IN BACKEND YET)
  deleteAssessment: async (id: string): Promise<ApiResponse<null>> => {
    console.log('🗑️ DUMMY API: Delete assessment:', id);
    
    await simulateApiDelay(600);
    const index = mockAssessments.findIndex(a => a.id === id);
    if (index !== -1) {
      mockAssessments.splice(index, 1);
      return { success: true };
    }
    return { success: false, message: 'Assessment not found' };
  },

  // POST /api/assessments/:id/save - Save assessment (DUMMY - NOT IN BACKEND YET)
  saveAssessment: async (id: string): Promise<ApiResponse<{ saved: boolean }>> => {
    console.log('💾 DUMMY API: Save assessment:', id);
    
    await simulateApiDelay(800);
    const assessment = mockAssessments.find(a => a.id === id);
    if (assessment) {
      // Mark as saved in localStorage for persistence
      const savedAssessments = JSON.parse(localStorage.getItem('savedAssessments') || '[]');
      if (!savedAssessments.includes(id)) {
        savedAssessments.push(id);
        localStorage.setItem('savedAssessments', JSON.stringify(savedAssessments));
      }
      return { success: true, data: { saved: true } };
    }
    return { success: false, message: 'Assessment not found' };
  },

  // GET /api/assessments/:id/download - Download assessment (DUMMY - NOT IN BACKEND YET)
  downloadAssessment: async (id: string, format: 'pdf' | 'json' = 'pdf'): Promise<ApiResponse<{ downloadUrl: string }>> => {
    console.log('📥 DUMMY API: Download assessment:', id, format);
    
    await simulateApiDelay(1000);
    const assessment = mockAssessments.find(a => a.id === id);
    if (assessment) {
      if (format === 'json') {
        // Create JSON download
        const dataStr = JSON.stringify(assessment, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        return { success: true, data: { downloadUrl: url } };
      } else {
        // Create PDF download (simulated)
        const pdfContent = `
          Health Assessment Report
          Date: ${assessment.date}
          Time: ${assessment.time}
          
          Symptoms: ${assessment.symptoms.join(', ')}
          Severity: ${assessment.severity}
          
          Recommendations:
          ${assessment.recommendations.map((rec, i) => `${i + 1}. ${rec}`).join('\n')}
          
          Primary Concerns: ${assessment.primaryConcerns.join(', ')}
        `;
        const dataBlob = new Blob([pdfContent], { type: 'text/plain' });
        const url = URL.createObjectURL(dataBlob);
        return { success: true, data: { downloadUrl: url } };
      }
    }
    return { success: false, message: 'Assessment not found' };
  },

  // GET /api/assessments/:id/details - Get assessment details (DUMMY - NOT IN BACKEND YET)
  getAssessmentDetails: async (id: string): Promise<ApiResponse<Assessment & { 
    createdAt: string;
    updatedAt: string;
    isSaved: boolean;
    downloadCount: number;
  }>> => {
    console.log('🔍 DUMMY API: Get assessment details:', id);
    
    await simulateApiDelay(500);
    const assessment = mockAssessments.find(a => a.id === id);
    if (assessment) {
      const savedAssessments = JSON.parse(localStorage.getItem('savedAssessments') || '[]');
      const downloadCount = parseInt(localStorage.getItem(`downloadCount_${id}`) || '0');
      
      const details = {
        ...assessment,
        createdAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
        updatedAt: new Date().toISOString(),
        isSaved: savedAssessments.includes(id),
        downloadCount
      };
      
      return { success: true, data: details };
    }
    return { success: false, message: 'Assessment not found' };
  }
};

// ===================================================================
// VOICE SERVICE - FLASK BACKEND INTEGRATION
// ===================================================================
export const voiceService = {
  // POST /upload-audio - Upload audio file (FLASK BACKEND)
  uploadAudio: async (audioBlob: Blob): Promise<ApiResponse<{ audioId: string; transcript: string; symptoms: string[]; severity: string; recommendations: string[] }>> => {
    console.log('🎤 FLASK API: Upload audio file');
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'audio.wav');
      
      const response = await fetch(`${API_BASE_URL}/upload-audio`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return { 
        success: true, 
        data: {
          audioId: data.audioId,
          transcript: data.transcript,
          symptoms: data.symptoms,
          severity: data.severity,
          recommendations: data.recommendations
        }
      };
    } catch (error) {
      console.error('Audio upload failed:', error);
      return { success: false, message: 'Failed to upload audio file' };
    }
  },

  // POST /upload-text - Process text input (FLASK BACKEND)
  processTextInput: async (text: string): Promise<ApiResponse<{ symptoms: string[]; severity: string; recommendations: string[]; assessmentId: string }>> => {
    console.log('📝 FLASK API: Process text input');
    
    try {
      const response = await fetch(`${API_BASE_URL}/upload-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return { 
        success: true, 
        data: {
          symptoms: data.symptoms,
          severity: data.severity,
          recommendations: data.recommendations,
          assessmentId: data.assessmentId
        }
      };
    } catch (error) {
      console.error('Text processing failed:', error);
      return { success: false, message: 'Failed to process text input' };
    }
  },

  // POST /assess-symptoms - Generate health assessment (FLASK BACKEND)
  assessSymptoms: async (symptoms: string[], additionalInfo?: string): Promise<ApiResponse<Assessment>> => {
    console.log('🧠 FLASK API: Assess symptoms:', symptoms);
    
    try {
      const response = await fetch(`${API_BASE_URL}/assess-symptoms`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          symptoms, 
          additionalInfo: additionalInfo || '' 
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const assessment = data.assessment;
      
      // Convert Flask response to our Assessment type
      const newAssessment: Assessment = {
        id: assessment.id,
        userId: '1', // TODO: Get from current user context
        date: assessment.date,
        time: assessment.time,
        symptoms: assessment.symptoms,
        severity: assessment.severity,
        recommendations: assessment.recommendations,
        primaryConcerns: assessment.primaryConcerns,
        additionalInfo: assessment.additionalInfo,
        createdAt: assessment.createdAt,
        updatedAt: assessment.createdAt
      };
      
      return { success: true, data: newAssessment };
    } catch (error) {
      console.error('Symptom assessment failed:', error);
      return { success: false, message: 'Failed to assess symptoms' };
    }
  }
};

// ===================================================================
// CHAT/ASSISTANT SERVICE - FLASK BACKEND INTEGRATION
// ===================================================================
export const chatService = {
  // GET /api/chat/sessions - Get chat sessions
  getChatSessions: async (): Promise<ApiResponse<ChatSession[]>> => {
    console.log('💬 FLASK API: Get chat sessions');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/sessions`, {
        method: 'GET',
        headers: createAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to get chat sessions' };
    } catch (error) {
      console.error('Get chat sessions failed:', error);
      return { success: false, message: 'Network error while getting chat sessions' };
    }
  },

  // POST /api/chat/sessions - Create chat session
  createChatSession: async (title?: string, language?: string): Promise<ApiResponse<ChatSession>> => {
    console.log('🆕 FLASK API: Create chat session');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/sessions`, {
        method: 'POST',
        headers: createAuthHeaders(),
        body: JSON.stringify({ 
          title: title || 'New Chat',
          language: language || 'en'
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to create chat session' };
    } catch (error) {
      console.error('Create chat session failed:', error);
      return { success: false, message: 'Network error while creating chat session' };
    }
  },

  // GET /api/chat/sessions/:id/messages - Get chat messages
  getChatMessages: async (sessionId: string): Promise<ApiResponse<Message[]>> => {
    console.log('📨 FLASK API: Get chat messages for session:', sessionId);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessionId}/messages`, {
        method: 'GET',
        headers: createAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to get chat messages' };
    } catch (error) {
      console.error('Get chat messages failed:', error);
      return { success: false, message: 'Network error while getting chat messages' };
    }
  },

  // POST /chat/message - Send message to Flask backend
  sendMessage: async (sessionId: string, message: string): Promise<ApiResponse<{userMessage: Message, response: Message}>> => {
    console.log('📤 FLASK API: Send message to session:', sessionId, message);
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat/message`, {
        method: 'POST',
        headers: createAuthHeaders(),
        body: JSON.stringify({ 
          message,
          sessionId
        })
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to send message' };
    } catch (error) {
      console.error('Chat message failed:', error);
      return { success: false, message: 'Network error while sending message' };
    }
  },

  // POST /chat/voice-message - Send voice message to backend
  sendVoiceMessage: async (sessionId: string, audioBlob: Blob, duration: number): Promise<ApiResponse<{userMessage: Message, response: Message}>> => {
    console.log('🎤 FLASK API: Send voice message to session:', sessionId, 'duration:', duration);
    
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'voice-message.webm');
      formData.append('duration', duration.toString());
      formData.append('sessionId', sessionId);
      
      const response = await fetch(`${API_BASE_URL}/chat/voice-message`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${getAuthToken()}`
        },
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to send voice message' };
    } catch (error) {
      console.error('Voice message failed:', error);
      return { success: false, message: 'Network error while sending voice message' };
    }
  },

  // PUT /api/chat/sessions/:id - Update chat session (rename, etc.)
  updateChatSession: async (sessionId: string, updates: { title?: string }): Promise<ApiResponse<ChatSession>> => {
    console.log('✏️ FLASK API: Update chat session:', sessionId, updates);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessionId}`, {
        method: 'PUT',
        headers: createAuthHeaders(),
        body: JSON.stringify(updates)
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to update chat session' };
    } catch (error) {
      console.error('Update chat session failed:', error);
      return { success: false, message: 'Network error while updating chat session' };
    }
  },

  // DELETE /api/chat/sessions/:id - Delete chat session
  deleteChatSession: async (sessionId: string): Promise<ApiResponse<null>> => {
    console.log('🗑️ FLASK API: Delete chat session:', sessionId);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: createAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true };
      }
      
      return { success: false, message: data.message || 'Failed to delete chat session' };
    } catch (error) {
      console.error('Delete chat session failed:', error);
      return { success: false, message: 'Network error while deleting chat session' };
    }
  },

  // DELETE /api/chat/sessions/:sessionId/messages/:messageId - Delete specific message
  deleteMessage: async (sessionId: string, messageId: string): Promise<ApiResponse<null>> => {
    console.log('🗑️ FLASK API: Delete message:', sessionId, messageId);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/sessions/${sessionId}/messages/${messageId}`, {
        method: 'DELETE',
        headers: createAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true };
      }
      
      return { success: false, message: data.message || 'Failed to delete message' };
    } catch (error) {
      console.error('Delete message failed:', error);
      return { success: false, message: 'Network error while deleting message' };
    }
  },

  // POST /chat/file-upload - Upload file and get AI response
  uploadFile: async (sessionId: string, file: File): Promise<ApiResponse<Message>> => {
    console.log('📎 FLASK API: Upload file to session:', sessionId, 'file:', file.name);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('sessionId', sessionId);
      
      const response = await fetch(`${API_BASE_URL}/chat/file-upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${getAuthToken()}`
        },
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        return { success: true, data: data.data.response };
      }
      
      return { success: false, message: data.message || 'Failed to upload file' };
    } catch (error) {
      console.error('File upload failed:', error);
      return { success: false, message: 'Network error while uploading file' };
    }
  },

  // POST /chat/symptoms - Send selected symptoms and get AI response
  sendSymptoms: async (sessionId: string, symptoms: string[]): Promise<ApiResponse<Message>> => {
    console.log('🩺 FLASK API: Send symptoms to session:', sessionId, 'symptoms:', symptoms);
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat/symptoms`, {
        method: 'POST',
        headers: createAuthHeaders(),
        body: JSON.stringify({ 
          symptoms,
          sessionId
        })
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        return { success: true, data: data.data.response };
      }
      
      return { success: false, message: data.message || 'Failed to process symptoms' };
    } catch (error) {
      console.error('Symptoms processing failed:', error);
      return { success: false, message: 'Network error while processing symptoms' };
    }
  },

  // POST /api/voice-chat/response - Generate voice response for voice-to-voice chat
  generateVoiceResponse: async (message: string, sessionId: string): Promise<ApiResponse<Message>> => {
    console.log('🎤 FLASK API: Generate voice response for message:', message.substring(0, 50) + '...');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/voice-chat/response`, {
        method: 'POST',
        headers: createAuthHeaders(),
        body: JSON.stringify({ 
          message,
          sessionId
        })
      });
      
      const data = await response.json();
      
      if (data.success && data.data) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to generate voice response' };
    } catch (error) {
      console.error('Voice response generation failed:', error);
      return { success: false, message: 'Network error while generating voice response' };
    }
  }
};

// ===================================================================
// USER PROFILE SERVICE - FLASK BACKEND INTEGRATION
// ===================================================================
export const userService = {
  // GET /api/users/profile - Get user profile
  getProfile: async (): Promise<ApiResponse<User>> => {
    console.log('👤 FLASK API: Get user profile');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/profile`, {
        method: 'GET',
        headers: createAuthHeaders()
      });
      
      const data = await response.json();
      
      if (data.success) {
        localStorage.setItem('user', JSON.stringify(data.data));
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to get user profile' };
    } catch (error) {
      console.error('Get user profile failed:', error);
      // Fallback to local storage
      const userStr = localStorage.getItem('user');
      const user = userStr ? JSON.parse(userStr) : null;
      return user ? { success: true, data: user } : { success: false, message: 'User not found' };
    }
  },

  // PUT /api/users/profile - Update user profile
  updateProfile: async (userData: Partial<User>): Promise<ApiResponse<User>> => {
    console.log('✏️ FLASK API: Update user profile:', userData);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/profile`, {
        method: 'PUT',
        headers: createAuthHeaders(),
        body: JSON.stringify(userData)
      });
      
      const data = await response.json();
      
      if (data.success) {
        localStorage.setItem('user', JSON.stringify(data.data));
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to update user profile' };
    } catch (error) {
      console.error('Update user profile failed:', error);
      return { success: false, message: 'Network error while updating profile' };
    }
  },

  // POST /api/users/change-password - Change password
  changePassword: async (currentPassword: string, newPassword: string): Promise<ApiResponse<null>> => {
    console.log('🔒 FLASK API: Change password');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/users/change-password`, {
        method: 'POST',
        headers: createAuthHeaders(),
        body: JSON.stringify({
          currentPassword,
          newPassword
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true };
      }
      
      return { success: false, message: data.message || 'Failed to change password' };
    } catch (error) {
      console.error('Change password failed:', error);
      return { success: false, message: 'Network error while changing password' };
    }
  }
};

// ===================================================================
// TEXT-TO-SPEECH SERVICE - FLASK BACKEND INTEGRATION
// ===================================================================
export const ttsService = {
  // POST /api/tts - Convert text to speech
  textToSpeech: async (text: string, language: string = 'en'): Promise<ApiResponse<{ audioUrl: string; duration: number; text: string; language: string }>> => {
    console.log('🔊 FLASK API: Text to speech for:', text.substring(0, 50) + '...');
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/tts`, {
        method: 'POST',
        headers: createAuthHeaders(),
        body: JSON.stringify({ text, language })
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to generate speech' };
    } catch (error) {
      console.error('Text-to-speech failed:', error);
      return { success: false, message: 'Network error while generating speech' };
    }
  }
};

// ===================================================================
// FILE UPLOAD SERVICE - FLASK BACKEND INTEGRATION
// ===================================================================
export const fileService = {
  // POST /api/upload - Upload files
  uploadFile: async (file: File): Promise<ApiResponse<{ fileId: string; filename: string; originalName: string; fileType: string; fileSize: number; uploadedAt: string }>> => {
    console.log('📁 FLASK API: Upload file:', file.name);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${getAuthToken()}`
        },
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success) {
        return { success: true, data: data.data };
      }
      
      return { success: false, message: data.message || 'Failed to upload file' };
    } catch (error) {
      console.error('File upload failed:', error);
      return { success: false, message: 'Network error while uploading file' };
    }
  }
};

// ===================================================================
// EXPORT ALL SERVICES
// ===================================================================
export default {
  authService,
  assessmentService,
  voiceService,
  chatService,
  userService,
  ttsService,
  fileService
};