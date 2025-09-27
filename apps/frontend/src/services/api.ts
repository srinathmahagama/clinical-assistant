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
const _handleApiResponse = async (response: Response): Promise<any> => {
  if (response.status === 401) {
    // Token expired, try to refresh
    const refreshResponse = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: createAuthHeaders()
    });
    
    if (refreshResponse.ok) {
      const refreshData = await refreshResponse.json();
      if (refreshData.success && refreshData.data?.token) {
        setAuthToken(refreshData.data.token);
        // Retry original request with new token
        return fetch(response.url, {
          method: 'GET', // Default method for retry
          headers: createAuthHeaders()
        }).then(res => res.json());
      }
    }
    
    // Refresh failed, redirect to login
    removeAuthToken();
    localStorage.removeItem('user');
    window.location.href = '/login';
    throw new Error('Authentication failed');
  }
  
  return response.json();
};

// ===================================================================
// DUMMY DATA - FOR NON-IMPLEMENTED ENDPOINTS ONLY
// ===================================================================
const mockUsers: User[] = [
  {
    id: '1',
    firstName: 'Nimash',
    lastName: 'Silva',
    email: 'nimash@example.com'
  },
  {
    id: '2',
    firstName: 'John',
    lastName: 'Doe',
    email: 'john@example.com'
  }
];

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

const mockChatHistory: Message[] = [
  {
    id: '1',
    text: "Hello! I'm your health assistant. I can help you understand your symptoms, explain your assessment results, or answer health questions. How can I help you today?",
    isUser: false,
    timestamp: '10:00 p.m',
    createdAt: '2025-01-16T22:00:00Z'
  }
];

const mockChatSessions: ChatSession[] = [
  {
    id: '1',
    userId: '1',
    messages: mockChatHistory,
    createdAt: '2025-01-16T10:00:00Z',
    updatedAt: '2025-01-16T10:00:00Z'
  }
];

// ===================================================================
// HELPER FUNCTIONS
// ===================================================================

// Simulate API delay
const simulateApiDelay = (ms: number = 1000) => new Promise(resolve => setTimeout(resolve, ms));

// Generate random ID
const generateId = () => Date.now().toString() + Math.random().toString(36).substr(2, 9);

// ===================================================================
// AUTHENTICATION SERVICE - DUMMY IMPLEMENTATION (NOT IN BACKEND YET)
// ===================================================================
export const authService = {
  // POST /api/auth/login
  login: async (email: string, password: string): Promise<ApiResponse<User>> => {
    console.log('🔐 DUMMY API: Login attempt for', email);
    
    await simulateApiDelay(1000);
    
    const user = mockUsers.find(u => u.email === email);
    if (user && password === 'password123') {
      const token = 'dummy-jwt-token-' + Date.now();
      setAuthToken(token);
      localStorage.setItem('user', JSON.stringify(user));
      return { success: true, data: user };
    }
    
    return { success: false, message: 'Invalid credentials' };
  },

  // POST /api/auth/register
  register: async (userData: Omit<User, 'id'> & { password: string }): Promise<ApiResponse<User>> => {
    console.log('📝 DUMMY API: Register attempt for', userData.email);
    
    await simulateApiDelay(1200);
    
    // Check if user already exists
    if (mockUsers.find(u => u.email === userData.email)) {
      return { success: false, message: 'User already exists' };
    }
    
    const newUser: User = {
      id: generateId(),
      firstName: userData.firstName,
      lastName: userData.lastName,
      email: userData.email
    };
    
    mockUsers.push(newUser);
    localStorage.setItem('user', JSON.stringify(newUser));
    localStorage.setItem('authToken', 'dummy-token-' + Date.now());
    return { success: true, data: newUser };
  },

  // POST /api/auth/logout
  logout: async (): Promise<ApiResponse<null>> => {
    console.log('🚪 DUMMY API: Logout');
    
    removeAuthToken();
    localStorage.removeItem('user');
    return { success: true };
  },

  // GET /api/auth/me
  getCurrentUser: async (): Promise<ApiResponse<User>> => {
    console.log('👤 DUMMY API: Get current user');
    
    const userStr = localStorage.getItem('user');
    const user = userStr ? JSON.parse(userStr) : null;
    return user ? { success: true, data: user } : { success: false, message: 'Not authenticated' };
  },

  // POST /api/auth/refresh
  refreshToken: async (): Promise<ApiResponse<{ token: string }>> => {
    console.log('🔄 DUMMY API: Refresh token');
    
    await simulateApiDelay(500);
    return { success: true, data: { token: 'dummy-refreshed-token-' + Date.now() } };
  },

  // POST /api/auth/forgot-password
  forgotPassword: async (email: string): Promise<ApiResponse<{ message: string }>> => {
    console.log('🔑 DUMMY API: Forgot password for', email);
    
    await simulateApiDelay(1000);
    return { success: true, data: { message: 'Password reset email sent' } };
  },

  // POST /api/auth/reset-password
  resetPassword: async (_token: string, _newPassword: string): Promise<ApiResponse<{ message: string }>> => {
    console.log('🔄 DUMMY API: Reset password with token');
    
    await simulateApiDelay(1000);
    return { success: true, data: { message: 'Password reset successfully' } };
  },

  // POST /api/auth/verify-email
  verifyEmail: async (_token: string): Promise<ApiResponse<{ message: string }>> => {
    console.log('📧 DUMMY API: Verify email with token');
    
    await simulateApiDelay(800);
    return { success: true, data: { message: 'Email verified successfully' } };
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
  // GET /api/chat/sessions - Get chat sessions (DUMMY - NOT IN BACKEND YET)
  getChatSessions: async (): Promise<ApiResponse<ChatSession[]>> => {
    console.log('💬 DUMMY API: Get chat sessions');
    
    await simulateApiDelay(600);
    return { success: true, data: [...mockChatSessions] };
  },

  // POST /api/chat/sessions - Create chat session (DUMMY - NOT IN BACKEND YET)
  createChatSession: async (): Promise<ApiResponse<ChatSession>> => {
    console.log('🆕 DUMMY API: Create chat session');
    
    await simulateApiDelay(800);
    const newSession: ChatSession = {
      id: generateId(),
      userId: '1',
      messages: [...mockChatHistory],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    mockChatSessions.push(newSession);
    return { success: true, data: newSession };
  },

  // GET /api/chat/sessions/:id/messages - Get chat messages (DUMMY - NOT IN BACKEND YET)
  getChatMessages: async (sessionId: string): Promise<ApiResponse<Message[]>> => {
    console.log('📨 DUMMY API: Get chat messages for session:', sessionId);
    
    await simulateApiDelay(500);
    const session = mockChatSessions.find(s => s.id === sessionId);
    return session 
      ? { success: true, data: session.messages }
      : { success: false, message: 'Session not found' };
  },

  // POST /chat/message - Send message to Flask backend (FLASK BACKEND)
  sendMessage: async (sessionId: string, message: string): Promise<ApiResponse<Message>> => {
    console.log('📤 FLASK API: Send message to session:', sessionId, message);
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const aiMessage: Message = {
        id: data.response.id,
        text: data.response.text,
        isUser: false,
        timestamp: data.response.timestamp,
        createdAt: data.response.createdAt
      };
      
      // Add message to local session for now
      const session = mockChatSessions.find(s => s.id === sessionId);
      if (session) {
        session.messages.push(aiMessage);
        session.updatedAt = new Date().toISOString();
      }
      
      return { success: true, data: aiMessage };
    } catch (error) {
      console.error('Chat message failed:', error);
      return { success: false, message: 'Failed to send chat message' };
    }
  },

  // DELETE /api/chat/sessions/:id - Delete chat session (DUMMY - NOT IN BACKEND YET)
  deleteChatSession: async (sessionId: string): Promise<ApiResponse<null>> => {
    console.log('🗑️ DUMMY API: Delete chat session:', sessionId);
    
    await simulateApiDelay(600);
    const index = mockChatSessions.findIndex(s => s.id === sessionId);
    if (index !== -1) {
      mockChatSessions.splice(index, 1);
      return { success: true };
    }
    return { success: false, message: 'Session not found' };
  }
};

// ===================================================================
// USER PROFILE SERVICE - DUMMY IMPLEMENTATION (NOT IN BACKEND YET)
// ===================================================================
export const userService = {
  // GET /api/users/profile - Get user profile
  getProfile: async (): Promise<ApiResponse<User>> => {
    console.log('👤 DUMMY API: Get user profile');
    
    await simulateApiDelay(500);
    const userStr = localStorage.getItem('user');
    const user = userStr ? JSON.parse(userStr) : null;
    return user ? { success: true, data: user } : { success: false, message: 'User not found' };
  },

  // PUT /api/users/profile - Update user profile
  updateProfile: async (userData: Partial<User>): Promise<ApiResponse<User>> => {
    console.log('✏️ DUMMY API: Update user profile:', userData);
    
    await simulateApiDelay(1000);
    const userStr = localStorage.getItem('user');
    if (userStr) {
      const user = JSON.parse(userStr);
      const updatedUser = { ...user, ...userData };
      localStorage.setItem('user', JSON.stringify(updatedUser));
      return { success: true, data: updatedUser };
    }
    return { success: false, message: 'User not found' };
  },

  // POST /api/users/change-password - Change password
  changePassword: async (_currentPassword: string, _newPassword: string): Promise<ApiResponse<null>> => {
    console.log('🔒 DUMMY API: Change password');
    
    await simulateApiDelay(1200);
    return { success: true };
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
  userService
};