import React, { useRef, useState, useEffect } from 'react';
import { Paperclip, X, Image, FileText } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  onClose: () => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, onClose }) => {
  const { theme } = useTheme();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const [dragOver, setDragOver] = useState(false);

  // Close modal when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [onClose]);

  const handleFileSelect = (file: File) => {
    // Validate file type - only images and PDFs
    const allowedTypes = [
      'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml',
      'application/pdf'
    ];

    if (!allowedTypes.includes(file.type)) {
      alert('File type not supported. Please select an image or PDF document only.');
      return;
    }

    // Validate file size (10MB limit)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      alert('File size too large. Please select a file smaller than 10MB.');
      return;
    }

    onFileSelect(file);
    onClose();
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const getFileTypeIcon = (type: string) => {
    if (type.startsWith('image/')) {
      return <Image className="w-8 h-8 text-green-500" />;
    } else if (type === 'application/pdf') {
      return <FileText className="w-8 h-8 text-red-500" />;
    } else {
      return <File className="w-8 h-8 text-gray-500" />;
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div ref={modalRef} className={`rounded-2xl max-w-sm w-full p-4 shadow-2xl transition-colors duration-300 ${
        theme === 'dark' ? 'bg-slate-800 border border-slate-600' : 'bg-white border border-gray-200'
      }`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className={`text-lg font-bold ${
            theme === 'dark' ? 'text-white' : 'text-gray-800'
          }`}>Share File</h3>
          <button
            onClick={onClose}
            className={`p-2 rounded-full transition-colors ${
              theme === 'dark'
                ? 'text-gray-400 hover:text-white hover:bg-slate-700'
                : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
            }`}
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* File Types Info */}
        <div className="mb-4">
          <p className={`mb-3 text-sm ${
            theme === 'dark' ? 'text-gray-300' : 'text-gray-600'
          }`}>Supported file types:</p>
          <div className="grid grid-cols-1 gap-2">
            <div className={`flex items-center space-x-2 p-2 rounded-lg ${
              theme === 'dark' ? 'bg-slate-700' : 'bg-gray-50'
            }`}>
              <Image className="w-4 h-4 text-green-500" />
              <span className={`text-xs ${
                theme === 'dark' ? 'text-gray-200' : 'text-gray-700'
              }`}>Images (JPG, PNG, GIF, WebP, SVG)</span>
            </div>
            <div className={`flex items-center space-x-2 p-2 rounded-lg ${
              theme === 'dark' ? 'bg-slate-700' : 'bg-gray-50'
            }`}>
              <FileText className="w-4 h-4 text-red-500" />
              <span className={`text-xs ${
                theme === 'dark' ? 'text-gray-200' : 'text-gray-700'
              }`}>PDF Documents</span>
            </div>
          </div>
          <p className={`text-xs mt-2 ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
          }`}>Maximum file size: 10MB</p>
        </div>

        {/* Drop Zone */}
        <div
          className={`border-2 border-dashed rounded-xl p-6 text-center transition-colors ${
            dragOver
              ? theme === 'dark'
                ? 'border-blue-400 bg-blue-900/30'
                : 'border-blue-500 bg-blue-50'
              : theme === 'dark'
                ? 'border-slate-500 hover:border-slate-400'
                : 'border-gray-300 hover:border-gray-400'
          }`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <Paperclip className={`w-8 h-8 mx-auto mb-3 ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-400'
          }`} />
          <p className={`mb-2 text-sm ${
            theme === 'dark' ? 'text-gray-300' : 'text-gray-600'
          }`}>Drag and drop a file here</p>
          <p className={`text-xs mb-3 ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-500'
          }`}>or</p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors text-sm"
          >
            Choose File
          </button>
        </div>

        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleFileInputChange}
          accept="image/*,.pdf"
          className="hidden"
        />
      </div>
    </div>
  );
};

export default FileUpload;
