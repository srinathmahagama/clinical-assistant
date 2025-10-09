import React from 'react';
import { FileText, Image, Video, File, Download } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import MessageOptions from '../MessageOptions/MessageOptions';

interface FileMessageProps {
  fileName: string;
  fileType: string;
  fileSize: number;
  fileUrl: string;
  isUser: boolean;
  timestamp: string;
  onDelete?: () => void;
}

const FileMessage: React.FC<FileMessageProps> = ({
  fileName,
  fileType,
  fileSize,
  fileUrl,
  isUser,
  timestamp,
  onDelete
}) => {
  const { theme } = useTheme();
  
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) {
      return <Image className="w-6 h-6" />;
    } else if (type.startsWith('video/')) {
      return <Video className="w-6 h-6" />;
    } else if (type === 'application/pdf') {
      return <FileText className="w-6 h-6" />;
    } else {
      return <File className="w-6 h-6" />;
    }
  };

  const getFileColor = (type: string) => {
    if (type.startsWith('image/')) {
      return 'text-green-600';
    } else if (type.startsWith('video/')) {
      return 'text-purple-600';
    } else if (type === 'application/pdf') {
      return 'text-red-600';
    } else {
      return 'text-gray-600';
    }
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = fileUrl;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handlePreview = () => {
    if (fileType.startsWith('image/') || fileType.startsWith('video/') || fileType === 'application/pdf') {
      window.open(fileUrl, '_blank');
    } else {
      handleDownload();
    }
  };

  return (
    <div className={`flex items-center space-x-2 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
      {/* File Message Bubble */}
      <div className={`flex items-center space-x-3 px-4 py-3 rounded-xl max-w-md ${
        isUser
          ? theme === 'dark' 
            ? 'bg-blue-900/45 text-white'
            : 'bg-blue-800/45 text-white'
          : theme === 'dark'
            ? 'bg-slate-700 text-slate-100'
            : 'bg-gray-200 text-gray-800'
      }`}>
        {/* File Icon */}
        <div className={`${getFileColor(fileType)}`}>
          {getFileIcon(fileType)}
        </div>

        {/* File Info */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{fileName}</p>
          <p className={`text-xs ${isUser ? 'text-blue-100' : 'text-gray-500'}`}>
            {formatFileSize(fileSize)}
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-1">
          <button
            onClick={handlePreview}
            className={`p-1 rounded-full transition-colors ${
              isUser
                ? 'hover:bg-blue-600'
                : 'hover:bg-gray-300'
            }`}
            title="Preview/Download"
          >
            <Download className="w-4 h-4" />
          </button>

          {onDelete && (
            <MessageOptions 
              onDelete={onDelete}
              isUser={isUser}
            />
          )}
        </div>
      </div>

      {/* Timestamp */}
      <div className={`text-xs ${
        isUser ? 'text-blue-100' : 'text-gray-500'
      }`}>
        {timestamp}
      </div>
    </div>
  );
};

export default FileMessage;
