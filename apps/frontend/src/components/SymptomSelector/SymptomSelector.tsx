import React, { useState, useEffect, useRef } from 'react';
import { X, ArrowLeft, Check, MoreVertical } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface SymptomImage {
  id: string;
  name: string;
  image: string;
  tags: string[];
}

interface SymptomCategory {
  id: string;
  name: string;
  icon: string;
  images: SymptomImage[];
}

interface SymptomSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSymptoms: (symptoms: SymptomImage[]) => void;
  isLoading?: boolean;
}

const SymptomSelector: React.FC<SymptomSelectorProps> = ({
  isOpen,
  onClose,
  onSelectSymptoms,
  isLoading = false
}) => {
  const { theme } = useTheme();
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedSymptoms, setSelectedSymptoms] = useState<SymptomImage[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Comprehensive symptom categories with real images
  const categories: SymptomCategory[] = [
    {
      id: 'head',
      name: 'Head & Face',
      icon: '🧠',
      images: [
        { id: 'headache', name: 'Headache', image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200&h=200&fit=crop&crop=face', tags: ['headache', 'pain', 'head'] },
        { id: 'migraine', name: 'Migraine', image: 'https://images.unsplash.com/photo-1517363898874-2a9114134d97?w=200&h=200&fit=crop&crop=face', tags: ['migraine', 'severe headache', 'head'] },
        { id: 'dizziness', name: 'Dizziness', image: 'https://images.unsplash.com/photo-1517486808906-6ca8b3f04846?w=200&h=200&fit=crop&crop=face', tags: ['dizziness', 'vertigo', 'head'] },
        { id: 'facial-pain', name: 'Facial Pain', image: 'https://images.unsplash.com/photo-1560250097-0b93528c311a?w=200&h=200&fit=crop&crop=face', tags: ['facial pain', 'face', 'head'] },
        { id: 'eye-pain', name: 'Eye Pain', image: 'https://images.unsplash.com/photo-1582213733776-fa1923c5c528?w=200&h=200&fit=crop&crop=face', tags: ['eye pain', 'vision', 'head'] },
        { id: 'ear-pain', name: 'Ear Pain', image: 'https://images.unsplash.com/photo-1505751172876-fa1923c5c528?w=200&h=200&fit=crop&crop=face', tags: ['ear pain', 'hearing', 'head'] },
        { id: 'tooth-pain', name: 'Tooth Pain', image: 'https://images.unsplash.com/photo-1572448804125-2be657d7e5b1?w=200&h=200&fit=crop&crop=face', tags: ['tooth pain', 'dental', 'mouth'] },
        { id: 'neck-pain', name: 'Neck Pain', image: 'https://images.unsplash.com/photo-1523539664694-3d7e2f4a0b0e?w=200&h=200&fit=crop&crop=face', tags: ['neck pain', 'stiffness', 'head'] }
      ]
    },
    {
      id: 'chest',
      name: 'Chest & Heart',
      icon: '❤️',
      images: [
        { id: 'chest-pain', name: 'Chest Pain', image: 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=center', tags: ['chest pain', 'heart', 'chest'] },
        { id: 'shortness-breath', name: 'Shortness of Breath', image: 'https://images.unsplash.com/photo-1581578731547-07e7177d9f9f?w=200&h=200&fit=crop&crop=center', tags: ['breathing', 'lungs', 'chest'] },
        { id: 'heart-palpitations', name: 'Heart Palpitations', image: 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=200&h=200&fit=crop&crop=center', tags: ['heart', 'palpitations', 'chest'] },
        { id: 'cough', name: 'Cough', image: 'https://images.unsplash.com/photo-1582774007591-5ca6b4b8d2c9?w=200&h=200&fit=crop&crop=center', tags: ['cough', 'throat', 'chest'] },
        { id: 'wheezing', name: 'Wheezing', image: 'https://images.unsplash.com/photo-1576092768241-dec231879af5?w=200&h=200&fit=crop&crop=center', tags: ['wheezing', 'breathing', 'lungs'] },
        { id: 'chest-tightness', name: 'Chest Tightness', image: 'https://images.unsplash.com/photo-1606890658317-7d14490b76fd?w=200&h=200&fit=crop&crop=center', tags: ['chest tightness', 'pressure', 'chest'] }
      ]
    },
    {
      id: 'abdomen',
      name: 'Abdomen & Stomach',
      icon: '🫀',
      images: [
        { id: 'stomach-pain', name: 'Stomach Pain', image: 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=200&h=200&fit=crop&crop=center', tags: ['stomach pain', 'abdomen', 'digestive'] },
        { id: 'nausea', name: 'Nausea', image: 'https://images.unsplash.com/photo-1582774007591-5ca6b4b8d2c9?w=200&h=200&fit=crop&crop=face', tags: ['nausea', 'sick', 'stomach'] },
        { id: 'diarrhea', name: 'Diarrhea', image: 'https://images.unsplash.com/photo-1606890658317-7d14490b76fd?w=200&h=200&fit=crop&crop=center', tags: ['diarrhea', 'digestive', 'stomach'] },
        { id: 'constipation', name: 'Constipation', image: 'https://images.unsplash.com/photo-1576092768241-dec231879af5?w=200&h=200&fit=crop&crop=center', tags: ['constipation', 'digestive', 'stomach'] },
        { id: 'vomiting', name: 'Vomiting', image: 'https://images.unsplash.com/photo-1582774007591-5ca6b4b8d2c9?w=200&h=200&fit=crop&crop=face', tags: ['vomiting', 'nausea', 'stomach'] },
        { id: 'bloating', name: 'Bloating', image: 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=center', tags: ['bloating', 'gas', 'stomach'] },
        { id: 'heartburn', name: 'Heartburn', image: 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=200&h=200&fit=crop&crop=center', tags: ['heartburn', 'acid reflux', 'stomach'] },
        { id: 'appetite-loss', name: 'Loss of Appetite', image: 'https://images.unsplash.com/photo-1581578731547-07e7177d9f9f?w=200&h=200&fit=crop&crop=center', tags: ['appetite loss', 'hunger', 'stomach'] }
      ]
    },
    {
      id: 'skin',
      name: 'Skin & Rashes',
      icon: '🦠',
      images: [
        { id: 'rash', name: 'Skin Rash', image: 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=center', tags: ['rash', 'skin', 'irritation'] },
        { id: 'hives', name: 'Hives', image: 'https://images.unsplash.com/photo-1576092768241-dec231879af5?w=200&h=200&fit=crop&crop=center', tags: ['hives', 'allergic reaction', 'skin'] },
        { id: 'acne', name: 'Acne', image: 'https://images.unsplash.com/photo-1582213733776-fa1923c5c528?w=200&h=200&fit=crop&crop=face', tags: ['acne', 'skin', 'face'] },
        { id: 'dry-skin', name: 'Dry Skin', image: 'https://images.unsplash.com/photo-1606890658317-7d14490b76fd?w=200&h=200&fit=crop&crop=center', tags: ['dry skin', 'flaky', 'skin'] },
        { id: 'itching', name: 'Itching', image: 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=200&h=200&fit=crop&crop=center', tags: ['itching', 'pruritus', 'skin'] },
        { id: 'swelling', name: 'Swelling', image: 'https://images.unsplash.com/photo-1581578731547-07e7177d9f9f?w=200&h=200&fit=crop&crop=center', tags: ['swelling', 'edema', 'skin'] },
        { id: 'bruising', name: 'Bruising', image: 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=center', tags: ['bruising', 'contusion', 'skin'] },
        { id: 'discoloration', name: 'Skin Discoloration', image: 'https://images.unsplash.com/photo-1576092768241-dec231879af5?w=200&h=200&fit=crop&crop=center', tags: ['discoloration', 'pigmentation', 'skin'] }
      ]
    },
    {
      id: 'muscle',
      name: 'Muscles & Joints',
      icon: '🦴',
      images: [
        { id: 'back-pain', name: 'Back Pain', image: 'https://images.unsplash.com/photo-1523539664694-3d7e2f4a0b0e?w=200&h=200&fit=crop&crop=center', tags: ['back pain', 'muscle', 'spine'] },
        { id: 'joint-pain', name: 'Joint Pain', image: 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=center', tags: ['joint pain', 'arthritis', 'muscle'] },
        { id: 'muscle-cramp', name: 'Muscle Cramp', image: 'https://images.unsplash.com/photo-1576092768241-dec231879af5?w=200&h=200&fit=crop&crop=center', tags: ['muscle cramp', 'pain', 'muscle'] },
        { id: 'stiffness', name: 'Stiffness', image: 'https://images.unsplash.com/photo-1582213733776-fa1923c5c528?w=200&h=200&fit=crop&crop=center', tags: ['stiffness', 'rigidity', 'muscle'] },
        { id: 'weakness', name: 'Muscle Weakness', image: 'https://images.unsplash.com/photo-1606890658317-7d14490b76fd?w=200&h=200&fit=crop&crop=center', tags: ['weakness', 'strength', 'muscle'] },
        { id: 'swollen-joints', name: 'Swollen Joints', image: 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=200&h=200&fit=crop&crop=center', tags: ['swollen joints', 'inflammation', 'joints'] },
        { id: 'leg-pain', name: 'Leg Pain', image: 'https://images.unsplash.com/photo-1581578731547-07e7177d9f9f?w=200&h=200&fit=crop&crop=center', tags: ['leg pain', 'limbs', 'muscle'] },
        { id: 'arm-pain', name: 'Arm Pain', image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200&h=200&fit=crop&crop=center', tags: ['arm pain', 'limbs', 'muscle'] }
      ]
    },
    {
      id: 'general',
      name: 'General Symptoms',
      icon: '🌡️',
      images: [
        { id: 'fever', name: 'Fever', image: 'https://images.unsplash.com/photo-1582774007591-5ca6b4b8d2c9?w=200&h=200&fit=crop&crop=face', tags: ['fever', 'temperature', 'general'] },
        { id: 'fatigue', name: 'Fatigue', image: 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=200&h=200&fit=crop&crop=face', tags: ['fatigue', 'tired', 'general'] },
        { id: 'weight-loss', name: 'Weight Loss', image: 'https://images.unsplash.com/photo-1606890658317-7d14490b76fd?w=200&h=200&fit=crop&crop=face', tags: ['weight loss', 'general', 'appetite'] },
        { id: 'weight-gain', name: 'Weight Gain', image: 'https://images.unsplash.com/photo-1582213733776-fa1923c5c528?w=200&h=200&fit=crop&crop=face', tags: ['weight gain', 'general', 'appetite'] },
        { id: 'sweating', name: 'Excessive Sweating', image: 'https://images.unsplash.com/photo-1576092768241-dec231879af5?w=200&h=200&fit=crop&crop=face', tags: ['sweating', 'perspiration', 'general'] },
        { id: 'chills', name: 'Chills', image: 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=face', tags: ['chills', 'cold', 'general'] },
        { id: 'night-sweats', name: 'Night Sweats', image: 'https://images.unsplash.com/photo-1581578731547-07e7177d9f9f?w=200&h=200&fit=crop&crop=face', tags: ['night sweats', 'sleep', 'general'] },
        { id: 'insomnia', name: 'Insomnia', image: 'https://images.unsplash.com/photo-1517486808906-6ca8b3f04846?w=200&h=200&fit=crop&crop=face', tags: ['insomnia', 'sleep', 'general'] }
      ]
    },
    {
      id: 'mental',
      name: 'Mental Health',
      icon: '🧠',
      images: [
        { id: 'anxiety', name: 'Anxiety', image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200&h=200&fit=crop&crop=face', tags: ['anxiety', 'worry', 'mental'] },
        { id: 'depression', name: 'Depression', image: 'https://images.unsplash.com/photo-1523539664694-3d7e2f4a0b0e?w=200&h=200&fit=crop&crop=face', tags: ['depression', 'sadness', 'mental'] },
        { id: 'mood-swings', name: 'Mood Swings', image: 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=face', tags: ['mood swings', 'emotions', 'mental'] },
        { id: 'memory-loss', name: 'Memory Loss', image: 'https://images.unsplash.com/photo-1576092768241-dec231879af5?w=200&h=200&fit=crop&crop=face', tags: ['memory loss', 'forgetfulness', 'mental'] },
        { id: 'confusion', name: 'Confusion', image: 'https://images.unsplash.com/photo-1582213733776-fa1923c5c528?w=200&h=200&fit=crop&crop=face', tags: ['confusion', 'disorientation', 'mental'] },
        { id: 'irritability', name: 'Irritability', image: 'https://images.unsplash.com/photo-1606890658317-7d14490b76fd?w=200&h=200&fit=crop&crop=face', tags: ['irritability', 'anger', 'mental'] }
      ]
    },
    {
      id: 'urinary',
      name: 'Urinary System',
      icon: '🚽',
      images: [
        { id: 'frequent-urination', name: 'Frequent Urination', image: 'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=200&h=200&fit=crop&crop=center', tags: ['frequent urination', 'bladder', 'urinary'] },
        { id: 'painful-urination', name: 'Painful Urination', image: 'https://images.unsplash.com/photo-1581578731547-07e7177d9f9f?w=200&h=200&fit=crop&crop=center', tags: ['painful urination', 'burning', 'urinary'] },
        { id: 'blood-urine', name: 'Blood in Urine', image: 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=center', tags: ['blood in urine', 'hematuria', 'urinary'] },
        { id: 'incontinence', name: 'Incontinence', image: 'https://images.unsplash.com/photo-1576092768241-dec231879af5?w=200&h=200&fit=crop&crop=center', tags: ['incontinence', 'bladder control', 'urinary'] }
      ]
    }
  ];

  const handleCategorySelect = (categoryId: string) => {
    setSelectedCategory(categoryId);
  };

  const handleSymptomToggle = (symptom: SymptomImage) => {
    setSelectedSymptoms(prev => {
      const isSelected = prev.find(s => s.id === symptom.id);
      if (isSelected) {
        return prev.filter(s => s.id !== symptom.id);
      } else {
        return [...prev, symptom];
      }
    });
  };

  const handleSendSymptoms = () => {
    console.log('Sending symptoms and closing selector...');
    onSelectSymptoms(selectedSymptoms);
    // Don't reset selectedSymptoms - keep them for the chat session
    setSelectedCategory(null);
    onClose();
  };

  const handleBackToCategories = () => {
    setSelectedCategory(null);
  };

  const handleUnselectAll = () => {
    setSelectedSymptoms([]);
    setShowDropdown(false);
  };

  const handleClose = () => {
    // Reset symptoms when closing without sending
    setSelectedSymptoms([]);
    setSelectedCategory(null);
    setShowDropdown(false);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className={`w-full max-w-3xl h-[500px] rounded-2xl shadow-2xl transition-colors duration-300 flex flex-col ${
        theme === 'dark' ? 'bg-slate-800' : 'bg-white'
      }`}>
        {/* Header */}
        <div className={`flex items-center justify-between p-6 border-b flex-shrink-0 ${
          theme === 'dark' ? 'border-slate-600' : 'border-gray-200'
        }`}>
          <div className="flex items-center space-x-3">
            {selectedCategory && (
              <button
                onClick={handleBackToCategories}
                className={`p-2 rounded-lg transition-colors ${
                  theme === 'dark' 
                    ? 'text-white hover:bg-slate-700' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
            )}
            <h2 className={`text-xl font-bold ${
              theme === 'dark' ? 'text-white' : 'text-gray-800'
            }`}>
              {selectedCategory ? categories.find(c => c.id === selectedCategory)?.name : 'Select Symptoms'}
            </h2>
          </div>
          <button
            onClick={handleClose}
            className={`p-2 rounded-lg transition-colors ${
              theme === 'dark' 
                ? 'text-white hover:bg-slate-700' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 p-6 overflow-y-auto">
          {!selectedCategory ? (
            // Categories View
            <div className="grid grid-cols-3 gap-4">
              {categories.map((category) => (
                <button
                  key={category.id}
                  onClick={() => handleCategorySelect(category.id)}
                  className={`p-6 rounded-xl border-2 transition-all hover:scale-105 ${
                    theme === 'dark'
                      ? 'bg-slate-600 border-slate-500 hover:border-blue-400 hover:bg-slate-500'
                      : 'bg-gray-50 border-gray-200 hover:border-blue-500 hover:bg-blue-50'
                  }`}
                >
                  <div className="text-4xl mb-3">{category.icon}</div>
                  <h3 className={`font-semibold ${
                    theme === 'dark' ? 'text-white' : 'text-gray-800'
                  }`}>
                    {category.name}
                  </h3>
                  <p className={`text-sm mt-1 ${
                    theme === 'dark' ? 'text-slate-400' : 'text-gray-500'
                  }`}>
                    {category.images.length} symptoms
                  </p>
                </button>
              ))}
            </div>
          ) : (
            // Symptoms View
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {categories.find(c => c.id === selectedCategory)?.images.map((symptom) => {
                const isSelected = selectedSymptoms.find(s => s.id === symptom.id);
                return (
                  <button
                    key={symptom.id}
                    onClick={() => handleSymptomToggle(symptom)}
                    className={`relative p-4 rounded-xl border-2 transition-all hover:scale-105 ${
                      isSelected
                        ? theme === 'dark'
                          ? 'border-blue-500 bg-blue-900/30'
                          : 'border-blue-500 bg-blue-50'
                        : theme === 'dark'
                          ? 'bg-slate-600 border-slate-500 hover:border-blue-400 hover:bg-slate-500'
                          : 'bg-gray-50 border-gray-200 hover:border-blue-500 hover:bg-blue-50'
                    }`}
                  >
                    {isSelected && (
                      <div className="absolute top-2 right-2 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                        <Check className="w-4 h-4 text-white" />
                      </div>
                    )}
                     <div className="w-full h-24 bg-gray-200 rounded-lg mb-3 overflow-hidden">
                       <img 
                         src={symptom.image} 
                         alt={symptom.name}
                         className="w-full h-full object-cover"
                         onError={(e) => {
                           const target = e.target as HTMLImageElement;
                           target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkltYWdlPC90ZXh0Pjwvc3ZnPg==';
                         }}
                       />
                     </div>
                    <h4 className={`font-medium text-sm ${
                      isSelected
                        ? theme === 'dark'
                          ? 'text-blue-100'
                          : 'text-blue-800'
                        : theme === 'dark'
                          ? 'text-white'
                          : 'text-gray-800'
                    }`}>
                      {symptom.name}
                    </h4>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        {selectedSymptoms.length > 0 && (
          <div className={`p-6 border-t flex-shrink-0 ${
            theme === 'dark' ? 'border-slate-600 bg-slate-700' : 'border-gray-200 bg-gray-50'
          }`}>
            <div className="flex items-center justify-between">
              <div className={`text-sm ${
                theme === 'dark' ? 'text-slate-300' : 'text-gray-600'
              }`}>
                {selectedSymptoms.length} symptom{selectedSymptoms.length !== 1 ? 's' : ''} selected
              </div>
              <div className="flex items-center space-x-3">
                {/* 3-dots menu */}
                <div className="relative" ref={dropdownRef}>
                  <button
                    onClick={() => setShowDropdown(!showDropdown)}
                    className={`p-2 rounded-lg transition-colors ${
                      theme === 'dark' 
                        ? 'text-white hover:bg-slate-600' 
                        : 'text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    <MoreVertical className="w-5 h-5" />
                  </button>
                  
                  {showDropdown && (
                    <div className={`absolute bottom-full right-0 mb-2 w-48 rounded-lg shadow-lg border ${
                      theme === 'dark' 
                        ? 'bg-slate-700 border-slate-600' 
                        : 'bg-white border-gray-200'
                    }`}>
                      <button
                        onClick={handleUnselectAll}
                        className={`w-full px-4 py-3 text-left text-sm rounded-lg transition-colors ${
                          theme === 'dark' 
                            ? 'text-white hover:bg-slate-600' 
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        Unselect All
                      </button>
                    </div>
                  )}
                </div>
                
                <button
                  onClick={handleSendSymptoms}
                  disabled={isLoading}
                  className={`px-6 py-2 rounded-lg transition-colors flex items-center space-x-2 ${
                    isLoading
                      ? 'bg-blue-400 text-white cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {isLoading && (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  )}
                  <span>{isLoading ? 'Sending...' : 'Send Symptoms'}</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SymptomSelector;