import React, { useState, useEffect, useRef } from 'react';
import { X, ArrowLeft, Check, MoreVertical } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';
import { useLanguage } from '../../contexts/LanguageContext';

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
  const { t } = useLanguage();
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  // Helper function to translate symptom names
  const translateSymptomName = (symptomId: string): string => {
    const translationKey = `symptoms.${symptomId}`;
    return t(translationKey);
  };

  // Helper function to create translated symptom image
  const createSymptomImage = (id: string, originalName: string, image: string, tags: string[]): SymptomImage => ({
    id,
    name: translateSymptomName(id),
    image,
    tags
  });
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
      name: t('symptomCategories.head'),
      icon: '🧠',
      images: [
        createSymptomImage('headache', 'Headache', 'https://regionalneurological.com/wp-content/uploads/2019/08/AdobeStock_244803452.jpeg', ['headache', 'pain', 'head']),
        createSymptomImage('migraine', 'Migraine', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTDsI3EyF0tFEpWQeVSRMf1pZzLuWpAze_2N4nwGlAI8S2_2UI4C-QzF1QFdVozwvRhvA8&usqp=CAU', ['migraine', 'severe headache', 'head']),
        createSymptomImage('dizziness', 'Dizziness', 'https://alldaymedicalcare.com/wp-content/uploads/2024/09/Dizziness.jpg', ['dizziness', 'vertigo', 'head']),
        createSymptomImage('facialPain', 'Facial Pain', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcROGdTjdy4_h3MKaCEL_3tO_ki56tGRmZUFlx6ZQtUWVAEspjWwsUc0o2rcMTZ66zYL0Pk&usqp=CAU', ['facial pain', 'face', 'head']),
        createSymptomImage('eyePain', 'Eye Pain', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeb_pqkYFBv94iTBq2lb3oQM1pB8M5jwUYWQHyRRfyUxRxAzCeiJezglAFkHXSh_i98AA&usqp=CAU', ['eye pain', 'vision', 'head']),
        createSymptomImage('earPain', 'Ear Pain', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcrM-9RAzvRbRnn9bRZ2rTDThKGxJ1BjqQ0OoWFTnkPTt4K6oTALoV2qif4W65W8s_Dx4&usqp=CAU', ['ear pain', 'hearing', 'head']),
        createSymptomImage('toothPain', 'Tooth Pain', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-3ZAeveaYQ8jEXcumMj_4mP-j3Oa6Oa6Lho_JeHIEcccrTTcOQVy8wPlF5QIXNt1zj6M&usqp=CAU', ['tooth pain', 'dental', 'mouth']),
        createSymptomImage('neckPain', 'Neck Pain', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSYtdG_EmVul6NJOakkw-hsZ9br3Uegwr3z2PkgrZRAvXnSwkE5Zfjn7d1E-cc03aXV1pg&usqp=CAU', ['neck pain', 'stiffness', 'head'])
      ]
    },
    {
      id: 'chest',
      name: t('symptomCategories.chest'),
      icon: '❤️',
      images: [
        createSymptomImage('chestPain', 'Chest Pain', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSaQ3NRUOGbZREb9bawPnJJx7_qmCcp-P-Ntz-etfUOeh6eprGoMCRwSvnL24H8ti8869M&usqp=CAU', ['chest pain', 'heart', 'chest']),
        createSymptomImage('shortnessOfBreath', 'Shortness of Breath', 'https://www.primehv.com/wp-content/uploads/2024/02/shortness-of-breath.jpeg', ['breathing', 'lungs', 'chest']),
        createSymptomImage('heartPalpitations', 'Heart Palpitations', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTA8BZ1Hhl7i9CF0aw3WAeyznWKlSTuqCA7IwB5JlSvkbXzOtNpUGODg5G2HLoPZwCvnr8&usqp=CAU', ['heart', 'palpitations', 'chest']),
        createSymptomImage('coughing', 'Cough', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQOUED43vh_bC1tB0SxPEop1OOIDETjb0hOZqm-fs2Aelhn_35WfMxskKVg9xL0Uh7rHv4&usqp=CAU', ['cough', 'throat', 'chest']),
        createSymptomImage('wheezing', 'Wheezing', 'https://www.lungandsleep.com.au/wp-content/uploads/2020/09/Wheeze-1.jpg', ['wheezing', 'breathing', 'lungs']),
        createSymptomImage('chestTightness', 'Chest Tightness', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSSfG675FvnSRh_uGBdGwvvSsMt6DwMikx6dGJiXaVbTBtbcwGjFOrWOQBKQfBS4R5-q-A&usqp=CAU', ['chest tightness', 'pressure', 'chest'])
      ]
    },
    {
      id: 'abdomen',
      name: t('symptomCategories.abdomen'),
      icon: '🫀',
      images: [
        createSymptomImage('stomachPain', 'Stomach Pain', 'https://www.emergencyphysicians.org/siteassets/emphysicians/all-images/kwtg/stomach-ache3.jpg', ['stomach pain', 'abdomen', 'digestive']),
        createSymptomImage('nausea', 'Nausea', 'https://www.visitcompletecare.com/wp-content/uploads/2025/05/shutterstock_1972998752-1.webp', ['nausea', 'sick', 'stomach']),
        createSymptomImage('diarrhea', 'Diarrhea', 'https://gastrofl.com/wp-content/uploads/2023/03/Gastro-image-1246295110.jpeg', ['diarrhea', 'digestive', 'stomach']),
        createSymptomImage('constipation', 'Constipation', 'https://www.newlifenutrition.com.au/wp-content/uploads/AdobeStock_207132330-1024x540.jpeg', ['constipation', 'digestive', 'stomach']),
        createSymptomImage('vomiting', 'Vomiting', 'https://drupal-cdn-hfaeddcdbng5hfbg.a01.azurefd.net/sites/default/files/2025-02/Nausea-and-Vomiting-scaled.jpg', ['vomiting', 'nausea', 'stomach']),
        createSymptomImage('bloating', 'Bloating', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzl95E8HZBjfUrqylQbJuAn7PK3fBZPC3l2w&s', ['bloating', 'gas', 'stomach']),
        createSymptomImage('heartburn', 'Heartburn', 'https://cdhf.ca/wp-content/uploads/2022/07/heartburn-causes-treatment-scaled.jpg', ['heartburn', 'acid reflux', 'stomach']),
        createSymptomImage('appetiteLoss', 'Loss of Appetite', 'https://www.sugarfit.com/assets/638dde01d46fe3ff88b82cf7_loss-of-appetite_Z1REA2s.jpg', ['appetite loss', 'hunger', 'stomach'])
      ]
    },
    {
      id: 'skin',
      name: t('symptomCategories.skin'),
      icon: '🦠',
      images: [
        { id: 'rash', name: 'Skin Rash', image: 'https://images.theconversation.com/files/209558/original/file-20180308-30983-e4u830.jpg?ixlib=rb-1.1.0&q=45&auto=format&w=754&fit=clip', tags: ['rash', 'skin', 'irritation'] },
        { id: 'hives', name: 'Hives', image: 'https://images.everydayhealth.com/images/2025/what-hives-look-like-alt-1440x810.jpg?sfvrsn=c9d65b48_3', tags: ['hives', 'allergic reaction', 'skin'] },
        { id: 'acne', name: 'Acne', image: 'https://southern-dermatology.com.au/cdn/shop/files/Acne_Vulgaris.png?v=1739831943&width=1080', tags: ['acne', 'skin', 'face'] },
        { id: 'dry-skin', name: 'Dry Skin', image: 'https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2F3mll21jv39ya1.jpg', tags: ['dry skin', 'flaky', 'skin'] },
        { id: 'itching', name: 'Itching', image: 'https://gladskin.com/cdn/shop/articles/TEMPLATE-Blog_Header_6_3b77c7f6-c2de-429c-9fb4-8589087afa07.jpg?v=1677712889', tags: ['itching', 'pruritus', 'skin'] },
        { id: 'swelling', name: 'Swelling', image: 'https://media.post.rvohealth.io/wp-content/uploads/sites/3/2020/05/swollen-toes.-angioedema-732x549-thumbnail.jpg', tags: ['swelling', 'edema', 'skin'] },
        { id: 'bruising', name: 'Bruising', image: 'https://www.nebraskamed.com/sites/default/files/images/dermatology/bruises_opengraph.jpg', tags: ['bruising', 'contusion', 'skin'] },
        { id: 'discoloration', name: 'Skin Discoloration', image: 'https://deyga.in/cdn/shop/articles/skin-discoloration.jpg?v=1652349499&width=1100', tags: ['discoloration', 'pigmentation', 'skin'] }
      ]
    },
    {
      id: 'muscle',
      name: t('symptomCategories.muscle'),
      icon: '🦴',
      images: [
        createSymptomImage('backPain', 'Back Pain', 'https://images.unsplash.com/photo-1523539664694-3d7e2f4a0b0e?w=200&h=200&fit=crop&crop=center', ['back pain', 'muscle', 'spine']),
        createSymptomImage('jointPain', 'Joint Pain', 'https://images.unsplash.com/photo-1603344205187-78a91b8a9533?w=200&h=200&fit=crop&crop=center', ['joint pain', 'arthritis', 'muscle']),
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
      name: t('symptomCategories.general'),
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
      name: t('symptomCategories.mental'),
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
      name: t('symptomCategories.urinary'),
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
        theme === 'noongar-dark' 
          ? 'bg-slate-800' 
          : theme === 'noongar-light'
          ? 'bg-white'
          : theme === 'dark' 
          ? 'bg-slate-800' 
          : 'bg-white'
      }`}>
        {/* Header */}
        <div className={`flex items-center justify-between p-6 border-b flex-shrink-0 ${
          theme === 'noongar-dark' 
            ? 'border-slate-600' 
            : theme === 'noongar-light'
            ? 'border-orange-200'
            : theme === 'dark' 
            ? 'border-slate-600' 
            : 'border-gray-200'
        }`}>
          <div className="flex items-center space-x-3">
            {selectedCategory && (
              <button
                onClick={handleBackToCategories}
                className={`p-2 rounded-lg transition-colors ${
                  (theme === 'noongar-dark' || theme === 'noongar-light')
                    ? theme === 'noongar-dark'
                      ? 'text-white hover:bg-slate-700'
                      : 'text-orange-600 hover:bg-orange-100'
                    : theme === 'dark' 
                    ? 'text-white hover:bg-slate-700' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
            )}
            <h2 className={`text-xl font-bold ${
              (theme === 'noongar-dark' || theme === 'noongar-light')
                ? theme === 'noongar-dark'
                  ? 'text-white'
                  : 'text-orange-800'
                : theme === 'dark' 
                ? 'text-white' 
                : 'text-gray-800'
            }`}>
              {selectedCategory ? categories.find(c => c.id === selectedCategory)?.name : 'Select Symptoms'}
            </h2>
          </div>
          <button
            onClick={handleClose}
            className={`p-2 rounded-lg transition-colors ${
              (theme === 'noongar-dark' || theme === 'noongar-light')
                ? theme === 'noongar-dark'
                  ? 'text-white hover:bg-slate-700'
                  : 'text-orange-600 hover:bg-orange-100'
                : theme === 'dark' 
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
                    (theme === 'noongar-dark' || theme === 'noongar-light')
                      ? theme === 'noongar-dark'
                        ? 'bg-slate-600 border-slate-500 hover:border-orange-400 hover:bg-slate-500'
                        : 'bg-orange-50 border-orange-200 hover:border-orange-500 hover:bg-orange-100'
                      : theme === 'dark'
                      ? 'bg-slate-600 border-slate-500 hover:border-blue-400 hover:bg-slate-500'
                      : 'bg-gray-50 border-gray-200 hover:border-blue-500 hover:bg-blue-50'
                  }`}
                >
                  <div className="text-4xl mb-3">{category.icon}</div>
                  <h3 className={`font-semibold ${
                    (theme === 'noongar-dark' || theme === 'noongar-light')
                      ? theme === 'noongar-dark'
                        ? 'text-white'
                        : 'text-orange-800'
                      : theme === 'dark' 
                      ? 'text-white' 
                      : 'text-gray-800'
                  }`}>
                    {category.name}
                  </h3>
                  <p className={`text-sm mt-1 ${
                    (theme === 'noongar-dark' || theme === 'noongar-light')
                      ? theme === 'noongar-dark'
                        ? 'text-slate-400'
                        : 'text-orange-600'
                      : theme === 'dark' 
                      ? 'text-slate-400' 
                      : 'text-gray-500'
                  }`}>
                    {category.images.length} {t('symptomCount')}
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
                        ? (theme === 'noongar-dark' || theme === 'noongar-light')
                          ? theme === 'noongar-dark'
                            ? 'border-orange-500 bg-orange-900/30'
                            : 'border-orange-500 bg-orange-50'
                          : theme === 'dark'
                          ? 'border-blue-500 bg-blue-900/30'
                          : 'border-blue-500 bg-blue-50'
                        : (theme === 'noongar-dark' || theme === 'noongar-light')
                          ? theme === 'noongar-dark'
                            ? 'bg-slate-600 border-slate-500 hover:border-orange-400 hover:bg-slate-500'
                            : 'bg-orange-50 border-orange-200 hover:border-orange-500 hover:bg-orange-100'
                          : theme === 'dark'
                          ? 'bg-slate-600 border-slate-500 hover:border-blue-400 hover:bg-slate-500'
                          : 'bg-gray-50 border-gray-200 hover:border-blue-500 hover:bg-blue-50'
                    }`}
                  >
                    {isSelected && (
                      <div className={`absolute top-2 right-2 w-6 h-6 rounded-full flex items-center justify-center ${
                        (theme === 'noongar-dark' || theme === 'noongar-light')
                          ? 'bg-orange-500'
                          : 'bg-blue-500'
                      }`}>
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
                        ? (theme === 'noongar-dark' || theme === 'noongar-light')
                          ? theme === 'noongar-dark'
                            ? 'text-orange-100'
                            : 'text-orange-800'
                          : theme === 'dark'
                          ? 'text-blue-100'
                          : 'text-blue-800'
                        : (theme === 'noongar-dark' || theme === 'noongar-light')
                          ? theme === 'noongar-dark'
                            ? 'text-white'
                            : 'text-orange-800'
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
            theme === 'noongar-dark' 
              ? 'border-slate-600 bg-slate-700' 
              : theme === 'noongar-light'
              ? 'border-orange-200 bg-orange-50'
              : theme === 'dark' 
              ? 'border-slate-600 bg-slate-700' 
              : 'border-gray-200 bg-gray-50'
          }`}>
            <div className="flex items-center justify-between">
              <div className={`text-sm ${
                (theme === 'noongar-dark' || theme === 'noongar-light')
                  ? theme === 'noongar-dark'
                    ? 'text-slate-300'
                    : 'text-orange-600'
                  : theme === 'dark' 
                  ? 'text-slate-300' 
                  : 'text-gray-600'
              }`}>
                {selectedSymptoms.length} symptom{selectedSymptoms.length !== 1 ? 's' : ''} selected
              </div>
              <div className="flex items-center space-x-3">
                {/* 3-dots menu */}
                <div className="relative" ref={dropdownRef}>
                  <button
                    onClick={() => setShowDropdown(!showDropdown)}
                    className={`p-2 rounded-lg transition-colors ${
                      (theme === 'noongar-dark' || theme === 'noongar-light')
                        ? theme === 'noongar-dark'
                          ? 'text-white hover:bg-slate-600'
                          : 'text-orange-600 hover:bg-orange-200'
                        : theme === 'dark' 
                        ? 'text-white hover:bg-slate-600' 
                        : 'text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    <MoreVertical className="w-5 h-5" />
                  </button>
                  
                  {showDropdown && (
                    <div className={`absolute bottom-full right-0 mb-2 w-48 rounded-lg shadow-lg border ${
                      (theme === 'noongar-dark' || theme === 'noongar-light')
                        ? theme === 'noongar-dark'
                          ? 'bg-slate-700 border-slate-600'
                          : 'bg-white border-orange-200'
                        : theme === 'dark' 
                        ? 'bg-slate-700 border-slate-600' 
                        : 'bg-white border-gray-200'
                    }`}>
                      <button
                        onClick={handleUnselectAll}
                        className={`w-full px-4 py-3 text-left text-sm rounded-lg transition-colors ${
                          (theme === 'noongar-dark' || theme === 'noongar-light')
                            ? theme === 'noongar-dark'
                              ? 'text-white hover:bg-slate-600'
                              : 'text-orange-700 hover:bg-orange-100'
                            : theme === 'dark' 
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
                      ? (theme === 'noongar-dark' || theme === 'noongar-light')
                        ? 'bg-orange-400 text-white cursor-not-allowed'
                        : 'bg-blue-400 text-white cursor-not-allowed'
                      : (theme === 'noongar-dark' || theme === 'noongar-light')
                        ? 'bg-orange-600 text-white hover:bg-orange-700'
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