import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useTheme } from '../../contexts/ThemeContext';

interface NoongarSlideshowProps {
  className?: string;
}

const NoongarSlideshow: React.FC<NoongarSlideshowProps> = ({ className = '' }) => {
  const [currentSlide, setCurrentSlide] = useState(0);
  const { theme } = useTheme();

  // Authentic Noongar cultural and medical images
  const slides = [
    {
      id: 1,
      image: '/src/images/noongar images/denmark-dooram-dancersweb-006.jpg',
      title: 'Boodja - Ngalak Kaaditj Kwobidak',
      description: 'Ngalak boodja koorliny wer mooditj djoorap nyininy',
      alt: 'Noongar dancers representing connection to country'
    },
    {
      id: 2,
      image: '/src/images/noongar images/denmark-doorum-dancers-web-033.jpg',
      title: 'Koorliny - Ngalak Mooditj Bidi',
      description: 'Ngalak koorliny djena koorliny wer mooditj djoorap nyininy',
      alt: 'Traditional Noongar cultural practices'
    },
    {
      id: 3,
      image: '/src/images/noongar images/ssd_coc_world_arts_exchange_full_res-0641_0.jpg',
      title: 'Moort - Ngalak Moorditj',
      description: 'Ngalak moort yokiny wer mooditj djoorap nyininy',
      alt: 'Noongar community and family connections'
    },
    {
      id: 4,
      image: '/src/images/noongar images/ssd_coc_world_arts_exchange_full_res-069.jpg',
      title: 'Kaartdijin - Ngalak Yarnang',
      description: 'Ngalak kaartdijin djena koorliny wer mooditj djoorap nyininy',
      alt: 'Noongar cultural knowledge and wisdom'
    },
    {
      id: 5,
      image: '/src/images/noongar images/yagan-reburial_033_2010.jpg',
      title: 'Wangkiny - Ngalak Moorditj',
      description: 'Ngalak wangkiny djena koorliny wer mooditj djoorap nyininy',
      alt: 'Noongar cultural ceremony and respect'
    }
  ];

  // Auto-advance slides
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % slides.length);
    }, 5000); // Change slide every 5 seconds

    return () => clearInterval(timer);
  }, [slides.length]);

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const goToSlide = (index: number) => {
    setCurrentSlide(index);
  };

  // This component is only rendered when theme is noongar-light or noongar-dark

  return (
    <div className={`relative w-full h-80 md:h-96 rounded-2xl overflow-hidden shadow-2xl border-2 ${
      theme === 'noongar-light' 
        ? 'border-orange-500/30' 
        : 'border-amber-600/30'
    } ${className}`}>
      {/* Slides */}
      <div className="relative w-full h-full">
        {slides.map((slide, index) => (
          <div
            key={slide.id}
            className={`absolute inset-0 transition-opacity duration-700 ${
              index === currentSlide ? 'opacity-100' : 'opacity-0'
            }`}
          >
            <img
              src={slide.image}
              alt={slide.alt}
              className="w-full h-full object-cover"
            />
            {/* Enhanced Overlay with Noongar colors */}
            <div className={`absolute inset-0 ${
              theme === 'noongar-light'
                ? 'bg-gradient-to-t from-orange-800/80 via-red-700/30 to-transparent'
                : 'bg-gradient-to-t from-amber-900/80 via-orange-900/30 to-transparent'
            }`} />
            
            {/* Content with better styling */}
            <div className="absolute bottom-0 left-0 right-0 p-6 text-white">
              <div className={`bg-black/40 backdrop-blur-sm rounded-lg p-3 border ${
                theme === 'noongar-light' 
                  ? 'border-orange-500/30' 
                  : 'border-amber-500/30'
              }`}>
                <h3 className={`text-lg md:text-xl font-bold mb-2 ${
                  theme === 'noongar-light' 
                    ? 'text-orange-200' 
                    : 'text-amber-200'
                }`}>{slide.title}</h3>
                <p className={`text-sm md:text-base leading-relaxed ${
                  theme === 'noongar-light' 
                    ? 'text-orange-100' 
                    : 'text-amber-100'
                }`}>{slide.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Navigation Arrows */}
      <button
        onClick={prevSlide}
        className={`absolute left-4 top-1/2 transform -translate-y-1/2 text-white p-3 rounded-full transition-all shadow-lg hover:shadow-xl ${
          theme === 'noongar-light'
            ? 'bg-orange-600/80 hover:bg-orange-500'
            : 'bg-amber-600/80 hover:bg-amber-500'
        }`}
        aria-label="Previous slide"
      >
        <ChevronLeft className="w-6 h-6" />
      </button>
      
      <button
        onClick={nextSlide}
        className={`absolute right-4 top-1/2 transform -translate-y-1/2 text-white p-3 rounded-full transition-all shadow-lg hover:shadow-xl ${
          theme === 'noongar-light'
            ? 'bg-orange-600/80 hover:bg-orange-500'
            : 'bg-amber-600/80 hover:bg-amber-500'
        }`}
        aria-label="Next slide"
      >
        <ChevronRight className="w-6 h-6" />
      </button>

      {/* Enhanced Dots Indicator */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
        {slides.map((_, index) => (
          <button
            key={index}
            onClick={() => goToSlide(index)}
            className={`w-3 h-3 rounded-full transition-all duration-300 ${
              index === currentSlide 
                ? theme === 'noongar-light'
                  ? 'bg-orange-400 shadow-lg scale-110'
                  : 'bg-amber-400 shadow-lg scale-110'
                : 'bg-white/60 hover:bg-white/80 hover:scale-105'
            }`}
            aria-label={`Go to slide ${index + 1}`}
          />
        ))}
      </div>
    </div>
  );
};

export default NoongarSlideshow;
