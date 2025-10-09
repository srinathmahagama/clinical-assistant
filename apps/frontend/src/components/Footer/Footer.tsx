import React from 'react';
import { Heart, Mail, Phone, MapPin, Shield, Users, Clock } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-[#183172] text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Company Info */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Heart className="w-6 h-6 text-white" />
              <h3 className="text-xl font-bold">CareMate</h3>
            </div>
            <p className="text-white/80 text-sm leading-relaxed">
              Your trusted health assessment companion. We help you understand your symptoms 
              and provide guidance for better health decisions.
            </p>
            <div className="flex space-x-4">
              <div className="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center">
                <Shield className="w-4 h-4" />
              </div>
              <div className="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center">
                <Users className="w-4 h-4" />
              </div>
              <div className="w-8 h-8 bg-white/10 rounded-full flex items-center justify-center">
                <Clock className="w-4 h-4" />
              </div>
            </div>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold">Quick Links</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="/" className="text-white/80 hover:text-white transition-colors">
                  Home
                </a>
              </li>
              <li>
                <a href="/voice-input" className="text-white/80 hover:text-white transition-colors">
                  Voice Assessment
                </a>
              </li>
              <li>
                <a href="/text-input" className="text-white/80 hover:text-white transition-colors">
                  Text Assessment
                </a>
              </li>
              <li>
                <a href="/assistant" className="text-white/80 hover:text-white transition-colors">
                  Health Assistant
                </a>
              </li>
              <li>
                <a href="/history" className="text-white/80 hover:text-white transition-colors">
                  Assessment History
                </a>
              </li>
            </ul>
          </div>

          {/* Support */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold">Support</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="text-white/80 hover:text-white transition-colors">
                  Help Center
                </a>
              </li>
              <li>
                <a href="#" className="text-white/80 hover:text-white transition-colors">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-white/80 hover:text-white transition-colors">
                  Terms of Service
                </a>
              </li>
              <li>
                <a href="#" className="text-white/80 hover:text-white transition-colors">
                  FAQ
                </a>
              </li>
              <li>
                <a href="#" className="text-white/80 hover:text-white transition-colors">
                  Contact Us
                </a>
              </li>
            </ul>
          </div>

          {/* Contact Info */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold">Contact Us</h4>
            <div className="space-y-3 text-sm">
              <div className="flex items-center space-x-3">
                <Mail className="w-4 h-4 text-white/60" />
                <span className="text-white/80">support@caremate.com</span>
              </div>
              <div className="flex items-center space-x-3">
                <Phone className="w-4 h-4 text-white/60" />
                <span className="text-white/80">(03) 9245 6754</span>
              </div>
              <div className="flex items-start space-x-3">
                <MapPin className="w-4 h-4 text-white/60 mt-0.5" />
                <span className="text-white/80">
                  465 Brunswick St<br />
                  Fitzroy North VIC 2069
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-white/20 mt-8 pt-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="text-sm text-white/60">
              © 2025 CareMate. All rights reserved.
            </div>
            <div className="text-sm text-white/60">
              <span className="mr-4">Made with ❤️ for better health</span>
              <span>Version 1.0.0</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
