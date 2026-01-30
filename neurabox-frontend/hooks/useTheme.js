/**
 * useTheme Hook
 * Manages theme state and provides theme utilities
 * Persists theme preference to localStorage
 */

'use client';

import { useState, useEffect, useContext, createContext } from 'react';
import { getTheme } from '@/lib/theme';

// Create context
export const ThemeContext = createContext(null);

/**
 * Get initial theme from localStorage or system preference
 */
function getInitialTheme() {
  // Check localStorage first
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('theme');
    if (stored) return stored;

    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
  }

  return 'dark'; // Default to dark
}

/**
 * Hook to use theme context
 * Safe to use in client components - provides fallback if provider not found
 */
export function useTheme() {
  const context = useContext(ThemeContext);
  
  if (!context) {
    // Fallback for components used outside provider
    // This prevents errors but use inside ThemeProvider for full functionality
    console.warn('useTheme called outside ThemeProvider - using default dark theme');
    return {
      theme: 'dark',
      themeConfig: getTheme('dark'),
      toggleTheme: () => {},
      setThemeMode: () => {},
      isDark: true,
      isLight: false,
    };
  }
  
  return context;
}

/**
 * Theme Provider Component
 * Wraps app and provides theme context
 */
export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('dark');
  const [mounted, setMounted] = useState(false);

  // Initialize theme on mount (client-side only)
  useEffect(() => {
    const initialTheme = getInitialTheme();
    setTheme(initialTheme);
    document.documentElement.setAttribute('data-theme', initialTheme);
    setMounted(true);
  }, []);

  // Toggle theme
  const toggleTheme = () => {
    setTheme(prevTheme => {
      const newTheme = prevTheme === 'dark' ? 'light' : 'dark';

      // Persist to localStorage
      localStorage.setItem('theme', newTheme);

      // Update DOM
      document.documentElement.setAttribute('data-theme', newTheme);

      return newTheme;
    });
  };

  // Set specific theme
  const setThemeMode = (newTheme) => {
    if (['light', 'dark'].includes(newTheme)) {
      setTheme(newTheme);
      localStorage.setItem('theme', newTheme);
      document.documentElement.setAttribute('data-theme', newTheme);
    }
  };

  // Prevent hydration mismatch - render nothing on server, full content on client
  if (!mounted) {
    return <>{children}</>;
  }

  const themeConfig = getTheme(theme);

  const value = {
    theme,
    themeConfig,
    toggleTheme,
    setThemeMode,
    isDark: theme === 'dark',
    isLight: theme === 'light',
  };

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}
