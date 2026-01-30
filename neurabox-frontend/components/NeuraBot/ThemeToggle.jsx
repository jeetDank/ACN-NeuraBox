'use client';

import { Sun, Moon } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import styles from './ThemeToggle.module.css';

/**
 * Professional Theme Toggle Component
 * Single button that toggles between light/dark themes
 * Uses centralized theme system for consistency
 * Theme preference is persisted in localStorage
 */
export default function ThemeToggle() {
    const { theme, toggleTheme, themeConfig } = useTheme();

    return (
        <button
            className={styles.toggleButton}
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            title={`Current: ${theme} theme`}
            style={{
                color: themeConfig.text.primary,
                background: themeConfig.bg.hover,
                border: `1px solid ${themeConfig.border.light}`,
            }}
        >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>
    );
}
