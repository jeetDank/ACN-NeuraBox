'use client';

import { useTheme } from '@/hooks/useTheme';
import styles from './FloatingButton.module.css';
import Image from 'next/image';

/**
 * Floating NeuroAI button - collapsed state
 * Circular button with gradient border and brain icon
 * Theme-aware: gradients change based on light/dark mode
 */
export default function FloatingButton({ onClick }) {
    const { themeConfig } = useTheme();

    // Theme-aware gradient colors
    const getGradientColors = () => {
        return {
            color1: themeConfig.accent.teal,
            color2: themeConfig.accent.purple,
            color3: '#c74b7a',
        };
    };

    const gradientColors = getGradientColors();

    return (
        <button
            className={styles.floatingButton}
            onClick={onClick}
            aria-label="Open  ACN Link "
            style={{
                background: `linear-gradient(135deg, ${gradientColors.color1} 0%, ${gradientColors.color2} 50%, ${gradientColors.color3} 100%)`,
            }}
        >
            
                {/* NeuroAI Brain Icon SVG */}
                <img
                    src="/images/logo.gif"
                    alt="NeuraBox"
                    className={styles.iconWrapper}
                    
                />
        </button>
    );
}
