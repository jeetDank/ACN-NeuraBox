/**
 * Theme Style Utilities
 * Functions to generate component styles from theme configuration
 * Eliminates hardcoded color values throughout components
 */

import { SPACING, BORDER_RADIUS, TYPOGRAPHY, TRANSITIONS } from '@/lib/theme';

/**
 * Generate container styles with theme
 */
export function getContainerStyles(theme) {
  return {
    background: theme.bg.primary,
    color: theme.text.primary,
    fontFamily: TYPOGRAPHY.fontFamily,
  };
}

/**
 * Generate card styles with theme
 */
export function getCardStyles(theme) {
  return {
    background: theme.card.bg,
    color: theme.card.text,
    border: `1px solid ${theme.card.border}`,
    borderRadius: BORDER_RADIUS.lg,
    boxShadow: theme.card.shadow,
  };
}

/**
 * Generate chat message styles based on sender
 */
export function getChatBubbleStyles(theme, isUser = false) {
  const bubble = isUser ? theme.chat.user : theme.chat.ai;

  return {
    background: bubble.bg,
    color: bubble.text,
    borderRadius: BORDER_RADIUS.lg,
    padding: `${SPACING.md} ${SPACING.lg}`,
  };
}

/**
 * Generate input field styles
 */
export function getInputStyles(theme) {
  return {
    background: theme.input.bg,
    color: theme.input.text,
    border: `1px solid ${theme.input.border}`,
    borderRadius: BORDER_RADIUS.md,
    padding: `${SPACING.sm} ${SPACING.md}`,
    '::placeholder': {
      color: theme.input.placeholder,
    },
    ':focus': {
      borderColor: theme.input.focus,
      outline: 'none',
    },
  };
}

/**
 * Generate button styles
 */
export function getButtonStyles(theme, variant = 'primary') {
  const buttonConfig = theme.button[variant] || theme.button.primary;

  return {
    background: buttonConfig.bg,
    color: buttonConfig.text,
    border: buttonConfig.border ? `1px solid ${buttonConfig.border}` : 'none',
    borderRadius: BORDER_RADIUS.md,
    padding: `${SPACING.sm} ${SPACING.md}`,
    cursor: 'pointer',
    transition: `all ${TRANSITIONS.fast}`,
    ':hover': {
      background: buttonConfig.hover,
      opacity: 0.9,
    },
  };
}

/**
 * Generate tab styles
 */
export function getTabStyles(theme, isActive = false) {
  return isActive
    ? {
        background: theme.tab.activeBg,
        color: theme.tab.activeText,
      }
    : {
        background: theme.tab.bg,
        color: theme.tab.text,
      };
}

/**
 * Generate border styles
 */
export function getBorderStyles(theme, strength = 'light') {
  return {
    borderColor: theme.border[strength],
  };
}

/**
 * Generate text styles
 */
export function getTextStyles(theme, variant = 'primary') {
  const textColor = theme.text[variant] || theme.text.primary;

  return {
    color: textColor,
  };
}

/**
 * Generate modal/overlay styles
 */
export function getModalStyles(theme) {
  return {
    background: theme.bg.primary,
    boxShadow: theme.modal.shadow,
    borderRadius: BORDER_RADIUS.xl,
  };
}

/**
 * Create CSS class strings from theme for CSS modules
 * Useful for creating consistent color utility classes
 */
export function createThemeClasses(theme) {
  return {
    bgPrimary: 'background: ' + theme.bg.primary,
    bgSecondary: 'background: ' + theme.bg.secondary,
    bgTertiary: 'background: ' + theme.bg.tertiary,
    textPrimary: 'color: ' + theme.text.primary,
    textSecondary: 'color: ' + theme.text.secondary,
    textMuted: 'color: ' + theme.text.muted,
    borderLight: 'border-color: ' + theme.border.light,
    borderMedium: 'border-color: ' + theme.border.medium,
    accentPurple: 'color: ' + theme.accent.purple,
    accentTeal: 'color: ' + theme.accent.teal,
  };
}

/**
 * Generate gradient styles using theme accents
 */
export function getGradientStyles(theme) {
  return {
    purpleGradient: `linear-gradient(135deg, ${theme.accent.teal} 0%, ${theme.accent.purple} 50%, #c74b7a 100%)`,
    purpleToTeal: `linear-gradient(to right, ${theme.accent.purple}, ${theme.accent.teal})`,
  };
}

/**
 * Generate focus styles for accessibility
 */
export function getFocusStyles(theme) {
  return {
    outline: `2px solid ${theme.accent.purple}`,
    outlineOffset: '2px',
  };
}

/**
 * Generate hover and active states for interactive elements
 */
export function getInteractiveStyles(theme) {
  return {
    hover: theme.bg.hover,
    active: theme.bg.active,
    hoverTransition: `background ${TRANSITIONS.fast}`,
  };
}
