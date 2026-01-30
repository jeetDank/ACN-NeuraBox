/**
 * Theme Configuration System
 * Centralized design tokens for light and dark themes
 * All hardcoded colors are abstracted to this single source of truth
 */

export const THEME = {
  light: {
    // Primary backgrounds
    bg: {
      primary: '#E4E6EB',      // Whole side panel and above panel
      secondary: '#FFFFFF',    // Chat panel background
      tertiary: '#FFFFFF',     // Input/output background
      hover: 'rgba(0, 0, 0, 0.05)',
      active: 'rgba(0, 0, 0, 0.08)',
    },
    // Text colors
    text: {
      primary: '#353536',      // Black color as specified
      secondary: '#353536',
      muted: '#353536',
    },
    // Accent colors
    accent: {
      blue: '#0071CD',         // Title color (kept same)
      purple: '#A98DF8',       // Chat input query display
      purpleHover: '#9880E8',
      purpleDark: '#8C7CE0',
      teal: '#00B894',
      tealHover: '#00D4AA',
    },
    // Borders
    border: {
      light: '#E4E6EB',
      medium: '#D9D9D9',
      strong: '#BFBFBF',
    },
    // Card styles
    card: {
      bg: '#FFFFFF',           // Chat panel and input area
      text: '#353536',         // Black text
      textSecondary: '#353536',
      border: '#E4E6EB',
      shadow: '0 1px 3px rgba(0, 0, 0, 0.05)',
    },
    // Chat bubbles
    chat: {
      ai: {
        bg: '#F5F5F5',          // LLM output box
        text: '#353536',        // LLM output text (black)
      },
      user: {
        bg: '#F5F3FE',          // Chat input query display
        text: '#353536',        // Black text
      },
    },
    // Input
    input: {
      bg: '#FFFFFF',           // Chat input box
      border: '#D9D9D9',
      placeholder: '#BFBFBF',
      focus: '#A98DF8',
      text: '#353536',
    },
    // Buttons
    button: {
      primary: {
        bg: '#0071CD',
        text: '#FFFFFF',
        hover: '#0057A3',
      },
      secondary: {
        bg: '#F3E8FF',
        text: '#A98DF8',
        border: '#A78BFA',
      },
    },
    // Tab styles
    tab: {
      bg: 'rgba(0, 0, 0, 0.04)',
      text: '#353536',
      activeBg: '#A98DF8',
      activeText: '#FFFFFF',
    },
    // Modal
    modal: {
      shadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
    },
    // Scrollbar
    scrollbar: 'rgba(0, 0, 0, 0.2)',
  },

  dark: {
    // Primary backgrounds - Figma dark theme
    bg: {
      primary: '#0F1729',     // Primary Navy
      secondary: '#18202A',   // Surface bg
      tertiary: '#29303B',    // Card bg
      hover: 'rgba(255, 255, 255, 0.08)',
      active: 'rgba(255, 255, 255, 0.12)',
    },
    // Text colors
    text: {
      primary: '#FFFFFF',
      secondary: '#EEEEEE',
      muted: '#9299A6',
    },
    // Accent colors - from Figma design
    accent: {
      blue: '#0FA0E3',        // Main accent blue
      blueLight: '#0FD4FF',   // Lighter variant
      purple: '#CF44E6',      // Gradient end
      calendar: '#A98DFB',    // Calendar icon
      location: '#F4B400',    // Location icon
    },
    // Borders
    border: {
      light: 'rgba(217, 217, 217, 0.25)',
      medium: 'rgba(255, 255, 255, 0.15)',
      strong: 'rgba(255, 255, 255, 0.25)',
    },
    // Card styles
    card: {
      bg: '#29303B',
      text: '#FFFFFF',
      textSecondary: '#9299A6',
      border: 'rgba(217, 217, 217, 0.25)',
      shadow: '0px 0px 50px 10px rgba(0, 0, 0, 0.25)',
    },
    // Chat bubbles
    chat: {
      ai: {
        bg: '#29303B',
        text: '#FFFFFF',
      },
      user: {
        bg: 'rgba(15, 160, 227, 0.25)',
        text: '#FFFFFF',
      },
    },
    // Input
    input: {
      bg: 'transparent',
      border: 'rgba(255, 255, 255, 0.15)',
      placeholder: '#9299A6',
      focus: '#0FA0E3',
      text: '#FFFFFF',
    },
    // Buttons
    button: {
      primary: {
        bg: '#0F1729',
        text: '#FFFFFF',
        hover: '#18202A',
      },
      secondary: {
        bg: 'rgba(255, 255, 255, 0.1)',
        text: '#FFFFFF',
        border: 'rgba(255, 255, 255, 0.25)',
      },
    },
    // Tab styles
    tab: {
      bg: 'rgba(255, 255, 255, 0.05)',
      text: '#9299A6',
      activeBg: '#0FA0E3',
      activeText: '#FFFFFF',
    },
    // Modal
    modal: {
      shadow: '0px 0px 50px 10px rgba(0, 0, 0, 0.25)',
    },
    // Scrollbar
    scrollbar: 'rgba(255, 255, 255, 0.3)',
    // Gradient for titles
    titleGradient: 'linear-gradient(90deg, #0FA0E3 0%, #CF44E6 100%)',
  },
};

// Shared spacing system
export const SPACING = {
  xs: '4px',
  sm: '8px',
  md: '12px',
  lg: '16px',
  xl: '24px',
  '2xl': '32px',
};

// Shared border radius
export const BORDER_RADIUS = {
  sm: '6px',
  md: '8px',
  lg: '12px',
  xl: '16px',
  full: '9999px',
};

// Typography
export const TYPOGRAPHY = {
 
  fontSize: {
    xs: '12px',
    sm: '13px',
    md: '14px',
    lg: '16px',
    xl: '18px',
    '2xl': '20px',
    '3xl': '24px',
  },
  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
};

// Z-index scale
export const Z_INDEX = {
  widget: 9999,
  modal: 10000,
};

// Transitions
export const TRANSITIONS = {
  fast: '100ms ease',
  normal: '200ms ease',
  slow: '300ms ease',
};

/**
 * Get theme object based on theme name
 * @param {string} themeName - 'light' or 'dark'
 * @returns {object} Theme configuration
 */
export function getTheme(themeName = 'dark') {
  return THEME[themeName] || THEME.dark;
}

/**
 * Merge theme values with component-specific overrides
 * @param {object} baseTheme - Theme object
 * @param {object} overrides - Component-specific overrides
 * @returns {object} Merged theme
 */
export function mergeTheme(baseTheme, overrides = {}) {
  return { ...baseTheme, ...overrides };
}

/**
 * Create CSS variables object for inline styles
 * Useful for generating style objects from theme
 * @param {object} theme - Theme object
 * @returns {object} CSS variables as camelCase object
 */
export function themeToStyles(theme) {
  const styles = {};

  const flatten = (obj, prefix = '') => {
    Object.entries(obj).forEach(([key, value]) => {
      const camelKey = prefix
        ? `${prefix}${key.charAt(0).toUpperCase()}${key.slice(1)}`
        : key;

      if (typeof value === 'object' && value !== null) {
        flatten(value, camelKey);
      } else {
        styles[camelKey] = value;
      }
    });
  };

  flatten(theme);
  return styles;
}
