'use client';

import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Maximize2, X, Send, Sparkles, Crosshair, CrossIcon, Plus } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import SuggestedPrompts from './SuggestedPrompts';
import styles from './ChatInputBar.module.css';

/**
 * Chat Input Bar - semi-expanded state
 * Shows header, input field, prompts dropdown, and send button
 * Theme-aware: colors from theme configuration
 */
export default function ChatInputBar({ onSend, onExpand, onCollapse, isLoading }) {
    const { themeConfig } = useTheme();
    const [inputValue, setInputValue] = useState('');
    const [showPrompts, setShowPrompts] = useState(false);
    const inputRef = useRef(null);

    useEffect(() => {
        // Focus input on mount
        inputRef.current?.focus();
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (inputValue.trim() && !isLoading) {
            onSend(inputValue);
            setInputValue('');
            onExpand(); // Expand to modal when sending
        }
    };

    const handlePromptSelect = (prompt) => {
        setInputValue(prompt);
        setShowPrompts(false);
        inputRef.current?.focus();
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Escape') {
            onCollapse();
        }
    };

    return (
        <div
            className={styles.inputBarContainer}
            style={{
                background: themeConfig.bg.primary,
                borderColor: themeConfig.border.light,
            }}
        >
            {/* Header */}
            <div className={styles.header} style={{ color: themeConfig.text.primary }}>
                <span className={styles.headerText}>One smart prompt connects you to ACN</span>
                <button
                    className={styles.closeButton}
                    onClick={onCollapse}
                    aria-label="Close chat"
                    style={{ color: themeConfig.text.primary }}
                >
                    <X size={16} />
                </button>
            </div>

            {/* Input Area */}
            <form className={styles.inputArea} onSubmit={handleSubmit}>
                <div className={styles.inputWrapper}>
                    <input
                        ref={inputRef}
                        type="text"
                        className={styles.input}
                        placeholder="Type your query or select from the prompt below"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={isLoading}
                        style={{
                            background: themeConfig.input.bg,
                            color: themeConfig.input.text,
                            border: `1px solid ${themeConfig.input.border}`,
                        }}
                    />
                </div>

                <div className={styles.actions}>
                    {/* Sparkle icon */}
                    <button
                        type="button"
                        className={styles.iconButton}
                        aria-label="AI features"
                        style={{ color: themeConfig.text.muted }}
                    >
                        <Plus size={18} />
                    </button>

                    {/* Prompts dropdown */}
                    <div className={styles.promptsDropdown}>
                        <button
                            type="button"
                            className={styles.promptsButton}
                            onClick={() => setShowPrompts(!showPrompts)}
                        >
                            Prompts
                            <ChevronDown size={14} className={showPrompts ? styles.rotated : ''} />
                        </button>
                        {showPrompts && (
                            <SuggestedPrompts
                                onSelect={handlePromptSelect}
                                onClose={() => setShowPrompts(false)}
                            />
                        )}
                    </div>

                    {/* Send button */}
                    <button
                        type="submit"
                        className={styles.sendButton}
                        disabled={!inputValue.trim() || isLoading}
                        aria-label="Send message"
                    >
                        <Send size={18} />
                    </button>
                </div>
            </form>
        </div>
    );
}
