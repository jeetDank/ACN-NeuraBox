'use client';

import { useTheme } from '@/hooks/useTheme';
import styles from './AnswerSuggestions.module.css';
import { ChevronRight, Calendar, BookOpen, Users, MessageSquare, HelpCircle, FileText, Loader } from 'lucide-react';
import { useState, useEffect } from 'react';

// Icon mapping
const iconMap = {
    Calendar: Calendar,
    BookOpen: BookOpen,
    Users: Users,
    MessageSquare: MessageSquare,
    HelpCircle: HelpCircle,
    FileText: FileText,
    MapPin: (props) => <MessageSquare {...props} />,
    Search: (props) => <HelpCircle {...props} />
};

/**
 * AI-driven answer suggestions fetched from backend
 * Appears below AI responses to guide users to related topics
 * For event responses, shows event-specific follow-up suggestions
 */
export default function AnswerSuggestions({ userQuery, answerText, intent, onSuggestionClick, isEventResponse = false }) {
    const { themeConfig } = useTheme();
    const [suggestions, setSuggestions] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchSuggestions = async () => {
            setLoading(true);
            try {
                const response = await fetch('http://localhost:8000/suggest', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        user_query: userQuery,
                        answer: answerText,
                        intent: intent || 'general'
                    })
                });

                if (!response.ok) {
                    throw new Error(`API error: ${response.status}`);
                }

                const data = await response.json();
                setSuggestions(data.suggestions || []);
            } catch (error) {
                console.error('Failed to fetch suggestions:', error);
                // Fallback to empty suggestions on error
                setSuggestions([]);
            } finally {
                setLoading(false);
            }
        };

        // Only fetch if we have both query and answer
        if (userQuery && answerText) {
            fetchSuggestions();
        }
    }, [userQuery, answerText, intent]);

    if (loading) {
        return (
            <div className={styles.suggestionsContainer} style={{ background: themeConfig.chat.ai.bg }}>
                <div className={styles.loadingPlaceholder}>
                    <Loader size={14} className={styles.spinner} />
                    <span style={{ fontSize: '12px', color: themeConfig.text.muted }}>Finding related topics...</span>
                </div>
            </div>
        );
    }

    if (!suggestions || suggestions.length === 0) {
        return null;
    }

    return (
        <div className={`${styles.suggestionsContainer} ${isEventResponse ? styles.eventMode : ''}`} style={isEventResponse ? { background: 'transparent' } : { background: themeConfig.chat.ai.bg }}>
            {!isEventResponse && <p className={styles.suggestionsLabel}>Related questions:</p>}
            <div className={styles.suggestionsList}>
                {suggestions.map((suggestion) => {
                    const IconComponent = isEventResponse ? null : (iconMap[suggestion.icon] || HelpCircle);
                    return (
                        <button
                            key={suggestion.id}
                            className={`${styles.suggestionItem} ${isEventResponse ? styles.eventSuggestion : ''}`}
                            onClick={() => onSuggestionClick && onSuggestionClick(suggestion.text)}
                            style={isEventResponse ? {
                                color: themeConfig.button.primary.text,
                                background: themeConfig.button.primary.bg,
                                border: 'none',
                                borderRadius: '6px',
                                padding: '10px 16px',
                                cursor: 'pointer',
                                fontSize: '13px',
                                fontWeight: '600',
                                whiteSpace: 'nowrap'
                            } : { color: themeConfig.chat.ai.text }}
                        >
                            {!isEventResponse && IconComponent && <IconComponent size={16} className={styles.suggestionIcon} />}
                            <span className={styles.suggestionText}>{suggestion.text}</span>
                            {!isEventResponse && <ChevronRight size={14} className={styles.suggestionChevron} />}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
