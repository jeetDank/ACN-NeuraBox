'use client';

import { Bot, User, Calendar, MapPin, ExternalLink } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import AnswerSuggestions from './AnswerSuggestions';
import styles from './ChatMessage.module.css';

/**
 * Chat Message component
 * Supports user and AI messages, including event card rendering
 * Theme-aware: all colors from theme configuration
 */
export default function ChatMessage({ message, onSuggestionClick }) {
    const { themeConfig } = useTheme();
    const isUser = message.type === 'user';

    // Format timestamp
    const formatTime = (date) => {
        return new Date(date).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    /**
     * Parse events from the AI's text response using Regex
     * Expected format: Numbered lists or bold titles with dates
     */
    const extractEventsFromText = (text) => {
        if (!text) return [];

        const events = [];
        // Regex to capture: "1. **Title**" followed loosely by "Date: <date>"
        // This is a heuristic parser

        const lines = text.split('\n');
        let currentEvent = null;

        lines.forEach(line => {
            // blacklist system instructions that might leak or be numbered
            const lowerLine = line.toLowerCase();
            if (lowerLine.includes('reject all') ||
                lowerLine.includes('today is') ||
                lowerLine.includes('use only') ||
                lowerLine.includes('verify the date') ||
                lowerLine.includes("answer is not")) {
                return;
            }

            // Regex to match "1. Title" - REQUIRES A SPACE after the dot to avoid matching phone numbers (e.g. 383.2952)
            const titleMatch = line.match(/^\d+\.\s+\*\*?(.+?)\*\*?$/) || line.match(/^\d+\.\s+(.+)$/);

            if (titleMatch) {
                const potentialTitle = titleMatch[1].trim();

                // IGNORE if title is just numbers or doesn't have letters
                if (!/[a-zA-Z]/.test(potentialTitle) || potentialTitle.length < 3) {
                    return;
                }

                if (currentEvent && currentEvent.date) events.push(currentEvent); // Push previous only if it had a date
                currentEvent = {
                    id: Math.random().toString(36).substr(2, 9),
                    title: potentialTitle,
                    date: '', // Must be filled to be valid
                    location: 'Online/TBA', // Default
                    type: 'event',
                    url: '' // Event page URL
                };
            } else if (currentEvent) {
                // Try to find Date
                if (line.toLowerCase().includes('date:')) {
                    currentEvent.date = line.split(/date:/i)[1].trim();
                }
                // Try to find Time
                else if (line.toLowerCase().includes('time:')) {
                    currentEvent.time = line.split(/time:/i)[1].trim();
                }
                // Try to find Location
                else if (line.toLowerCase().includes('location:')) {
                    currentEvent.location = line.split(/location:/i)[1].trim();
                }
                // Try to find Learn More URL
                else if (line.toLowerCase().includes('learn more:')) {
                    // Extract URL from markdown link format: [text](url)
                    const urlMatch = line.match(/\[.*?\]\((https?:\/\/[^)]+)\)/);
                    if (urlMatch) {
                        currentEvent.url = urlMatch[1];
                    }
                }
            }
        });
        // Only push the last event if it looks valid 
        if (currentEvent && currentEvent.date && !currentEvent.title.toLowerCase().includes('reject')) {
            events.push(currentEvent);
        }

        return events;
    };

    const extractedEvents = !isUser ? extractEventsFromText(message.text) : [];
    const hasEvents = extractedEvents.length > 0;

    // Render event cards with theme-aware styles
    const renderEventCards = () => {
        return (

            <>
                <div className={styles.eventsContainer}>
                    <p className={styles.eventsIntro}>Here are the upcoming events:</p>
                    {extractedEvents.map((event) => (
                        <div
                            key={event.id}
                            className={styles.eventCard}
                            style={{
                                background: themeConfig.card.bg,
                                border: `1px solid ${themeConfig.card.border}`,
                                color: themeConfig.card.text,
                            }}
                        >
                            <h4 className={styles.eventTitle} style={{ color: themeConfig.card.text }}>
                                {event.title}
                            </h4>
                            <div className={styles.eventMeta} style={{ color: themeConfig.card.textSecondary }}>
                                <div className={styles.eventDate}>
                                    <Calendar size={14} />
                                    <span>
                                        {event.date || 'Date TBA'}
                                        {event.time ? ` | ${event.time}` : ''}
                                    </span>
                                </div>
                                <div className={styles.eventLocation}>
                                    <MapPin size={14} />
                                    <span>{event.location}</span>
                                </div>
                            </div>
                            <a
                                href={event.url || 'https://www.appliedclientnetwork.org/events'}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={styles.eventButton}
                                style={{
                                    background: themeConfig.button.primary.bg,
                                    color: themeConfig.button.primary.text,
                                    textDecoration: 'none',
                                    display: 'inline-flex',
                                    alignItems: 'center',
                                    gap: '6px',
                                }}
                            >
                                {event.title.toLowerCase().includes('webinar') ? 'Register' : 'Learn More'}
                                <ExternalLink size={14} />
                            </a>
                        </div>
                    ))}
                </div>
                <div className={styles.suggestionsWrapper}>
                    <AnswerSuggestions
                        userQuery={message.userQuery || message.text}
                        answerText={message.text}
                        intent={message.intent || 'general'}
                        onSuggestionClick={onSuggestionClick}
                        isEventResponse={hasEvents}
                    />
                </div>

            </>






        );
    };

    return (
        <div className={`${styles.messageWrapper} ${isUser ? styles.userMessage : hasEvents ? styles.aiMsgUpdated : styles.aiMessage}`}>
            {/* Avatar */}
            {!isUser && (
                <div className={styles.avatar} style={{ color: themeConfig.accent.purple }}>
                    {/* <Bot size={18} /> */}
                    <img src="/images/logo_text.png" alt="" />
                </div>
            )}

            {/* Message content */}
            <div className={styles.messageContent}>
                {isUser ? (
                    <div
                        className={styles.userBubble}
                        style={{
                            background: themeConfig.chat.user.bg,
                            color: themeConfig.chat.user.text,
                        }}
                    >
                        {message.text}
                    </div>
                ) : (
                    <>

                        {hasEvents ? (

                            renderEventCards()


                        ) : (
                            <></>
                        )}
                        {/* Suggestions positioned at top-right for event responses */}

                        <div
                            className={styles.aiBubbleContainer}
                            style={{
                                background: themeConfig.chat.ai.bg,
                                color: themeConfig.chat.ai.text,
                            }}
                        >
                            <div
                                className={hasEvents ? '' : styles.aiBubble}
                            >

                                {!hasEvents ? (<p className={styles.messageText} style={{ whiteSpace: 'pre-wrap' }}>
                                    {message.text}
                                </p>) : ""}



                                {/* Display confidence and sources (common footer) */}
                                {message.confidence !== undefined && false && (
                                    <div style={{ marginTop: '0.75rem', fontSize: '0.85rem', color: themeConfig.text.muted }}>
                                        <div style={{ marginBottom: '0.5rem' }}>
                                            <strong>Confidence:</strong> {(message.confidence * 100).toFixed(0)}%
                                        </div>
                                        {message.sources && message.sources.length > 0 && (
                                            <div>
                                                <strong>Sources:</strong>
                                                <ul style={{ marginTop: '0.25rem', marginLeft: '1rem', color: themeConfig.accent.purple }}>
                                                    {message.sources.map((source, idx) => (
                                                        <li key={idx} style={{ wordBreak: 'break-word', fontSize: '0.8rem' }}>
                                                            <a
                                                                href={source}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                style={{ color: themeConfig.accent.purple, textDecoration: 'underline' }}
                                                            >
                                                                {source}
                                                            </a>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>


                            {/* Related suggestions below the answer - for non-event responses */}
                            {!hasEvents && (
                                <AnswerSuggestions
                                    userQuery={message.userQuery || message.text}
                                    answerText={message.text}
                                    intent={message.intent || 'general'}
                                    onSuggestionClick={onSuggestionClick}
                                    isEventResponse={false}
                                />
                            )}
                        </div>
                    </>
                )}
                <span className={styles.timestamp} style={{ color: themeConfig.text.muted }}>
                    {formatTime(message.timestamp)}
                </span>
            </div>

            {/* User label */}
            {isUser && (
                <span className={styles.userLabel} style={{ color: themeConfig.text.muted }}>
                    You
                </span>
            )}
        </div>
    );
}
