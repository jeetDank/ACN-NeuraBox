'use client';

import { useState, useRef, useEffect } from 'react';
import { X, Minus, Send, Plus, Sparkles, History, Lightbulb } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import ConversationHistory from './ConversationHistory';
import ChatMessage from './ChatMessage';
import SuggestedPrompts from './SuggestedPrompts';
import ThemeToggle from './ThemeToggle';
import styles from './ChatModal.module.css';
import Image from 'next/image';

export default function ChatModal({
    messages,
    conversations,
    isLoading,
    error,
    onSend,
    onClose,
    onMinimize,
    onNewConversation,
    onLoadConversation,
    onClearHistory,
}) {
    const { themeConfig } = useTheme();

    const [inputValue, setInputValue] = useState('');
    const [showPrompts, setShowPrompts] = useState(false);
    const [sidebarTab, setSidebarTab] = useState('prompts'); // 'history' | 'prompts'

    const messagesEndRef = useRef(null);
    const inputRef = useRef(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (inputValue.trim() && !isLoading) {
            onSend(inputValue);
            setInputValue('');
        }
    };

    const handlePromptSelect = (prompt) => {
        if (prompt.trim() && !isLoading) {
            onSend(prompt);
            setShowPrompts(false);
        }
    };

    const handleSuggestionClick = (suggestion) => {
        if (suggestion.trim() && !isLoading) {
            onSend(suggestion);
        }
    };

    const handleHistorySelect = (conversationId) => {
        onLoadConversation(conversationId);
    };

    /* ---------- RENDER FUNCTIONS ---------- */

    const renderIconRail = () => (
        <div className={styles.iconRail}>
            <div className={styles.logo}>
                <img
                    src="/images/logo.jpg"
                    alt="NeuraBox"
                    className={styles.logoImage}
                />
            </div>


            <button
                className={`${styles.railBtn} ${sidebarTab === 'prompts' ? styles.activeRail : ''}`}
                onClick={() => setSidebarTab('prompts')}
                aria-label="Suggestions"
            >
                <Lightbulb size={18} />
            </button>

            <button
                className={`${styles.railBtn} ${sidebarTab === 'history' ? styles.activeRail : ''}`}
                onClick={() => setSidebarTab('history')}
                aria-label="History"
            >
                <History size={18} />
            </button>

            <div className={styles.railBottom}>
                <ThemeToggle />
            </div>
        </div>
    );

    const renderSidebarContent = () => (
        <div className={styles.sidebar}>
            <div className={styles.sidebarHeader}>
                <h2 className={styles.sidebarTitle}>
                    <span className={styles.titlePink}>ACN Link</span>
                </h2>

                <button
                    className={styles.newConversationBtn}
                    onClick={onNewConversation}
                >
                    <Plus size={16} />
                    <span>New Conversation</span>
                </button>
            </div>

            <div className={styles.sidebarContent}>
                {sidebarTab === 'history' ? (
                    <ConversationHistory
                        conversations={conversations}
                        onSelect={handleHistorySelect}
                        onClear={onClearHistory}
                    />
                ) : (
                    <SuggestedPrompts onSelect={handlePromptSelect} />
                )}
            </div>
        </div>
    );

    /* ---------- JSX ---------- */

    return (
        <div className={styles.modalContainer}>
            {renderIconRail()}
            {renderSidebarContent()}

            <div className={styles.chatPanel}>
                <div className={styles.chatHeader}>
                    <h3 className={styles.chatTitle}>Upcoming Events</h3>
                    <div className={styles.headerActions}>
                        <button className={styles.headerBtn} onClick={onMinimize}>
                            <Minus size={18} />
                        </button>
                        <button className={styles.headerBtn} onClick={onClose}>
                            <X size={18} />
                        </button>
                    </div>
                </div>

                <div className={styles.messagesArea}>
                    {messages.length === 0 ? (
                        <div className={styles.emptyState}>
                            <Sparkles size={32} />
                            <p>Start a conversation with ACN Link</p>
                        </div>
                    ) : (
                        <>
                            {messages.map((m) => (
                                <ChatMessage 
                                    key={m.id} 
                                    message={m}
                                    onSuggestionClick={handleSuggestionClick}
                                />
                            ))}
                            {isLoading && <div className={styles.loadingDots}><span /><span /><span /></div>}
                            {error && <div className={styles.errorMessage}>{error}</div>}
                            <div ref={messagesEndRef} />
                        
                            
                        </>

                      
                    )}
                </div>

                <form className={styles.inputArea} onSubmit={handleSubmit}>
                    <div className={styles.inputWrapper}>
                        <button
                            type="button"
                            className={styles.inputIconBtn}
                            onClick={() => setShowPrompts(!showPrompts)}
                        >
                            <Plus size={18} />
                        </button>

                        {showPrompts && (
                            <div className={styles.promptsDropdownModal}>
                                <SuggestedPrompts
                                    onSelect={handlePromptSelect}
                                    onClose={() => setShowPrompts(false)}
                                />
                            </div>
                        )}

                        <input
                            ref={inputRef}
                            className={styles.input}
                            placeholder="Type your query"
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            disabled={isLoading}
                        />

                        <button
                            type="submit"
                            className={styles.sendBtn}
                            disabled={!inputValue.trim() || isLoading}
                        >
                            <Send size={18} />
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
}
