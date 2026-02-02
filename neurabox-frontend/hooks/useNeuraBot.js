'use client';

import { useState, useCallback } from 'react';
import { queryNeuraBot, parseAIResponse } from '@/lib/api';

/**
 * Custom hook for managing NeuraBot widget state
 */
export function useNeuraBot() {
    // Widget state: 'collapsed' | 'input-bar' | 'modal'
    const [widgetState, setWidgetState] = useState('collapsed');

    // Current conversation messages
    const [messages, setMessages] = useState([]);

    // All conversations (for history sidebar)
    const [conversations, setConversations] = useState([]);

    // Current conversation ID
    const [currentConversationId, setCurrentConversationId] = useState(null);

    // Store conversation messages by ID
    const [conversationMessages, setConversationMessages] = useState({});

    // Loading state
    const [isLoading, setIsLoading] = useState(false);

    // Error state
    const [error, setError] = useState(null);

    // Expand to input bar
    const expandToInputBar = useCallback(() => {
        setWidgetState('input-bar');
        setError(null);
    }, []);

    // Expand to full modal
    const expandToModal = useCallback(() => {
        setWidgetState('modal');
        setError(null);
    }, []);

    // Collapse to button
    const collapse = useCallback(() => {
        setWidgetState('collapsed');
    }, []);

    // Minimize to input bar
    const minimize = useCallback(() => {
        setWidgetState('input-bar');
    }, []);

    // Start a new conversation
    const startNewConversation = useCallback(() => {
        const newId = Date.now().toString();
        setCurrentConversationId(newId);
        setMessages([]);
        setError(null);
    }, []);

    // Send a message
    const sendMessage = useCallback(async (text) => {
        if (!text.trim()) return;

        // Create new conversation if needed
        let convId = currentConversationId;
        if (!convId) {
            convId = Date.now().toString();
            setCurrentConversationId(convId);
        }

        // Add user message
        const userMessage = {
            id: Date.now().toString(),
            type: 'user',
            text: text.trim(),
            timestamp: new Date(),
        };

        // Store message in conversation
        setConversationMessages(prev => ({
            ...prev,
            [convId]: [...(prev[convId] || []), userMessage]
        }));

        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);
        setError(null);

        try {
            const response = await queryNeuraBot(text);
            const parsed = parseAIResponse(response.answer, response.confidence);

            const aiMessage = {
                id: (Date.now() + 1).toString(),
                type: 'ai',
                text: response.answer,
                answer: response.answer,  // ✅ Include answer field
                userQuery: text.trim(),
                intent: response.intent || 'general',
                confidence: response.confidence,
                sources: response.sources,
                ui: response.ui,  // ✅ Include UI structure from backend
                parsedData: parsed,
                timestamp: new Date(),
            };

            // Store AI message in conversation
            setConversationMessages(prev => ({
                ...prev,
                [convId]: [...(prev[convId] || []), aiMessage]
            }));

            setMessages(prev => [...prev, aiMessage]);

            // Update conversation history
            setConversations(prev => {
                const existing = prev.find(c => c.id === convId);
                if (existing) {
                    return prev.map(c =>
                        c.id === convId
                            ? { ...c, lastMessage: text, updatedAt: new Date() }
                            : c
                    );
                }
                return [...prev, {
                    id: convId,
                    title: text.slice(0, 50),
                    lastMessage: text,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                }];
            });

        } catch (err) {
            setError('Failed to get response. Please try again.');
            console.error('Send message error:', err);
        } finally {
            setIsLoading(false);
        }
    }, [currentConversationId]);

    // Load a conversation from history
    const loadConversation = useCallback((conversationId) => {
        setCurrentConversationId(conversationId);
        // Load all messages from this conversation
        const loadedMessages = conversationMessages[conversationId] || [];
        setMessages(loadedMessages);
    }, [conversationMessages]);

    // Clear conversation history
    const clearHistory = useCallback(() => {
        setConversations([]);
        setCurrentConversationId(null);
        setMessages([]);
    }, []);

    return {
        // State
        widgetState,
        messages,
        conversations,
        currentConversationId,
        isLoading,
        error,

        // Actions
        expandToInputBar,
        expandToModal,
        collapse,
        minimize,
        startNewConversation,
        sendMessage,
        loadConversation,
        clearHistory,
    };
}

export default useNeuraBot;
