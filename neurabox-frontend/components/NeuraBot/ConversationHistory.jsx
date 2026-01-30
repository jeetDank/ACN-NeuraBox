'use client';

import { MessageSquare, Trash2 } from 'lucide-react';
import { useTheme } from '@/hooks/useTheme';
import styles from './ConversationHistory.module.css';

/**
 * Conversation History sidebar component
 * Shows previous conversations grouped by time
 * Theme-aware: colors from theme configuration
 */
export default function ConversationHistory({ conversations, onSelect, onClear }) {
    const { themeConfig } = useTheme();

    // Group conversations by time period
    const groupConversations = (convs) => {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
        const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

        const groups = {
            today: [],
            yesterday: [],
            lastWeek: [],
            older: [],
        };

        convs.forEach((conv) => {
            const convDate = new Date(conv.updatedAt);
            if (convDate >= today) {
                groups.today.push(conv);
            } else if (convDate >= yesterday) {
                groups.yesterday.push(conv);
            } else if (convDate >= lastWeek) {
                groups.lastWeek.push(conv);
            } else {
                groups.older.push(conv);
            }
        });

        return groups;
    };

    const groups = groupConversations(conversations);

    const formatTime = (date) => {
        return new Date(date).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const renderGroup = (title, items) => {
        if (items.length === 0) return null;
        return (
            <div className={styles.group}>
                <div className={styles.groupTitle} style={{ color: themeConfig.text.secondary }}>
                    {title}
                </div>
                {items.map((conv) => (
                    <button
                        key={conv.id}
                        className={styles.conversationItem}
                        onClick={() => onSelect(conv.id)}
                        style={{
                            color: themeConfig.text.primary,
                        }}
                    >
                        <span className={styles.convTitle}>{conv.title}</span>
                        <span className={styles.convTime} style={{ color: themeConfig.text.muted }}>
                            {formatTime(conv.updatedAt)}
                        </span>
                    </button>
                ))}
            </div>
        );
    };

    return (
        <div className={styles.historyContainer}>
            {conversations.length > 0 ? (
                <>
                    <div className={styles.header}>
                        <span className={styles.headerTitle} style={{ color: themeConfig.text.secondary }}>
                            Your Conversations
                        </span>
                        <button
                            className={styles.clearBtn}
                            onClick={onClear}
                            aria-label="Clear all conversations"
                            style={{ color: themeConfig.text.muted }}
                        >
                            Clear all
                        </button>
                    </div>

                    <div className={styles.conversationsList}>
                        {renderGroup('Last 7 Days', [...groups.today, ...groups.yesterday, ...groups.lastWeek])}
                        {renderGroup('Last Month', groups.older)}
                    </div>
                </>
            ) : (
                <div className={styles.emptyHistory}>
                    <MessageSquare size={20} className={styles.emptyIcon} />
                    <p>No conversations yet</p>
                </div>
            )}
        </div>
    );
}
