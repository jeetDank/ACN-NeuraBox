'use client';

import { useTheme } from '@/hooks/useTheme';
import styles from './SuggestedPrompts.module.css';
import { MessageSquare, Calendar, BookOpen, HelpCircle, Users, FileText, ChevronRight } from 'lucide-react';

const SUGGESTED_PROMPTS = [
    {
        id: 'membership',
        icon: Users,
        text: 'Learn About Membership',
    },
    {
        id: 'events',
        icon: Calendar,
        text: 'Explore Events (Applied Net, Webinars)',
    },
    {
        id: 'resources',
        icon: BookOpen,
        text: 'Find Resources & Communities',
    },
    {
        id: 'login',
        icon: HelpCircle,
        text: 'Help with Login or My Account',
    },
    {
        id: 'contact',
        icon: MessageSquare,
        text: 'Contact ACN Team',
    },
    {
        id: 'question',
        icon: HelpCircle,
        text: 'Ask a Question',
    },
    {
        id: 'invoices',
        icon: FileText,
        text: 'Show Me My Invoices',
    },
];

/**
 * Suggested Prompts dropdown menu
 * Theme-aware: colors from theme configuration
 */
export default function SuggestedPrompts({ onSelect, onClose }) {
    const { themeConfig } = useTheme();

    const handlePromptClick = (prompt) => {
        onSelect(prompt.text);
        if (typeof onClose === 'function') {
            onClose();
        }
    };

    return (
        <div className={styles.promptsMenu}>
            <div className={styles.header}>
                <span className={styles.headerLabel} >
                    Suggested Prompts
                </span>
            </div>
            <ul className={styles.promptsList}>
                {SUGGESTED_PROMPTS.map((prompt) => {
                    const Icon = prompt.icon;
                    return (
                        <li key={prompt.id}>
                            <button
                                className={styles.promptItem}
                                onClick={() => handlePromptClick(prompt)}
                               
                            >
                                {/* <Icon size={16} className={styles.promptIcon} style={{ color: themeConfig.accent.purple }} /> */}
                                <span >{prompt.text}</span>
                                <ChevronRight size={14} className={styles.chevron} style={{ color: themeConfig.text.muted }} />
                            </button>
                        </li>
                    );
                })}
            </ul>
        </div>
    );
}
