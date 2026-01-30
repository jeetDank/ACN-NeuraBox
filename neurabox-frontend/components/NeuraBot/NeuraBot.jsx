'use client';

import { useNeuraBot } from '@/hooks/useNeuraBot';
import FloatingButton from './FloatingButton';
import ChatInputBar from './ChatInputBar';
import ChatModal from './ChatModal';
import styles from './NeuraBot.module.css';

/**
 * Main NeuraBot Widget Component
 * Manages the three states: collapsed (button), input-bar, and modal
 */
export default function NeuraBot() {
    const {
        widgetState,
        messages,
        conversations,
        isLoading,
        error,
        expandToInputBar,
        expandToModal,
        collapse,
        minimize,
        startNewConversation,
        sendMessage,
        loadConversation,
        clearHistory,
    } = useNeuraBot();

    return (
        <div className={styles.neuraBotContainer}>
            {/* Collapsed State - Just the floating button */}
            {widgetState === 'collapsed' && (
                <FloatingButton onClick={expandToInputBar} />
            )}

            {/* Input Bar State - Compact chat input */}
            {widgetState === 'input-bar' && (
                <ChatInputBar
                    onSend={sendMessage}
                    onExpand={expandToModal}
                    onCollapse={collapse}
                    isLoading={isLoading}
                />
            )}

            {/* Modal State - Full chat interface */}
            {widgetState === 'modal' && (
                <ChatModal
                    messages={messages}
                    conversations={conversations}
                    isLoading={isLoading}
                    error={error}
                    onSend={sendMessage}
                    onClose={collapse}
                    onMinimize={minimize}
                    onNewConversation={startNewConversation}
                    onLoadConversation={loadConversation}
                    onClearHistory={clearHistory}
                />
            )}
        </div>
    );
}
