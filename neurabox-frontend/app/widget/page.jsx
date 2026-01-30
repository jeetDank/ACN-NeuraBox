'use client';

import { useEffect } from 'react';
import NeuraBot from '@/components/NeuraBot/NeuraBot';

/**
 * Embeddable Widget Page
 * This page is designed to be embedded in an iframe on any webpage
 * The floating button can communicate with parent page via postMessage
 * 
 * Usage in HTML:
 * <iframe id="neurabot-widget" src="http://localhost:3000/widget" 
 *         width="100%" height="100vh" style="border: none;"></iframe>
 */
export default function WidgetPage() {
    useEffect(() => {
        // Notify parent window that widget is ready
        if (window.parent && window.parent !== window) {
            window.parent.postMessage({ 
                type: 'NEURABOT_WIDGET_READY',
                timestamp: Date.now()
            }, '*');
        }

        // Listen for messages from parent window
        const handleMessage = (event) => {
            // Verify origin for security (adjust as needed for production)
            if (event.data.type === 'OPEN_CHAT') {
                // Trigger chat modal opening here if needed
                console.log('Parent requested to open chat');
            }
        };

        window.addEventListener('message', handleMessage);
        return () => window.removeEventListener('message', handleMessage);
    }, []);

    return (
        <div style={{ 
            width: '100%', 
            height: '100vh', 
            margin: 0, 
            padding: 0, 
            backgroundColor: 'transparent',
            overflow: 'hidden'
        }}>
            <NeuraBot />
        </div>
    );
}
