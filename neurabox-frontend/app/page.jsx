import NeuraBot from '@/components/NeuraBot/NeuraBot';

export default function Home() {
    return (
        <main style={{ minHeight: '100vh', padding: '2rem', backgroundColor: '#f5f5f5' }}>
            {/* Main content area */}
            <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
                <h1 style={{ fontSize: '2.5rem', marginBottom: '1rem', color: '#1a1a2e' }}>
                    Applied Client Network - ACN
                </h1>
                <p style={{ fontSize: '1.25rem', color: '#666', marginBottom: '2rem' }}>
                    Intelligent AI Assistant powered by your knowledge base
                </p>
                
                <div style={{ 
                    backgroundColor: 'white', 
                    padding: '2rem', 
                    borderRadius: '8px', 
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                    marginBottom: '2rem'
                }}>
                    <h2 style={{ color: '#1a1a2e', marginBottom: '1rem' }}>Welcome to NeuraBox</h2>
                    <p style={{ color: '#666', lineHeight: '1.6' }}>
                        Ask any questions about Applied Client Network, events, services, and more. 
                        The AI assistant in the bottom-right corner will help you find answers quickly.
                    </p>
                    <ul style={{ color: '#666', marginTop: '1rem' }}>
                        <li>📚 Search through curated knowledge base</li>
                        <li>🎯 Get instant, accurate answers</li>
                        <li>💬 Have conversations about ACN topics</li>
                    </ul>
                </div>

                <p style={{ color: '#888', fontSize: '0.95rem' }}>
                    💡 Tip: Click the chat icon in the bottom-right corner to start asking questions!
                </p>
            </div>

            {/* NeuraBot Widget - renders as floating button */}
            <NeuraBot />
        </main>
    );
}
