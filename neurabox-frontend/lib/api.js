/**
 * API client for communicating with the NeuraBox backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Send a query to the NeuraBox AI backend
 * @param {string} question - The user's question
 * @param {number} k - Number of context chunks to retrieve (default: 5)
 * @returns {Promise<{answer: string, confidence: number, sources: string[]}>}
 */
export async function queryNeuraBot(question, k = 5) {
    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question, k }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        return {
            answer: data.answer || '',
            confidence: data.confidence || 0,
            sources: data.sources || [],
        };
    } catch (error) {
        console.error('NeuraBot API error:', error);
        throw error;
    }
}

/**
 * Parse AI response for structured data
 * @param {string} answer - Raw answer from AI
 * @param {number} confidence - Confidence score from backend
 * @returns {Object} Parsed response with type and data
 */
export function parseAIResponse(answer, confidence = 0) {
    // Check if confidence is too low
    if (confidence < 0.50) {
        return {
            type: 'lowConfidence',
            rawText: answer,
        };
    }

    // Check if response contains event-like structured data
    const eventPatterns = [
        /applied net 20\d{2}/i,  // Matches 2024, 2025, 2026, etc.
        /quarterly virtual roundtable/i,
        /upcoming events?/i,
        /webinar/i,
        /summit/i,
    ];

    const hasEventContent = eventPatterns.some(pattern => pattern.test(answer));

    if (hasEventContent) {
        return {
            type: 'events',
            rawText: answer,
        };
    }

    return {
        type: 'text',
        rawText: answer,
    };
}
