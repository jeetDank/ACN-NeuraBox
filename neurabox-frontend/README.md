# NeuraBox AI Frontend

A React/Next.js chatbot widget for the NeuraBox AI Agent, designed to integrate with the ACN RAG backend.

## Features

- **3 Widget States**: Collapsed button → Input bar → Full modal
- **Conversation History**: View and manage past conversations
- **Suggested Prompts**: Quick-access common queries
- **Event Cards**: Display structured event information
- **Dark Theme**: Modern, sleek design matching Figma specs

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- ACN RAG Backend running on `http://localhost:8000`

### Installation

```bash
cd neurabox-frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## Configuration

Set the backend API URL via environment variable:

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Project Structure

```
neurabox-frontend/
├── app/
│   ├── globals.css      # Design system & variables
│   ├── layout.jsx       # Root layout
│   └── page.jsx         # Demo page with widget
├── components/
│   └── NeuraBot/
│       ├── NeuraBot.jsx          # Main container
│       ├── FloatingButton.jsx    # Collapsed state
│       ├── ChatInputBar.jsx      # Semi-expanded input
│       ├── ChatModal.jsx         # Full modal view
│       ├── ChatMessage.jsx       # Message bubbles
│       ├── ConversationHistory.jsx
│       └── SuggestedPrompts.jsx
├── hooks/
│   └── useNeuraBot.js    # State management hook
└── lib/
    └── api.js            # Backend API client
```

## Backend API

The frontend communicates with the FastAPI backend:

```
POST /query
Body: { "question": string, "k": number }
Response: { "answer": string }
```

Start the backend:
```bash
cd ..
uvicorn api:app --reload
```

## License

Private - ACN Project
