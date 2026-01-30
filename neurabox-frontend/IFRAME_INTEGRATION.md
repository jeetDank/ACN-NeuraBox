# NeuraBox Widget - Iframe Integration Guide

## Overview

The NeuraBox chat widget can be embedded in any webpage using an HTML iframe. This allows you to add the intelligent ACN AI assistant to your website without modifying your existing codebase.

---

## Quick Start

### 1. Start the Servers

**Terminal 1 - Backend API:**
```bash
cd /home/ubuntu/acn-project
source llm_env/bin/activate
uvicorn api:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd /home/ubuntu/acn-project/neurabox-frontend
npm install  # if first time
npm run dev
```

### 2. View the Demo

Open one of these in your browser:

- **Widget Standalone:** http://localhost:3000/widget
- **Demo Page with iframe:** Open `demo-iframe.html` in your browser (or serve it with a local server)

---

## Basic Integration

Add this iframe code to any HTML page:

```html
<iframe 
    src="http://localhost:3000/widget"
    width="100%"
    height="600px"
    frameborder="0"
    allowfullscreen>
</iframe>
```

That's it! The widget will appear and users can click the floating button to interact with the AI assistant.

---

## How It Works

1. **Widget Page**: `/app/widget/page.jsx` - A Next.js page that renders just the NeuraBot component
2. **Parent Page**: Embeds the widget page using `<iframe>`
3. **Communication**: The iframe communicates with your backend API at `http://localhost:8000/query`
4. **CORS**: Already configured in `api.py` to allow cross-origin requests

---

## File Structure

```
neurabox-frontend/
├── app/
│   ├── page.jsx          # Main landing page (with floating button)
│   ├── layout.jsx        # Root layout
│   └── widget/
│       └── page.jsx      # Embeddable widget (for iframe)
├── components/NeuraBot/
│   ├── NeuraBot.jsx      # Main chat component
│   ├── ChatModal.jsx     # Full-screen modal
│   ├── ChatInputBar.jsx  # Input bar state
│   └── ...               # Other UI components
└── lib/
    └── api.js            # API client that calls /query endpoint
```

---

## Environment Configuration

### Development

The widget uses `http://localhost:8000` as the API URL by default.

Edit `.env.local` if needed:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Production

Update `.env.local` with your deployed API URL:
```
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

Then rebuild and deploy the frontend.

---

## Advanced Usage

### Custom Size

```html
<!-- Tall widget -->
<iframe 
    src="http://localhost:3000/widget"
    width="100%"
    height="800px"
    frameborder="0">
</iframe>

<!-- Sidebar widget -->
<iframe 
    src="http://localhost:3000/widget"
    width="400px"
    height="600px"
    frameborder="0">
</iframe>
```

### Multiple Instances

```html
<!-- Left column -->
<div style="width: 48%; float: left; margin-right: 2%;">
    <iframe 
        src="http://localhost:3000/widget"
        width="100%"
        height="600px"
        frameborder="0">
    </iframe>
</div>

<!-- Right column -->
<div style="width: 48%; float: right;">
    <iframe 
        src="http://localhost:3000/widget"
        width="100%"
        height="600px"
        frameborder="0">
    </iframe>
</div>
```

### Responsive Container

```html
<div style="position: relative; width: 100%; padding-bottom: 66.67%; height: 0; overflow: hidden;">
    <iframe 
        src="http://localhost:3000/widget"
        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
        frameborder="0"
        allowfullscreen>
    </iframe>
</div>
```

### With Custom Styling

```html
<style>
    .widget-frame {
        border: 2px solid #667eea;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        overflow: hidden;
    }
</style>

<div class="widget-frame">
    <iframe 
        src="http://localhost:3000/widget"
        width="100%"
        height="600px"
        frameborder="0">
    </iframe>
</div>
```

---

## API Communication

The widget communicates with your backend using POST requests:

```
POST http://localhost:8000/query

Request Body:
{
    "question": "What is Applied Client Network?",
    "k": 5
}

Response:
{
    "answer": "Applied Client Network is..."
}
```

The API endpoint is defined in `api.py`:
- Accepts questions and context retrieval parameter (k)
- Uses RAG engine to find relevant documents
- Returns AI-generated answers based on your knowledge base

---

## Troubleshooting

### Widget Not Loading

1. **Check Backend is Running**
   ```bash
   # Should return API documentation
   curl http://localhost:8000/docs
   ```

2. **Check Frontend is Running**
   ```bash
   # Should show the widget
   curl http://localhost:3000/widget
   ```

3. **Check CORS Settings**
   - `api.py` has CORS middleware configured for all origins
   - Frontend `.env.local` has `NEXT_PUBLIC_API_URL` set correctly

### Chat Not Working

1. Ensure backend is running (`uvicorn api:app --reload --port 8000`)
2. Check that RAG engine has loaded data (`acn_data/chroma_db/`)
3. Open browser DevTools (F12) → Console tab to see error messages

### Port Already in Use

```bash
# Change backend port
uvicorn api:app --reload --port 9000

# Change frontend port (in .env.local)
# Then run: npm run dev
```

---

## Deployment

### Deploy Frontend to Vercel

```bash
cd neurabox-frontend
vercel
```

Update `.env.local` with production API URL before deploying.

### Deploy Backend to Cloud

```bash
# Example with Heroku
heroku create your-acn-api
git push heroku main
```

Then update iframe src in your website:
```html
<iframe src="https://your-domain.vercel.app/widget"></iframe>
```

---

## File Locations

| File | Purpose |
|------|---------|
| `/neurabox-frontend/app/widget/page.jsx` | Widget page (embeddable) |
| `/demo-iframe.html` | Demo page with iframe example |
| `/api.py` | Backend API server |
| `/neurabox-frontend/.env.local` | Frontend configuration |
| `/neurabox-frontend/lib/api.js` | API client code |

---

## Next Steps

1. ✅ Start backend and frontend servers
2. ✅ Open `demo-iframe.html` to see iframe in action
3. ✅ Copy iframe code to your website
4. ✅ Update API URL in `.env.local` for production
5. ✅ Deploy to your hosting platform

---

## Support

For issues or questions:
- Check browser console for errors (F12)
- Verify both servers are running
- Review `SETUP_INSTRUCTIONS.md` for full setup details
- Check API response at http://localhost:8000/docs

Happy integrating! 🚀
