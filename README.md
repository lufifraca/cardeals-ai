# CarDeals AI

A full-stack car deal aggregator for the Los Angeles area. Users search in plain English — like "best RAV4 lease under $350/mo" — and get real, current offers scraped directly from dealership websites, with source links for every result.

**Live:** Deployed on Vercel (frontend) + Railway (backend)

> 📖 **How the extraction works:** [When to use an LLM (and when not to)](docs/extraction-walkthrough.md) — a walkthrough of the hybrid CSS + GPT-4 pipeline.

---

## Overview

CarDeals AI scrapes 11 dealership websites daily (5 Toyota, 6 Honda across LA), extracts lease and finance offers using a hybrid CSS + GPT-4 pipeline, validates the data, and serves it through a natural language chat interface powered by GPT-4 function calling.

The core idea: every deal shown links back to the actual dealer page it came from. No fabricated numbers, no hallucinated offers.

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS | 8 pages, 7 components, mobile-first |
| Backend | FastAPI (Python, async) | JWT auth, rate limiting, in-memory caching |
| Database | PostgreSQL on Supabase | 5 tables, 9 partial indexes |
| Auth | Supabase (PKCE flow, JWT verification) | ES256 JWKS + HS256 fallback |
| AI | GPT-4 Turbo | Function calling for search, structured extraction for scraping |
| Scraping | Playwright + BeautifulSoup | Headless Chrome for JS-heavy dealer sites |
| Hosting | Vercel + Railway | GitHub Actions cron for daily scraper |

## Key Features

- **AI Chat** — natural language search with 6-message conversation context, fallback recommendations when no exact matches are found, and general car Q&A support
- **Deal Browser** — filterable offer grid with sorting by payment, confidence score, or down payment
- **3-Way Comparison** — select up to 3 offers and compare side by side; AI dynamically fills in specs (horsepower, MPG, etc.) on request
- **Save/Favorites** — bookmark deals and view them from the settings page
- **Usage Tracking** — 10 free AI searches/day per user, with per-source limits (chat vs. compare)

## Scraper Pipeline

> For a full walkthrough of *why* this is hybrid and not pure-LLM, see [docs/extraction-walkthrough.md](docs/extraction-walkthrough.md).
Each dealer site uses a different platform (Octane, DealerOn, DealerInspire, Dealer.com, etc.), so the extraction had to handle completely different HTML structures.

**Approach:** Platform-specific CSS extractors handle the 3 most common platforms (~600 lines of parsing logic). For unrecognized platforms, the pipeline falls back to GPT-4 extraction. This hybrid strategy keeps OpenAI API costs low while maintaining coverage across all 11 dealers.

**Per-dealer flow:**
1. Fetch specials page via Playwright (falls back to `requests` on failure)
2. Detect platform, attempt CSS extraction
3. If CSS fails or returns nothing, clean HTML and send to GPT-4 for structured extraction
4. Validate each offer — payment ranges, known model names, confidence thresholds
5. Deactivate stale offers for that dealer, insert new ones
6. 2-second delay before next dealer

Runs on a daily GitHub Actions cron. Offers older than 48 hours are automatically deactivated.

## Database Schema

5 tables with 9 partial indexes (filtered on `active=TRUE`):

- **dealers** — 11 dealerships with platform metadata
- **offers** — vehicle info, payment details, source URLs, AI confidence scores, raw extraction data (JSONB)
- **users** — email + premium flag (Stripe ID field ready for future payments)
- **search_logs** — raw queries + parsed parameters for analytics
- **chat_usage** — per-user, per-source daily usage tracking

## Project Structure

```
cardeals-ai/
├── frontend/                # Next.js 14
│   ├── app/                 # Home, chat, deals, settings, login, signup, offer detail
│   ├── components/          # ChatInput, ChatMessages, OfferCard, Header, ComparePanel
│   ├── context/             # AuthContext (Supabase session + premium state)
│   └── lib/                 # API client, types, utils, saved offers
│
├── backend/                 # FastAPI (async)
│   ├── api/                 # /chat, /offers, /health endpoints
│   ├── services/            # AI agent, search + caching, usage tracking
│   ├── models.py            # SQLAlchemy ORM
│   └── auth.py              # JWT verification (JWKS)
│
├── scraper/                 # Daily pipeline
│   ├── css_extractors.py    # Platform-specific extraction (3 platforms)
│   ├── extractor.py         # GPT-4 fallback extraction + image mapping
│   ├── fetcher.py           # Playwright + requests
│   ├── validators.py        # Validation rules + model normalization
│   └── saver.py             # DB persistence with dedup
│
└── .github/workflows/       # Daily scraper cron (8 AM UTC)
```

## Running Locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
# Set DATABASE_URL, OPENAI_API_KEY, SUPABASE_JWT_SECRET in .env
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
# Set NEXT_PUBLIC_API_URL=http://localhost:8000 in .env.local
npm run dev
```

**Scraper:**
```bash
cd scraper
pip install -r requirements.txt
playwright install chromium
python main.py
```

## Roadmap

- [ ] Add test coverage (pytest for backend validators and search logic)
- [ ] Wire Stripe for premium subscriptions
- [ ] Expand to more cities and brands
- [ ] Embedding-based search via pgvector
- [ ] Demo video and screenshots

## Acknowledgments

Built with help from Claude (Anthropic) for implementation. Architecture, product decisions, and debugging were my own. See `Co-Authored-By` tags in commit history for full transparency.
