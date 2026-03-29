# Fragment Mapper

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Koher](https://img.shields.io/badge/koher-tool-teal.svg)](https://koher.app)

See the structural relationships between your scattered text fragments.

**Part of [Koher](https://koher.app) -- AI handles language. Code handles judgment. Humans make decisions.**

---

## Quick Start

### Run Locally (Open Access Mode)

```bash
git clone https://github.com/koherarchitecture/fragment-mapper.git
cd fragment-mapper
pip install -r backend/requirements.txt
cp config.env.example config.env
# Edit config.env: add your HF_TOKEN and OPENROUTER_API_KEY
PYTHONPATH=. uvicorn backend.main:app --reload
# Open http://localhost:8000
```

Requires Python 3.12+, a free [Hugging Face account](https://huggingface.co/) (for embedding), and an [OpenRouter API key](https://openrouter.ai/) (for narration).

---

## What It Does

Write text fragments. Get a structural map showing how they relate -- where they cluster, where they diverge, where they echo.

### Four Signals (Stage 1: Qualification)

| Signal | Method | What It Captures |
|--------|--------|------------------|
| **Embedding similarity** | HF Inference API (all-MiniLM-L6-v2) | Topical proximity -- shared vocabulary and concepts |
| **TF-IDF distinctiveness** | scikit-learn TfidfVectorizer | What makes each fragment unique within this set |
| **Sentiment polarity** | VADER (nltk) | Emotional temperature (-1 to +1 compound score) |
| **Lexical profile** | Pure Python | Writing character: question density, sentence length, vocabulary richness, hedging, first-person ratio |

### Six Rules (Stage 2: Deterministic Judgment)

| Rule | Signals Used | What It Finds |
|------|-------------|---------------|
| **Neighbours** | Embedding | Groups of mutual semantic similarity |
| **Strays** | Embedding | Fragments with no semantic neighbours |
| **Rifts** | Embedding + Sentiment + Lexical | Same topic, opposing emotional pull |
| **Forks** | Embedding + TF-IDF | Same topic, different emphasis/vocabulary |
| **Echoes** | Embedding + Sentiment + Lexical | Different topics, same voice and feeling |
| **Shifts** | Lexical profile | Voice shift -- markedly different writing character |

### Narration (Stage 3: Language)

Claude Haiku describes what the rules found. It cannot alter the counts or relationships. Observational, not prescriptive.

---

## Configuration

All configuration is managed through `config.env` (copy from `config.env.example`).

### Feature Toggle

| Setting | Values | Default | Description |
|---------|--------|---------|-------------|
| `ENABLE_AUTH` | `0` / `1` | `0` | Gated access: email verification + admin panel |

### Deployment Modes

#### Open Access (Default)

Anyone can use the tool. No registration required.

```env
ENABLE_AUTH=0
HF_TOKEN=your_hf_token
OPENROUTER_API_KEY=your_key
```

#### Gated Access

Users must verify email. Usage limits apply. Admin panel at `/admin`.

```env
ENABLE_AUTH=1
HF_TOKEN=your_hf_token
OPENROUTER_API_KEY=your_key
SESSION_SECRET=your_64_char_secret
ADMIN_PASSWORD=your_admin_password
MAX_ANALYSES_PER_DAY=10
```

### Environment Variables

#### Required (Always)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face API token for embedding (free account) |
| `OPENROUTER_API_KEY` | For Stage 3 narration (Claude Haiku via OpenRouter) |

#### Optional (Always)

| Variable | Description |
|----------|-------------|
| `ADMIN_PASSWORD` | Password for accessing `/admin` panel |

#### Required (if ENABLE_AUTH=1)

| Variable | Description |
|----------|-------------|
| `SESSION_SECRET` | 64-character secret for signing session cookies |
| `BASE_URL` | Base URL for verification emails (e.g., `https://yoursite.com`) |

#### Optional (if ENABLE_AUTH=1)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_ANALYSES_PER_DAY` | `10` | Analyses per user per day |
| `ALLOWED_EMAILS` | -- | Comma-separated email allowlist |
| `ALLOWED_DOMAINS` | -- | Comma-separated domain allowlist |
| `SMTP_HOST` | -- | SMTP server for verification emails |
| `SMTP_PORT` | `587` | SMTP port (usually 587 for TLS) |
| `SMTP_USERNAME` | -- | SMTP username or API key |
| `SMTP_PASSWORD` | -- | SMTP password or API key |
| `SMTP_FROM_EMAIL` | `noreply@example.com` | Sender email address |
| `SMTP_FROM_NAME` | `Fragment Mapper` | Sender display name |

**Note:** If SMTP is not configured, verification links are printed to console (useful for development).

---

## File Structure

```
fragment-mapper/
├── README.md                     # This file
├── LICENSE                       # MIT
├── config.py                     # Configuration loader (feature flags, validation)
├── config.env.example            # Configuration template
├── study.py                      # Study module (terminal walkthrough, no API needed)
├── Dockerfile                    # Container build
├── entrypoint.sh                 # Starts server
├── backend/
│   ├── main.py                   # FastAPI server (conditionally loads auth)
│   ├── narrator.py               # Stage 3 Haiku narrator
│   └── requirements.txt          # Python dependencies
├── frontend/
│   ├── index.html                # Full frontend (with conditional auth)
│   └── koher-ui.css              # Koher UI system
└── src/
    └── fragment_analyser.py      # Stage 1 signals + Stage 2 rules
```

---

## Architecture

This tool demonstrates the Koher three-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Qualification                                     │
│  Four signals: embedding, TF-IDF, sentiment, lexical        │
│  Input: text fragments  Output: similarity matrices,        │
│         sentiment scores, lexical profiles                   │
│  Principle: AI reads language patterns                      │
└─────────────────────────────────────────────────────────────┘
                              |
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Deterministic Rules                               │
│  Pure Python code -- no AI                                  │
│  Input: signal data  Output: structural findings            │
│  Six rules: neighbours, strays, rifts, forks, echoes,       │
│             shifts                                           │
│  Principle: Code handles judgment                           │
└─────────────────────────────────────────────────────────────┘
                              |
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Language Interface                                │
│  Claude Haiku describes the findings                        │
│  Input: structural findings  Output: plain language          │
│         narrative                                            │
│  Principle: AI narrates decisions already made              │
└─────────────────────────────────────────────────────────────┘
```

**Why separate layers?**
- Stage 1 (AI): Good at pattern recognition across language
- Stage 2 (Code): Judgment is auditable, reproducible, explicit
- Stage 3 (AI): Good at narrating decisions already made

When you ask AI to "judge whether this is good," you lose auditability. When you separate the layers, you gain it back.

---

## Running Locally

### Requirements

- Python 3.12+
- A Hugging Face account (free) for embedding inference
- An OpenRouter account for Stage 3 narration

### Step 1: Get API Keys

**Hugging Face (free, no payment required):**
1. Create an account at [huggingface.co](https://huggingface.co/join)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token (read access is sufficient)
4. Copy the token — it starts with `hf_`

**OpenRouter (pay-as-you-go, pennies per analysis):**
1. Create an account at [openrouter.ai](https://openrouter.ai/)
2. Go to [Keys](https://openrouter.ai/keys)
3. Create a new key
4. Add a small credit balance ($5 is enough for thousands of analyses)
5. Copy the key — it starts with `sk-or-`

### Step 2: Install and Run

```bash
# Clone the repository
git clone https://github.com/koherarchitecture/fragment-mapper.git
cd fragment-mapper

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Download the VADER sentiment lexicon (about 1 MB, one-time)
python -c "import nltk; nltk.download('vader_lexicon')"

# Create your configuration
cp config.env.example config.env
```

### Step 3: Add Your Keys

Open `config.env` in a text editor and paste your keys:

```env
HF_TOKEN=hf_your_token_here
OPENROUTER_API_KEY=sk-or-your_key_here
```

### Step 4: Start the Server

```bash
PYTHONPATH=. uvicorn backend.main:app --reload
```

Open `http://localhost:8000` in your browser. Write at least three fragments separated by blank lines and click "Map my fragments."

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate venv: `source .venv/bin/activate` |
| `RuntimeError: HF_TOKEN is required` | Add your token to `config.env` |
| Analysis returns 500 | Check your HF_TOKEN is valid at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Narrative says "unavailable" | Check your OPENROUTER_API_KEY has credit |
| `VADER lexicon not found` | Run `python -c "import nltk; nltk.download('vader_lexicon')"` |

---

## Study Module

The study module (`study.py`) is a terminal walkthrough designed for design students. It explains how the tool works step by step, with live examples. No API keys required.

```bash
python study.py              # Full walkthrough (recommended first time)
python study.py --stage 1    # Stage 1: The four signals
python study.py --stage 2    # Stage 2: The six rules
python study.py --stage 3    # Stage 3: The narrator
python study.py --explore    # Interactive fragment explorer
```

**What each stage covers:**
- **Stage 1:** Shows actual similarity matrices, TF-IDF keywords, sentiment scores, and lexical profiles for five example fragments. Explains what each signal captures and why.
- **Stage 2:** Fires all six rules live, showing thresholds, scores, and which rules activate. Every finding is traceable to a constant in the source code.
- **Stage 3:** Explains the narrator's constraints — what it receives, what it cannot do, why language comes last.
- **Explorer:** Enter your own fragments and see the rules fire in real time (uses TF-IDF as a similarity proxy so no API key is needed).

**Design principles:**
- Written for design students, not computer scientists
- Explains WHY each step exists, not just WHAT it does
- All examples run locally with pre-computed data
- Shows actual scores and thresholds — no hand-waving

### Experiment

After running the study module, try changing a threshold in `src/fragment_analyser.py`:

```python
# Change this from 0.50 to 0.30 and re-run study.py --stage 2
NEIGHBOUR_THRESHOLD = 0.50
```

The rules change. Different fragments cluster. The structure shifts. This is the separation at work — the thresholds are the judgment, and you can see them.

### Docker

```bash
# Build
docker build -t fragment-mapper .

# Run (open access mode)
docker run -p 8000:8000 \
  -e HF_TOKEN=your_hf_token \
  -e OPENROUTER_API_KEY=your_key \
  -v fragment-data:/app/data \
  fragment-mapper

# Run (gated access mode)
docker run -p 8000:8000 \
  -e ENABLE_AUTH=1 \
  -e HF_TOKEN=your_hf_token \
  -e OPENROUTER_API_KEY=your_key \
  -e SESSION_SECRET=your_64_char_secret \
  -e ADMIN_PASSWORD=your_admin_password \
  -e BASE_URL=https://yoursite.com \
  -v fragment-data:/app/data \
  fragment-mapper
```

### Docker Compose

```yaml
version: '3.8'
services:
  fragment-mapper:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      # Uncomment for gated access:
      # - ENABLE_AUTH=1
      # - SESSION_SECRET=${SESSION_SECRET}
      # - ADMIN_PASSWORD=${ADMIN_PASSWORD}
      # - BASE_URL=https://yoursite.com
    volumes:
      - fragment-data:/app/data

volumes:
  fragment-data:
```

---

## Stage 2 Rules

The judgment logic lives in `src/fragment_analyser.py`. It is pure Python -- no AI, no network calls, no randomness.

**Thresholds:**
```python
# Rule 1: Neighbours (agglomerative clustering)
NEIGHBOUR_THRESHOLD = 0.50

# Rule 2: Strays (max similarity below threshold)
STRAY_THRESHOLD = 0.20

# Rule 3: Rifts (topical similarity + opposing sentiment/lexical opposition)
RIFT_SIM_THRESHOLD = 0.35
RIFT_SENTIMENT_DELTA = 0.40

# Rule 4: Forks (high embedding similarity + low TF-IDF similarity)
FORK_EMBED_THRESHOLD = 0.40
FORK_TFIDF_THRESHOLD = 0.15

# Rule 5: Echoes (low embedding similarity + close sentiment + shared lexical traits)
ECHO_EMBED_MAX = 0.25
ECHO_SENTIMENT_WINDOW = 0.15

# Rule 6: Shifts (lexical profile > 1.5 std devs from centroid)
SHIFT_OUTLIER_THRESHOLD = 1.5
```

---

## API Endpoints

### Core Endpoints

#### `POST /api/analyse`

Analyse text fragments and return structural findings with narration.

**Request:**
```json
{
  "fragments": [
    "The screen flattens what was meant to be spatial.",
    "Paper forces a different kind of attention.",
    "Students learn more from confusion than from clarity."
  ]
}
```

**Response:**
```json
{
  "analysis": {
    "fragment_count": 3,
    "neighbours": [[0, 1]],
    "strays": [2],
    "rifts": [],
    "forks": [],
    "echoes": [],
    "shifts": []
  },
  "narrative": "Fragments 1 and 2 orbit the same territory..."
}
```

#### `POST /api/analyse/stream`

Same as above, but streams the narration via Server-Sent Events.

### Auth Endpoints (if ENABLE_AUTH=1)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/register` | POST | Register user, send verification email |
| `/auth/verify/{token}` | GET | Verify email via magic link |
| `/auth/check` | GET | Check authentication status |
| `/auth/logout` | POST | Clear session cookie |

### Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin` | GET | Serve admin panel |
| `/admin/login` | POST | Verify admin password |
| `/admin/users` | GET | List all users (auth mode only) |
| `/admin/stats` | GET | Usage statistics |

---

## Cost Analysis (Haiku Stage 3)

| Scale | Analyses/month | Cost |
|-------|----------------|------|
| Small class (25 x 3) | 75 | ~$0.38 |
| Weekly (100/week) | 400 | ~$2.00 |
| Daily (100/day) | 3,000 | ~$15.00 |

Embedding via Hugging Face Inference API is free.

---

## Licence

MIT -- use it, modify it, ship it.

---

## Part of Koher

This tool demonstrates the Koher architecture: AI handles language, code handles judgment, humans make decisions.

- **Website:** [koher.app](https://koher.app)
- **All tools:** [github.com/koherarchitecture](https://github.com/koherarchitecture)
