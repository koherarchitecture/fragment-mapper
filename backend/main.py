"""
Fragment Mapper -- FastAPI Application (v3)

Three-stage pipeline:
1. Four signals: embedding (HF API), TF-IDF, VADER sentiment, lexical profile
2. Six deterministic rules: neighbours, strays, rifts, forks, echoes, shifts
3. Claude Haiku narrates what the rules found

Architecture: AI handles language. Code handles judgment. Humans make decisions.
"""

import asyncio
import json
import secrets
import smtplib
import sqlite3
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from config import (
    ENABLE_AUTH,
    HF_TOKEN,
    OPENROUTER_API_KEY,
    ADMIN_PASSWORD,
    SESSION_SECRET,
    BASE_URL,
    MAX_ANALYSES_PER_DAY,
    ALLOWED_EMAILS,
    ALLOWED_DOMAINS,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    SMTP_FROM_EMAIL,
    SMTP_FROM_NAME,
    DB_PATH,
    MIN_FRAGMENTS,
    MAX_FRAGMENTS,
    MIN_FRAGMENT_WORDS,
    MAX_FRAGMENT_WORDS,
    validate_config,
    print_config_summary,
)


# =============================================================================
# Conditional auth imports
# =============================================================================

if ENABLE_AUTH:
    from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
    serializer = URLSafeTimedSerializer(SESSION_SECRET)

COOKIE_NAME = "fragmentmapper_session"

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Paths
BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fragment-mapper")


def is_email_allowed(email: str) -> bool:
    """Check if email is on the allowlist. If no allowlist is configured, all emails are allowed."""
    if not ALLOWED_EMAILS and not ALLOWED_DOMAINS:
        return True
    email = email.strip().lower()
    if email in ALLOWED_EMAILS:
        return True
    domain = email.split("@")[-1]
    if domain in ALLOWED_DOMAINS:
        return True
    return False


# =============================================================================
# Database Management
# =============================================================================

def init_db():
    """Initialise SQLite database for user management and submissions."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    if ENABLE_AUTH:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                verified INTEGER DEFAULT 0,
                verification_token TEXT,
                token_expires_at TEXT,
                created_at TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0
            )
        """)

    # Check if submissions table has v2 schema (cluster_count) and drop it
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='submissions'")
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(submissions)")
        cols = {row[1] for row in cursor.fetchall()}
        if "cluster_count" in cols or "neighbour_count" not in cols:
            logger.info("Dropping old v2 submissions table -- recreating with v3 schema")
            cursor.execute("DROP TABLE submissions")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_email TEXT NOT NULL,
            fragment_count INTEGER NOT NULL,
            neighbour_count INTEGER NOT NULL DEFAULT 0,
            stray_count INTEGER NOT NULL DEFAULT 0,
            rift_count INTEGER NOT NULL DEFAULT 0,
            fork_count INTEGER NOT NULL DEFAULT 0,
            echo_count INTEGER NOT NULL DEFAULT 0,
            shift_count INTEGER NOT NULL DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()


def get_db():
    """Get database connection."""
    return sqlite3.connect(str(DB_PATH))


# =============================================================================
# User Management (only used when ENABLE_AUTH=1)
# =============================================================================

def register_user(name: str, email: str) -> dict:
    """
    Register a new user or resend verification for an existing user.
    Returns status dict indicating what happened.
    """
    email_lower = email.strip().lower()

    conn = get_db()
    cursor = conn.cursor()

    # Check if user already exists
    cursor.execute(
        "SELECT id, name, email, verified FROM users WHERE email = ?",
        (email_lower,)
    )
    existing = cursor.fetchone()

    if existing:
        user_id, existing_name, existing_email, verified = existing

        # Generate new verification token
        token = secrets.token_urlsafe(32)
        expires = (datetime.now() + timedelta(hours=1)).isoformat()

        cursor.execute(
            "UPDATE users SET verification_token = ?, token_expires_at = ?, name = ? WHERE id = ?",
            (token, expires, name, user_id)
        )
        conn.commit()
        conn.close()

        send_verification_email(name, email_lower, token)

        if verified:
            return {"status": "existing_verified", "message": "Verification link sent"}
        else:
            return {"status": "existing_unverified", "message": "Verification link resent"}

    # New user
    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(hours=1)).isoformat()
    now = datetime.now().isoformat()

    cursor.execute(
        """INSERT INTO users (name, email, verified, verification_token, token_expires_at, created_at, usage_count)
           VALUES (?, ?, 0, ?, ?, ?, 0)""",
        (name, email_lower, token, expires, now)
    )
    conn.commit()
    conn.close()

    send_verification_email(name, email_lower, token)

    return {"status": "created", "message": "Verification link sent"}


def auto_register_verified(name: str, email: str) -> dict:
    """Register user as already verified (for allowed emails). If exists, mark verified."""
    email_lower = email.strip().lower()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT id, verified FROM users WHERE email = ?", (email_lower,))
    existing = cursor.fetchone()

    if existing:
        user_id, verified = existing
        if not verified:
            cursor.execute(
                "UPDATE users SET verified = 1, name = ? WHERE id = ?",
                (name, user_id)
            )
            conn.commit()
        conn.close()
        return {"id": user_id, "name": name, "email": email_lower}

    now = datetime.now().isoformat()
    cursor.execute(
        """INSERT INTO users (name, email, verified, verification_token, token_expires_at, created_at, usage_count)
           VALUES (?, ?, 1, NULL, NULL, ?, 0)""",
        (name, email_lower, now)
    )
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return {"id": user_id, "name": name, "email": email_lower}


def verify_token(token: str) -> Optional[dict]:
    """
    Verify a registration token. Returns user dict if valid, None otherwise.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, name, email, token_expires_at FROM users WHERE verification_token = ?",
        (token,)
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    user_id, name, email, expires_at = row

    # Check expiry
    if expires_at:
        expires = datetime.fromisoformat(expires_at)
        if datetime.now() > expires:
            conn.close()
            return None

    # Mark verified, clear token
    cursor.execute(
        "UPDATE users SET verified = 1, verification_token = NULL, token_expires_at = NULL WHERE id = ?",
        (user_id,)
    )
    conn.commit()
    conn.close()

    return {"id": user_id, "name": name, "email": email}


def get_authenticated_user(request: Request) -> Optional[dict]:
    """Get authenticated user from session cookie. Returns user dict or None."""
    if not ENABLE_AUTH:
        return None

    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None

    try:
        email = serializer.loads(token)
    except (BadSignature, SignatureExpired):
        return None

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, email, verified, usage_count FROM users WHERE email = ? AND verified = 1",
        (email,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    daily_usage = get_daily_usage_count(row[2])

    return {
        "id": row[0],
        "name": row[1],
        "email": row[2],
        "verified": row[3],
        "usage_count": row[4],
        "daily_usage": daily_usage,
        "remaining_analyses": MAX_ANALYSES_PER_DAY - daily_usage,
        "limit_reached": daily_usage >= MAX_ANALYSES_PER_DAY,
    }


def increment_usage(email: str):
    """Increment lifetime usage count for a user."""
    if not ENABLE_AUTH:
        return
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET usage_count = usage_count + 1 WHERE email = ?",
        (email,)
    )
    conn.commit()
    conn.close()


def get_daily_usage_count(email: str) -> int:
    """
    Get how many analyses this user has done today (IST).
    Resets at midnight IST (Indian Standard Time, UTC+5:30).
    """
    today_ist = datetime.now(IST).strftime("%Y-%m-%d")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM submissions WHERE user_email = ? AND date(timestamp) = ?",
        (email, today_ist)
    )
    count = cursor.fetchone()[0]
    conn.close()

    return count


def get_all_users() -> list[dict]:
    """Get all users with their daily usage counts."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, email, verified, created_at, usage_count FROM users ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    users = []
    for row in rows:
        daily = get_daily_usage_count(row[2])
        users.append({
            "id": row[0],
            "name": row[1],
            "email": row[2],
            "verified": row[3],
            "created_at": row[4],
            "usage_count": row[5],
            "daily_usage": daily,
        })

    return users


def log_submission(
    user_email: str,
    fragment_count: int,
    neighbour_count: int,
    stray_count: int,
    rift_count: int,
    fork_count: int,
    echo_count: int,
    shift_count: int,
):
    """Log a submission to the database."""
    now_ist = datetime.now(IST).isoformat()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO submissions
        (timestamp, user_email, fragment_count, neighbour_count, stray_count,
         rift_count, fork_count, echo_count, shift_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (now_ist, user_email, fragment_count, neighbour_count, stray_count,
         rift_count, fork_count, echo_count, shift_count)
    )
    conn.commit()
    conn.close()


# =============================================================================
# Email (only used when ENABLE_AUTH=1)
# =============================================================================

def build_verification_email(name: str, verify_url: str) -> tuple[str, str]:
    """Build HTML and plain text versions of the verification email."""
    plain_text = f"""Hi {name},

Click the link below to verify your email and access Fragment Mapper:

{verify_url}

This link expires in 1 hour.

-- Fragment Mapper"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Georgia, 'Source Serif 4', serif; font-size: 16px; line-height: 1.7; color: #e0e8ea; max-width: 520px; margin: 0 auto; padding: 40px 20px; background: #0f1a1c;">
    <p style="margin-bottom: 20px;">Hi {name},</p>
    <p style="margin-bottom: 24px;">Click the button below to verify your email and access Fragment Mapper.</p>
    <p style="margin-bottom: 32px;">
        <a href="{verify_url}" style="display: inline-block; padding: 14px 28px; background: #d4a157; color: #0f1a1c; text-decoration: none; border-radius: 6px; font-family: 'IBM Plex Sans', sans-serif; font-size: 15px; font-weight: 500;">Verify Email</a>
    </p>
    <p style="font-size: 14px; color: #8a9fa4; margin-bottom: 8px;">Or copy this link:</p>
    <p style="font-size: 13px; color: #8a9fa4; word-break: break-all; margin-bottom: 32px;">{verify_url}</p>
    <p style="font-size: 14px; color: #8a9fa4;">This link expires in 1 hour.</p>
    <hr style="border: none; border-top: 1px solid #2a4248; margin: 32px 0;">
    <p style="font-size: 13px; color: #8a9fa4;">Fragment Mapper &mdash; <a href="https://koher.app" style="color: #d4a157; text-decoration: none;">koher.app</a></p>
</body>
</html>"""

    return html, plain_text


def send_verification_email(name: str, email: str, token: str):
    """Send verification email via SMTP. Always prints URL to console."""
    verify_url = f"{BASE_URL}/auth/verify/{token}"
    html_body, plain_body = build_verification_email(name, verify_url)

    # Always print to console (essential for local dev, useful for debugging)
    print(f"\n{'='*60}")
    print(f"VERIFICATION EMAIL")
    print(f"To: {email}")
    print(f"Verify URL: {verify_url}")
    print(f"{'='*60}\n")

    if not SMTP_HOST or not SMTP_USERNAME:
        return

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    msg["To"] = email
    msg["Subject"] = "Verify your email \u2014 Fragment Mapper"

    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, email, msg.as_string())
        print(f"Verification email sent to {email}")
    except Exception as e:
        print(f"Failed to send verification email to {email}: {e}")


# =============================================================================
# Session Cookie Helpers (only used when ENABLE_AUTH=1)
# =============================================================================

def set_session_cookie(response: Response, email: str):
    """Sign email into a session cookie."""
    token = serializer.dumps(email)
    is_secure = BASE_URL.startswith("https")
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        path="/",
    )


def get_session_email(request: Request) -> Optional[str]:
    """Extract email from session cookie. Returns None if invalid."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        email = serializer.loads(token)
        return email
    except (BadSignature, SignatureExpired):
        return None


# =============================================================================
# Global state (set during lifespan)
# =============================================================================

analyser = None
narrator = None

# Concurrency gate -- up to 3 concurrent analyses.
# Additional requests wait in a queue (up to 60 seconds) rather than being rejected.
MAX_CONCURRENT_ANALYSES = 3
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise analyser and narrator at startup, clean up on shutdown."""
    global analyser, narrator

    # Validate configuration
    validate_config()
    print_config_summary()

    logger.info("Initialising database...")
    init_db()

    logger.info("Initialising fragment analyser...")

    if not HF_TOKEN:
        logger.warning("HF_TOKEN not set -- embedding will fail at analysis time")

    from src.fragment_analyser import FragmentAnalyser
    analyser = FragmentAnalyser(hf_token=HF_TOKEN)
    logger.info("Fragment analyser ready (VADER lexicon loaded, HF API configured)")

    # Stage 3: Haiku narrator
    if OPENROUTER_API_KEY:
        from backend.narrator import FragmentNarrator
        narrator = FragmentNarrator(api_key=OPENROUTER_API_KEY)
        logger.info("Narrator ready (Claude Haiku via OpenRouter)")
    else:
        logger.warning("OPENROUTER_API_KEY not set -- narrator disabled")
        narrator = None

    logger.info("Fragment Mapper ready")
    yield

    logger.info("Shutting down")


# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title="Fragment Mapper",
    description="See the shape of your scattered thinking.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and return them as JSON with details."""
    import traceback
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.url.path}: {type(exc).__name__}: {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# =============================================================================
# Request/Response models
# =============================================================================

class AnalyseRequest(BaseModel):
    """Input: a list of text fragments."""
    fragments: list[str]

    @field_validator("fragments")
    @classmethod
    def validate_fragments(cls, v):
        if len(v) < MIN_FRAGMENTS:
            raise ValueError(f"At least {MIN_FRAGMENTS} fragments required")
        if len(v) > MAX_FRAGMENTS:
            raise ValueError(f"At most {MAX_FRAGMENTS} fragments allowed")

        for i, fragment in enumerate(v):
            word_count = len(fragment.split())
            if word_count < MIN_FRAGMENT_WORDS:
                raise ValueError(
                    f"Fragment {i + 1} has {word_count} words "
                    f"(minimum {MIN_FRAGMENT_WORDS})"
                )
            if word_count > MAX_FRAGMENT_WORDS:
                raise ValueError(
                    f"Fragment {i + 1} has {word_count} words "
                    f"(maximum {MAX_FRAGMENT_WORDS})"
                )

        return v


class RegisterRequest(BaseModel):
    """Input: name and email for registration."""
    name: str
    email: str


class AdminLoginRequest(BaseModel):
    """Input: admin password."""
    password: str


# =============================================================================
# Auth Routes (only registered when ENABLE_AUTH=1)
# =============================================================================

if ENABLE_AUTH:
    @app.post("/auth/register")
    async def auth_register(body: RegisterRequest):
        """Register a new user or resend verification. Allowed emails skip verification."""
        name = body.name.strip()
        email = body.email.strip().lower()

        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        if not email or "@" not in email:
            raise HTTPException(status_code=400, detail="Valid email is required")

        if not is_email_allowed(email):
            raise HTTPException(
                status_code=403,
                detail="This tool is in beta. Your email is not on the access list."
            )

        # Allowed emails: auto-verify and set session cookie immediately
        user = auto_register_verified(name, email)
        response = JSONResponse(
            content={"status": "auto_verified", "message": "Access granted"}
        )
        set_session_cookie(response, email)
        return response

    @app.get("/auth/verify/{token}")
    async def auth_verify(token: str):
        """Verify email token, set session cookie, redirect to /."""
        user = verify_token(token)

        if not user:
            error_html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Verification Failed</title>
<style>body { font-family: Georgia, serif; background: #0f1a1c; color: #e0e8ea; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
.msg { text-align: center; max-width: 400px; } a { color: #d4a157; }</style></head>
<body><div class="msg"><h2>Verification link expired or invalid</h2><p>Please <a href="/">register again</a> to get a new link.</p></div></body></html>"""
            return HTMLResponse(content=error_html, status_code=400)

        # Set session cookie and redirect
        response = HTMLResponse(
            content="<html><head><meta http-equiv='refresh' content='0;url=/'></head><body>Redirecting...</body></html>",
            status_code=200,
        )
        set_session_cookie(response, user["email"])
        return response

    @app.get("/auth/check")
    async def auth_check(request: Request):
        """Check if user is logged in. Returns user info or 401."""
        user = get_authenticated_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="Not authenticated")

        return {
            "name": user["name"],
            "email": user["email"],
            "daily_usage": user["daily_usage"],
            "remaining_analyses": user["remaining_analyses"],
            "limit_reached": user["limit_reached"],
        }

    @app.post("/auth/logout")
    async def auth_logout():
        """Clear session cookie."""
        response = JSONResponse(content={"status": "ok"})
        response.delete_cookie(key=COOKIE_NAME, path="/")
        return response


# =============================================================================
# Frontend Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return HTMLResponse(
        content=index_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/admin", response_class=HTMLResponse)
async def serve_admin():
    """Serve the admin panel."""
    admin_path = FRONTEND_DIR / "admin.html"
    if not admin_path.exists():
        raise HTTPException(status_code=404, detail="Admin panel not found")
    return HTMLResponse(content=admin_path.read_text(encoding="utf-8"))


# Serve static frontend assets (JS, CSS, lib/)
app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend-static")


@app.get("/health")
async def health():
    """Health check endpoint. Reports whether the server can accept an analysis."""
    return {
        "status": "ok",
        "analyser_loaded": analyser is not None,
        "narrator_available": narrator is not None,
        "active_analyses": MAX_CONCURRENT_ANALYSES - analysis_semaphore._value,
        "max_concurrent": MAX_CONCURRENT_ANALYSES,
        "hf_token_set": bool(HF_TOKEN),
        "auth_enabled": ENABLE_AUTH,
    }


# =============================================================================
# Analysis Routes
# =============================================================================

@app.post("/api/analyse")
async def analyse(request: Request, body: AnalyseRequest):
    """
    Full analysis pipeline.

    Stage 1: Four signals (embedding, TF-IDF, sentiment, lexical profile)
    Stage 2: Six rules (neighbours, strays, rifts, forks, echoes, shifts)
    Stage 3: Haiku narration (optional, depends on API key)
    """
    if analyser is None:
        raise HTTPException(status_code=503, detail="Analyser not initialised")

    # Auth check (conditional on ENABLE_AUTH)
    user_email = "anonymous"
    if ENABLE_AUTH:
        user = get_authenticated_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        if user["limit_reached"]:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached ({MAX_ANALYSES_PER_DAY} analyses per day). Resets at midnight IST."
            )
        user_email = user["email"]

    # Concurrency gate -- up to 3 concurrent, then queue with timeout.
    try:
        await asyncio.wait_for(analysis_semaphore.acquire(), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="All analysis slots are busy. Please try again in a moment.",
        )

    try:
        # Stage 1 + Stage 2 (async -- HF API call is async, rest is sync within)
        logger.info(
            f"Analysing {len(body.fragments)} fragments from {user_email}"
        )
        try:
            analysis_dict = await analyser.analyse(body.fragments)
        except Exception as e:
            logger.error(f"Analysis failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis error: {type(e).__name__}")

        # Stage 3: Narration (async)
        narrative = None
        if narrator is not None:
            try:
                narrative = await narrator.narrate(analysis_dict)
            except Exception as e:
                logger.error(f"Narrator failed: {e}")
                narrative = None

        # Log submission
        log_submission(
            user_email=user_email,
            fragment_count=analysis_dict["fragment_count"],
            neighbour_count=len(analysis_dict["neighbours"]),
            stray_count=len(analysis_dict["strays"]),
            rift_count=len(analysis_dict["rifts"]),
            fork_count=len(analysis_dict["forks"]),
            echo_count=len(analysis_dict["echoes"]),
            shift_count=len(analysis_dict["shifts"]),
        )

        if ENABLE_AUTH:
            increment_usage(user_email)
            daily_count = get_daily_usage_count(user_email)
            return {
                "analysis": analysis_dict,
                "narrative": narrative,
                "remaining_today": MAX_ANALYSES_PER_DAY - daily_count,
            }
        else:
            return {
                "analysis": analysis_dict,
                "narrative": narrative,
            }
    finally:
        analysis_semaphore.release()


@app.post("/api/analyse/stream")
async def analyse_stream(request: Request, body: AnalyseRequest):
    """
    Analysis with streaming narration.

    Stage 1 + Stage 2 return immediately via SSE.
    Stage 3 (Haiku) streams as Server-Sent Events.
    """
    if analyser is None:
        raise HTTPException(status_code=503, detail="Analyser not initialised")

    # Auth check (conditional on ENABLE_AUTH)
    user_email = "anonymous"
    if ENABLE_AUTH:
        user = get_authenticated_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        if user["limit_reached"]:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached ({MAX_ANALYSES_PER_DAY} analyses per day)."
            )
        user_email = user["email"]

    # Concurrency gate -- up to 3 concurrent, then queue with timeout.
    try:
        await asyncio.wait_for(analysis_semaphore.acquire(), timeout=60)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="All analysis slots are busy. Please try again in a moment.",
        )

    try:
        # Stage 1 + Stage 2 (async)
        try:
            analysis_dict = await analyser.analyse(body.fragments)
        except Exception as e:
            logger.error(f"Stream analysis failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis error: {type(e).__name__}: {e}")

        # Log submission
        log_submission(
            user_email=user_email,
            fragment_count=analysis_dict["fragment_count"],
            neighbour_count=len(analysis_dict["neighbours"]),
            stray_count=len(analysis_dict["strays"]),
            rift_count=len(analysis_dict["rifts"]),
            fork_count=len(analysis_dict["forks"]),
            echo_count=len(analysis_dict["echoes"]),
            shift_count=len(analysis_dict["shifts"]),
        )

        if ENABLE_AUTH:
            increment_usage(user_email)
            daily_count = get_daily_usage_count(user_email)
    finally:
        analysis_semaphore.release()

    async def event_stream():
        # First event: analysis data (Stage 1 + Stage 2)
        yield f"event: analysis\ndata: {json.dumps(analysis_dict)}\n\n"

        # Second event: remaining count (only in auth mode)
        if ENABLE_AUTH:
            yield f"event: remaining\ndata: {json.dumps({'remaining_today': MAX_ANALYSES_PER_DAY - daily_count})}\n\n"

        # Stream narration if available (async)
        if narrator is not None:
            try:
                async for chunk in narrator.narrate_stream(analysis_dict):
                    yield f"event: narrative\ndata: {json.dumps({'text': chunk})}\n\n"
                yield f"event: narrative_done\ndata: {{}}\n\n"
            except Exception as e:
                logger.error(f"Narrator stream failed: {e}")
                yield f"event: narrative_error\ndata: {json.dumps({'error': str(e)})}\n\n"
        else:
            yield f"event: narrative_unavailable\ndata: {{}}\n\n"

        yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Admin Routes (always available if ADMIN_PASSWORD is set)
# =============================================================================

@app.post("/admin/login")
async def admin_login(body: AdminLoginRequest):
    """Verify admin password."""
    if not ADMIN_PASSWORD or body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorised")
    return {"status": "ok"}


@app.get("/admin/users")
async def admin_users(admin_password: str = ""):
    """List all users with daily usage."""
    if not ADMIN_PASSWORD or admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorised")

    if ENABLE_AUTH:
        users = get_all_users()
        return {"users": users}
    else:
        return {"users": []}


@app.get("/admin/stats")
async def admin_stats(admin_password: str = ""):
    """Aggregate statistics."""
    if not ADMIN_PASSWORD or admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorised")

    conn = get_db()
    cursor = conn.cursor()

    total_users = 0
    verified_users = 0
    if ENABLE_AUTH:
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM users WHERE verified = 1")
        verified_users = cursor.fetchone()[0]

    # Active users (have at least one submission)
    cursor.execute("SELECT COUNT(DISTINCT user_email) FROM submissions")
    active_users = cursor.fetchone()[0]

    # Total analyses
    cursor.execute("SELECT COUNT(*) FROM submissions")
    total_analyses = cursor.fetchone()[0]

    # Average fragments per analysis
    cursor.execute("SELECT AVG(fragment_count) FROM submissions")
    avg_fragments = cursor.fetchone()[0] or 0

    # Average neighbours per analysis
    cursor.execute("SELECT AVG(neighbour_count) FROM submissions")
    avg_neighbours = cursor.fetchone()[0] or 0

    # Average rifts per analysis
    cursor.execute("SELECT AVG(rift_count) FROM submissions")
    avg_rifts = cursor.fetchone()[0] or 0

    conn.close()

    return {
        "total_users": total_users,
        "verified_users": verified_users,
        "active_users": active_users,
        "total_analyses": total_analyses,
        "avg_fragments": round(avg_fragments, 1),
        "avg_neighbours": round(avg_neighbours, 1),
        "avg_rifts": round(avg_rifts, 1),
    }
