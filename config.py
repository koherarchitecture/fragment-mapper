"""
Fragment Mapper Configuration

This file controls which features are enabled. For open source deployments,
you can run the tool without authentication or admin features.

Configuration via environment variables (preferred for Docker/production):
    ENABLE_AUTH=0|1          Enable gated access (auth + admin panel)

Configuration via config.env file (preferred for local development):
    Copy config.env.example to config.env and edit values.

Precedence: Environment variables override config.env values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load config.env if it exists (project root)
CONFIG_ENV_PATH = Path(__file__).parent / "config.env"
if CONFIG_ENV_PATH.exists():
    load_dotenv(CONFIG_ENV_PATH)

# Also load .env for API keys and secrets (backwards compatibility)
ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable (0/1, true/false, yes/no)."""
    value = os.environ.get(key, "").lower().strip()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


# =============================================================================
# Feature Flags
# =============================================================================

# Enable email verification, user accounts, and admin panel
# When disabled (0): Open access -- anyone can use the tool without registration
# When enabled (1): Gated access -- users must verify email, admin panel available at /admin
ENABLE_AUTH = get_bool_env("ENABLE_AUTH", default=False)


# =============================================================================
# API Keys (Required)
# =============================================================================

# Hugging Face API token for embedding inference (free account)
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# OpenRouter API key for Stage 3 narration (Claude Haiku)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()


# =============================================================================
# Admin Configuration (always available if ADMIN_PASSWORD is set)
# =============================================================================

# Admin password for /admin panel
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip()


# =============================================================================
# Auth Configuration (only used if ENABLE_AUTH=1)
# =============================================================================

# Session secret for cookie signing
# Required if ENABLE_AUTH=1
SESSION_SECRET = os.environ.get("SESSION_SECRET", "")

# Base URL for verification emails
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")

# Usage limits
MAX_ANALYSES_PER_DAY = int(os.environ.get("MAX_ANALYSES_PER_DAY", "10"))

# Access control -- allowlist (beta phase)
# Comma-separated emails that can register. Empty string = open to all.
ALLOWED_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ALLOWED_EMAILS", "").split(",")
    if e.strip()
}
# Comma-separated domains (e.g. "anu.edu.in,iitb.ac.in"). Empty string = no domain allowlist.
ALLOWED_DOMAINS = {
    d.strip().lower()
    for d in os.environ.get("ALLOWED_DOMAINS", "").split(",")
    if d.strip()
}


# =============================================================================
# SMTP Configuration (only used if ENABLE_AUTH=1)
# =============================================================================

# If SMTP is not configured, verification links are printed to console
# Works with any SMTP provider (SendGrid, Mailgun, AWS SES, Postmark, etc.)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", "noreply@example.com")
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME", "Fragment Mapper")


# =============================================================================
# Paths
# =============================================================================

DB_PATH = Path(__file__).parent / "data" / "fragmentmapper.db"


# =============================================================================
# Analysis Limits
# =============================================================================

MIN_FRAGMENTS = 2
MAX_FRAGMENTS = 20
MIN_FRAGMENT_WORDS = 5
MAX_FRAGMENT_WORDS = 200


# =============================================================================
# Validation
# =============================================================================

def validate_config():
    """Validate configuration. Raises RuntimeError for missing required values."""
    errors = []

    # HF token is always required (for embedding)
    if not HF_TOKEN:
        errors.append("HF_TOKEN is required (free Hugging Face account)")

    # Auth requirements
    if ENABLE_AUTH:
        if not SESSION_SECRET:
            errors.append("SESSION_SECRET is required when ENABLE_AUTH=1")

    if errors:
        raise RuntimeError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))


def print_config_summary():
    """Print configuration summary for logging."""
    print("\n" + "=" * 60)
    print("FRAGMENT MAPPER CONFIGURATION")
    print("=" * 60)
    mode = "gated access (auth + admin)" if ENABLE_AUTH else "open access"
    print(f"  Mode:         {mode}")
    print(f"  HF Token:     {'configured' if HF_TOKEN else 'MISSING'}")
    print(f"  OpenRouter:   {'configured' if OPENROUTER_API_KEY else 'not set (narrator disabled)'}")
    print(f"  Admin:        {'configured' if ADMIN_PASSWORD else 'not set'}")

    if ENABLE_AUTH:
        print(f"  Session:      {'configured' if SESSION_SECRET else 'MISSING'}")
        print(f"  SMTP:         {'configured' if SMTP_HOST else 'console output only'}")
        print(f"  Base URL:     {BASE_URL}")
        print(f"  Limits:       {MAX_ANALYSES_PER_DAY} analyses/user/day")
        if ALLOWED_EMAILS:
            print(f"  Allowlist:    {len(ALLOWED_EMAILS)} email(s)")
        if ALLOWED_DOMAINS:
            print(f"  Domains:      {len(ALLOWED_DOMAINS)} domain(s)")

    print("=" * 60 + "\n")
