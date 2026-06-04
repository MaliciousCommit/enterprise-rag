# src/api/auth.py
#
# JWT Bearer token authentication for the Enterprise RAG API.
#
# HOW JWT AUTHENTICATION WORKS IN OUR SYSTEM:
#
# 1. User authenticates against your identity provider (Auth0, Okta,
#    internal SSO). The IdP issues a signed JWT.
#
# 2. Client includes the JWT in every request:
#    Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
#
# 3. Our FastAPI auth dependency:
#    a. Extracts the token from the Authorization header
#    b. Verifies the signature (using JWT_SECRET_KEY from env)
#    c. Checks expiry (JWT standard claim "exp")
#    d. Extracts user_id from "sub" claim
#    e. Returns user_id to the route handler
#
# DEV MODE (no JWT_SECRET_KEY set):
# Auth is bypassed entirely. Returns "dev-user@localhost" as user_id.
# This lets you develop and test without setting up an IdP.
# NEVER run without JWT_SECRET_KEY in production.
#
# PRODUCTION:
# Set JWT_SECRET_KEY to a random 256-bit secret.
# All tokens must be signed with this key.
# Expired or tampered tokens → HTTP 401.
#
# PHASE EVOLUTION:
# Module 3: HS256 symmetric JWT (simple, single-server)
# Phase 10: RS256 asymmetric JWT (public/private key pair,
#            multiple services can verify without sharing secret)

import logging
import os
from typing import Optional

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# HTTPBearer extracts "Bearer <token>" from the Authorization header.
# auto_error=False means FastAPI won't automatically raise 403 if the
# header is missing — we handle that logic ourselves in get_current_user()
# so we can implement the dev-mode bypass.
bearer_scheme = HTTPBearer(auto_error=False)

# JWT algorithm — HS256 for symmetric (same key signs and verifies).
# RS256 (asymmetric) would be used in production multi-service setups.
JWT_ALGORITHM = "HS256"


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> str:
    """
    FastAPI dependency: extract and verify user identity from JWT.

    Used in route handlers:
        @router.post("/api/v1/query")
        async def query(user_id: str = Depends(get_current_user)):
            # user_id is "alice@company.com" (from JWT sub claim)
            # or "dev-user@localhost" (in dev mode)

    RETURN VALUE:
    The user_id string — typically an email address or UUID from
    the "sub" (subject) claim of the JWT. Used for:
      - Audit logging (who asked what)
      - Rate limiting (per-user request counts)
      - Token budget enforcement (per-user daily token limit)

    RAISES:
      HTTP 401: Token missing (when JWT_SECRET_KEY is set)
      HTTP 401: Token expired (jwt.ExpiredSignatureError)
      HTTP 401: Token invalid (jwt.InvalidTokenError — bad signature, etc.)

    Args:
        credentials: Extracted by HTTPBearer from Authorization header.
                     None if the header is absent.

    Returns:
        str: user_id extracted from the JWT "sub" claim,
             or "dev-user@localhost" in dev mode.
    """
    secret_key = os.getenv("JWT_SECRET_KEY")

    # ── DEV MODE: no secret key set ──────────────────────────────────────────
    # Return a fixed user ID without verifying any token.
    # All rate limits and token budgets apply to this single user.
    if not secret_key:
        logger.debug("Auth: DEV MODE — JWT_SECRET_KEY not set, bypassing auth")
        return "dev-user@localhost"

    # ── PRODUCTION MODE: verify the JWT ──────────────────────────────────────

    # Check that a token was actually provided
    if not credentials or not credentials.credentials:
        logger.warning("Auth: No token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        # Decode and verify the JWT:
        # - Verifies the signature using secret_key
        # - Verifies the "exp" claim (rejects expired tokens)
        # - Verifies the "iat" claim if present
        payload = jwt.decode(
            token,
            secret_key,
            algorithms=[JWT_ALGORITHM],
            # options={"require": ["exp", "sub"]}  # enforce required claims
        )

        # Extract user identity from "sub" (subject) claim
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'sub' claim",
                headers={"WWW-Authenticate": "Bearer"},
            )

        logger.debug(f"Auth: verified user_id='{user_id}'")
        return user_id

    except jwt.ExpiredSignatureError:
        logger.warning("Auth: Token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please re-authenticate.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Auth: Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token. Please re-authenticate.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Type alias ────────────────────────────────────────────────────────────────
from typing import Annotated
from fastapi import Depends

CurrentUser = Annotated[str, Depends(get_current_user)]
# Usage in routes:
#   async def query(user_id: CurrentUser, ...):
#       # user_id is the verified user identifier
