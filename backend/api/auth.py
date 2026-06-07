"""
Auth API — registration, login, token refresh, and API key management.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from backend.database.mongodb import mongodb
from backend.database.schemas import new_user_doc, serialize_doc
from backend.database.models import (
    RefreshRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from backend.utils.auth import get_current_user
from backend.utils.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    sanitize_input,
    verify_password,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: UserCreate) -> dict[str, Any]:
    """Register a new user account."""
    # Check existing
    email = body.email.strip().lower()
    existing = await mongodb.users.find_one({"email": email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create user
    password_hash = hash_password(body.password)
    doc = new_user_doc(
        email=email,
        password_hash=password_hash,
        name=sanitize_input(body.name, max_length=100),
        role=body.role,
    )
    result = await mongodb.users.insert_one(doc)
    user_id = str(result.inserted_id)

    # Generate tokens
    access_token = create_access_token(user_id, role=body.role)
    refresh_token = create_refresh_token(user_id)

    logger.info("User registered", extra={"user_id": user_id, "email": email})

    return {
        "success": True,
        "data": TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=30 * 60,  # 30 minutes in seconds
            user=UserResponse(
                id=user_id,
                email=email,
                name=body.name,
                role=body.role,
                created_at=doc["created_at"].isoformat(),
            ),
        ),
    }


@router.post("/login")
async def login(body: UserLogin) -> dict[str, Any]:
    """Authenticate and get JWT tokens."""
    email = body.email.strip().lower()
    user = await mongodb.users.find_one({"email": email})

    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    user_id = str(user["_id"])
    role = user.get("role", "user")

    access_token = create_access_token(user_id, role=role)
    refresh_token = create_refresh_token(user_id)

    logger.info("User logged in", extra={"user_id": user_id, "email": email})

    return {
        "success": True,
        "data": TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=30 * 60,
            user=UserResponse(
                id=user_id,
                email=email,
                name=user.get("name", ""),
                role=role,
                created_at=user["created_at"].isoformat(),
            ),
        ),
    }


@router.post("/refresh")
async def refresh_token(body: RefreshRequest) -> dict[str, Any]:
    """Refresh an expired access token using a refresh token."""
    payload = decode_token(body.refresh_token)
    if not payload or payload.role != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user = await mongodb.users.find_one({"_id": payload.sub})
    if not user or not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    user_id = str(user["_id"])
    role = user.get("role", "user")

    new_access = create_access_token(user_id, role=role)
    new_refresh = create_refresh_token(user_id)

    return {
        "success": True,
        "data": TokenResponse(
            access_token=new_access,
            refresh_token=new_refresh,
            token_type="bearer",
            expires_in=30 * 60,
            user=UserResponse(
                id=user_id,
                email=user.get("email", ""),
                name=user.get("name", ""),
                role=role,
                created_at=user["created_at"].isoformat(),
            ),
        ),
    }


@router.get("/me")
async def get_me(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get current authenticated user."""
    return {
        "success": True,
        "data": user,
    }


@router.post("/api-key")
async def generate_new_api_key(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Generate a new API key for the authenticated user."""
    api_key = generate_api_key()
    hashed = hash_api_key(api_key)

    await mongodb.users.update_one(
        {"_id": user["id"]},
        {"$set": {"api_key_hashed": hashed}},
    )

    return {
        "success": True,
        "data": {
            "api_key": api_key,
            "note": "Save this key — it will not be shown again",
        },
    }
