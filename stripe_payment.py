"""
Stripe Payment Integration for PermitVista
Handles checkout session creation, payment verification, and temporary record storage.
"""

import os
import time
import logging
import threading
import stripe

# Pricing tiers (amounts in cents)
PRICING_TIERS = {
    "standard": {
        "name": "Standard Permit Certificate",
        "amount": 7500,       # $75.00
        "description": "Official building permit certificate",
    },
    "rush": {
        "name": "Rush Permit Certificate (Same Day)",
        "amount": 12500,      # $125.00
        "description": "Priority same-day permit certificate",
    },
    "premium": {
        "name": "Premium Commercial Certificate",
        "amount": 15000,      # $150.00
        "description": "Commercial property permit certificate with full detail",
    },
}

# Temporary storage for pending payment sessions
# Maps stripe session_id -> { record_data, pricing_tier, created_at }
_pending_sessions: dict[str, dict] = {}
_session_lock = threading.Lock()
SESSION_TTL_SECONDS = 3600  # 1 hour expiry


def init_stripe():
    """Initialize Stripe with API key from environment."""
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    if not stripe.api_key:
        logging.warning("STRIPE_SECRET_KEY not set — payment features disabled")
        return False
    logging.info("Stripe initialized (key starts with %s...)", stripe.api_key[:12])
    return True


def is_stripe_enabled() -> bool:
    return bool(stripe.api_key)


def store_pending_session(session_id: str, record_data: dict, pricing_tier: str, unit_number: str = ""):
    """Store record data for retrieval after successful payment."""
    with _session_lock:
        _pending_sessions[session_id] = {
            "record": record_data,
            "tier": pricing_tier,
            "unit_number": unit_number,
            "created_at": time.time(),
        }
    _cleanup_expired_sessions()


def retrieve_pending_session(session_id: str) -> dict | None:
    """Retrieve and remove stored record data after payment."""
    with _session_lock:
        session = _pending_sessions.pop(session_id, None)
    if session and (time.time() - session["created_at"]) > SESSION_TTL_SECONDS:
        logging.warning("Session %s expired", session_id)
        return None
    return session


def _cleanup_expired_sessions():
    """Remove sessions older than TTL."""
    now = time.time()
    with _session_lock:
        expired = [sid for sid, data in _pending_sessions.items()
                   if (now - data["created_at"]) > SESSION_TTL_SECONDS]
        for sid in expired:
            del _pending_sessions[sid]
    if expired:
        logging.info("Cleaned up %d expired payment sessions", len(expired))


def create_checkout_session(
    record_data: dict,
    pricing_tier: str,
    unit_number: str,
    success_base_url: str,
    cancel_url: str,
) -> dict:
    """
    Create a Stripe Checkout session.

    Returns:
        {"checkout_url": str, "session_id": str} on success
    Raises:
        ValueError for invalid tier
        stripe.error.StripeError for Stripe API issues
    """
    tier = PRICING_TIERS.get(pricing_tier)
    if not tier:
        raise ValueError(f"Invalid pricing tier: {pricing_tier}. Valid: {list(PRICING_TIERS.keys())}")

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": tier["name"],
                    "description": tier["description"],
                },
                "unit_amount": tier["amount"],
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=f"{success_base_url}?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=cancel_url,
    )

    store_pending_session(session.id, record_data, pricing_tier, unit_number)
    logging.info("Created Stripe checkout session %s for tier=%s", session.id, pricing_tier)

    return {
        "checkout_url": session.url,
        "session_id": session.id,
    }


def verify_payment(session_id: str) -> dict | None:
    """
    Verify a Stripe Checkout session payment status.

    Returns:
        Stored session data if payment is confirmed, None otherwise.
    """
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except stripe.error.StripeError as e:
        logging.error("Stripe session retrieve error: %s", e)
        return None

    if session.payment_status != "paid":
        logging.warning("Session %s not paid (status=%s)", session_id, session.payment_status)
        return None

    pending = retrieve_pending_session(session_id)
    if not pending:
        logging.warning("No pending session data for %s (may have expired or already used)", session_id)
        return None

    logging.info("Payment verified for session %s, tier=%s", session_id, pending["tier"])
    return pending


def get_pricing_tiers() -> dict:
    """Return pricing tiers for frontend display."""
    return {
        key: {"name": val["name"], "amount": val["amount"], "price": f"${val['amount'] / 100:.2f}"}
        for key, val in PRICING_TIERS.items()
    }
