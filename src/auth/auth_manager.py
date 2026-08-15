"""
Authentication Manager Engine

Author: Pratim Mistry
Description:
Secure dual-access gatekeeper for Streamlit UI (Public Prediction vs Authorized Retraining).
"""

import hmac
import hashlib


class AuthManager:
    """
    Provides secure role separation and password validation.
    """

    DEFAULT_ADMIN_HASH = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"  # sha256 for 'admin'

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    @classmethod
    def verify_credentials(cls, username: str, password: str) -> bool:
        if username.strip().lower() != "admin":
            return False
        
        entered_hash = cls.hash_password(password.strip())
        return hmac.compare_digest(entered_hash, cls.DEFAULT_ADMIN_HASH)