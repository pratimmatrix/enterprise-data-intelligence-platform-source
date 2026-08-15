"""
Authentication Manager Engine

Author: Pratim Mistry
Description:
Role separation for Streamlit UI (Public Prediction vs Authorized Retraining).
"""

import hmac


class AuthManager:
    """
    Credentials management for Admin Retraining Mode.
    """

    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "99"

    @classmethod
    def verify_credentials(cls, username: str, password: str) -> bool:
        """
        Safely verifies admin credentials preventing NoneType crashes 
        and timing attacks.
        """
        if not username or not password:
            return False

        user_clean = str(username).strip()
        pass_clean = str(password).strip()

        # Constant-time comparison for security
        user_match = hmac.compare_digest(user_clean, cls.ADMIN_USERNAME)
        pass_match = hmac.compare_digest(pass_clean, cls.ADMIN_PASSWORD)

        return user_match and pass_match