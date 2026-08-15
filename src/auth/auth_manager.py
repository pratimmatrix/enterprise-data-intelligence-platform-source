"""
Authentication Manager Engine

Author: Pratim Mistry
Description:
Role separation for Streamlit UI (Public Prediction vs Authorized Retraining).
"""


class AuthManager:
    """
    Credentials management for Admin Retraining Mode.
    """

    # Yahan jo username aur password doge, wahi samne wale ko share karna
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "99"

    @classmethod
    def verify_credentials(cls, username: str, password: str) -> bool:
        return (
            username.strip() == cls.ADMIN_USERNAME 
            and password.strip() == cls.ADMIN_PASSWORD
        )