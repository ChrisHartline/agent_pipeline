"""
Client-side encryption for Clara's memory system.

Encrypts personal data BEFORE it leaves the agent, ensuring that
neither Supabase nor FalkorDB can read the actual content.

Uses AES-256-GCM for authenticated encryption.
"""

import os
import base64
import hashlib
import secrets
from typing import Optional, Tuple
from dataclasses import dataclass

# Use cryptography library for proper AES-GCM
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


@dataclass
class EncryptedData:
    """Container for encrypted content with metadata."""
    ciphertext: bytes
    nonce: bytes
    salt: bytes  # For key derivation if using password

    def to_string(self) -> str:
        """Encode as base64 string for storage."""
        combined = self.salt + self.nonce + self.ciphertext
        return base64.b64encode(combined).decode('utf-8')

    @classmethod
    def from_string(cls, encoded: str) -> "EncryptedData":
        """Decode from base64 string."""
        combined = base64.b64decode(encoded.encode('utf-8'))
        salt = combined[:16]
        nonce = combined[16:28]
        ciphertext = combined[28:]
        return cls(ciphertext=ciphertext, nonce=nonce, salt=salt)


class MemoryEncryption:
    """
    Client-side encryption for memory content.

    Usage:
        # Initialize with a master key (store securely!)
        enc = MemoryEncryption.from_password("your-secure-password")

        # Or use a raw key
        enc = MemoryEncryption(key=your_32_byte_key)

        # Encrypt before storing
        encrypted = enc.encrypt("Personal fact about me")

        # Decrypt after retrieving
        plaintext = enc.decrypt(encrypted)

    Security notes:
        - Key NEVER leaves your machine
        - Each encryption uses a unique nonce (no IV reuse)
        - AES-256-GCM provides authenticated encryption
        - Supabase/FalkorDB only see encrypted blobs
    """

    def __init__(self, key: bytes):
        """
        Initialize with a 32-byte key.

        Args:
            key: 32-byte encryption key (AES-256)
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography package required for encryption. "
                "Install with: pip install cryptography"
            )

        if len(key) != 32:
            raise ValueError("Key must be exactly 32 bytes for AES-256")

        self._key = key
        self._aesgcm = AESGCM(key)

    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> Tuple["MemoryEncryption", bytes]:
        """
        Derive encryption key from a password.

        Args:
            password: User's password/passphrase
            salt: Optional salt (generated if not provided)

        Returns:
            Tuple of (MemoryEncryption instance, salt used)
            Save the salt - you'll need it to decrypt later!
        """
        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # OWASP 2023 recommendation
        )
        key = kdf.derive(password.encode('utf-8'))

        return cls(key), salt

    @classmethod
    def from_env(cls, env_var: str = "CLARA_MEMORY_KEY") -> "MemoryEncryption":
        """
        Load key from environment variable.

        The env var should contain a base64-encoded 32-byte key.
        Generate one with: python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"

        Args:
            env_var: Name of environment variable

        Returns:
            MemoryEncryption instance
        """
        key_b64 = os.environ.get(env_var)
        if not key_b64:
            raise ValueError(f"Environment variable {env_var} not set")

        key = base64.b64decode(key_b64)
        return cls(key)

    @staticmethod
    def generate_key() -> Tuple[bytes, str]:
        """
        Generate a new random encryption key.

        Returns:
            Tuple of (raw key bytes, base64-encoded key string)
        """
        key = secrets.token_bytes(32)
        key_b64 = base64.b64encode(key).decode('utf-8')
        return key, key_b64

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.

        Args:
            plaintext: The text to encrypt

        Returns:
            Base64-encoded encrypted string (safe for storage)
        """
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        salt = secrets.token_bytes(16)   # Not used for encryption, but included for format consistency

        ciphertext = self._aesgcm.encrypt(
            nonce,
            plaintext.encode('utf-8'),
            None  # No additional authenticated data
        )

        encrypted = EncryptedData(ciphertext=ciphertext, nonce=nonce, salt=salt)
        return encrypted.to_string()

    def decrypt(self, encrypted_string: str) -> str:
        """
        Decrypt an encrypted string.

        Args:
            encrypted_string: Base64-encoded encrypted data

        Returns:
            Original plaintext

        Raises:
            ValueError: If decryption fails (wrong key or tampered data)
        """
        try:
            encrypted = EncryptedData.from_string(encrypted_string)
            plaintext_bytes = self._aesgcm.decrypt(
                encrypted.nonce,
                encrypted.ciphertext,
                None
            )
            return plaintext_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")

    def encrypt_if_personal(self, content: str, is_personal: bool = True) -> Tuple[str, bool]:
        """
        Conditionally encrypt based on content sensitivity.

        Args:
            content: The content to potentially encrypt
            is_personal: Whether this is personal/sensitive data

        Returns:
            Tuple of (content (possibly encrypted), was_encrypted)
        """
        if is_personal:
            return self.encrypt(content), True
        return content, False

    def decrypt_if_needed(self, content: str, is_encrypted: bool) -> str:
        """
        Conditionally decrypt based on encryption flag.

        Args:
            content: The content (possibly encrypted)
            is_encrypted: Whether the content is encrypted

        Returns:
            Plaintext content
        """
        if is_encrypted:
            return self.decrypt(content)
        return content


# Fallback for when cryptography isn't available
class NoEncryption:
    """Fallback when encryption is disabled or unavailable."""

    def encrypt(self, plaintext: str) -> str:
        return plaintext

    def decrypt(self, ciphertext: str) -> str:
        return ciphertext

    def encrypt_if_personal(self, content: str, is_personal: bool = True) -> Tuple[str, bool]:
        return content, False

    def decrypt_if_needed(self, content: str, is_encrypted: bool) -> str:
        return content


def get_encryption(
    key: Optional[bytes] = None,
    password: Optional[str] = None,
    env_var: Optional[str] = None,
    disabled: bool = False
) -> MemoryEncryption:
    """
    Factory function to get an encryption instance.

    Args:
        key: Raw 32-byte key
        password: Password to derive key from
        env_var: Environment variable containing base64 key
        disabled: If True, return NoEncryption (for testing)

    Returns:
        Encryption instance
    """
    if disabled or not CRYPTO_AVAILABLE:
        return NoEncryption()

    if key:
        return MemoryEncryption(key)

    if password:
        enc, _ = MemoryEncryption.from_password(password)
        return enc

    if env_var:
        return MemoryEncryption.from_env(env_var)

    # Try default env var
    try:
        return MemoryEncryption.from_env()
    except ValueError:
        # No key configured, return no-op encryption with warning
        import warnings
        warnings.warn(
            "No encryption key configured. Personal data will be stored unencrypted. "
            "Set CLARA_MEMORY_KEY environment variable for encryption."
        )
        return NoEncryption()
