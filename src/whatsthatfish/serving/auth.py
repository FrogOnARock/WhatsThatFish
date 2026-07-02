"""Google ID-token verification — the single seam between the app and the
identity provider.

We use Google Identity Services (GIS): the SPA obtains a Google ID token via the
`google.accounts.id` SDK, sends it as `Authorization: Bearer <token>`, and this
module verifies it with `google-auth`. `verify_oauth2_token` checks the signature
against Google's public keys and validates issuer, audience (our OAuth client id),
and expiry — so a passing token is a genuine, current Google sign-in.

To switch to Firebase / Identity Platform later, this is the ONLY function that
changes: swap `id_token.verify_oauth2_token` for `firebase_admin.auth.verify_id_token`.
"""

import os
from dotenv import load_dotenv
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from .error import AuthenticationException

# OAuth 2.0 Web Client ID from the Google Cloud console (the token's expected
# audience). Set GOOGLE_OAUTH_CLIENT_ID in the env / Secret Manager.
load_dotenv()
_GOOGLE_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
_request = google_requests.Request()


def verify_google_token(token: str) -> dict:
    """Verify a Google ID token and return its claims (sub, email, name, picture…).

    Raises AuthenticationException (401) for a missing config, or an invalid /
    expired / wrong-audience token.
    """
    if not _GOOGLE_CLIENT_ID:
        raise AuthenticationException(
            "Server auth not configured: GOOGLE_OAUTH_CLIENT_ID is unset"
        )
    try:
        claims = id_token.verify_oauth2_token(token, _request, _GOOGLE_CLIENT_ID)
    except ValueError as exc:
        # verify_oauth2_token raises ValueError on bad signature / aud / iss / exp.
        raise AuthenticationException("Invalid or expired Google token") from exc

    if not claims.get("sub"):
        raise AuthenticationException("Token missing subject")
    return claims
