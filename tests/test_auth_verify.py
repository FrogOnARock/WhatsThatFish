"""Auth seam tests (P2) — verify_google_token (the single identity-provider
boundary) and get_current_user's bearer parsing. The Google verifier is mocked
so no network/keys are needed.
"""

import pytest

from whatsthatfish.serving import auth as auth_mod
from whatsthatfish.serving import dependencies as deps
from whatsthatfish.serving.error import AuthenticationException


class TestVerifyGoogleToken:
    def test_valid_token_returns_claims(self, monkeypatch):
        monkeypatch.setattr(auth_mod, "_GOOGLE_CLIENT_ID", "client-123")
        monkeypatch.setattr(
            auth_mod.id_token,
            "verify_oauth2_token",
            lambda token, req, aud: {"sub": "s1", "email": "x@y.z"},
        )
        claims = auth_mod.verify_google_token("tok")
        assert claims["sub"] == "s1"

    def test_bad_token_raises_401(self, monkeypatch):
        monkeypatch.setattr(auth_mod, "_GOOGLE_CLIENT_ID", "client-123")

        def boom(*a, **k):
            raise ValueError("bad signature")

        monkeypatch.setattr(auth_mod.id_token, "verify_oauth2_token", boom)
        with pytest.raises(AuthenticationException):
            auth_mod.verify_google_token("tok")

    def test_missing_sub_raises_401(self, monkeypatch):
        monkeypatch.setattr(auth_mod, "_GOOGLE_CLIENT_ID", "client-123")
        monkeypatch.setattr(
            auth_mod.id_token,
            "verify_oauth2_token",
            lambda *a, **k: {"email": "no-subject@y.z"},
        )
        with pytest.raises(AuthenticationException):
            auth_mod.verify_google_token("tok")

    def test_unconfigured_client_id_raises_401(self, monkeypatch):
        monkeypatch.setattr(auth_mod, "_GOOGLE_CLIENT_ID", None)
        with pytest.raises(AuthenticationException):
            auth_mod.verify_google_token("tok")


class TestGetCurrentUserParsing:
    def test_missing_header_raises(self, session_factory):
        with session_factory() as s:
            with pytest.raises(AuthenticationException):
                deps.get_current_user(authorization=None, session=s)

    def test_non_bearer_scheme_raises(self, session_factory):
        with session_factory() as s:
            with pytest.raises(AuthenticationException):
                deps.get_current_user(authorization="Basic abc123", session=s)

    def test_valid_bearer_resolves_user(self, session_factory, monkeypatch):
        monkeypatch.setattr(
            deps,
            "verify_google_token",
            lambda tok: {
                "sub": "u1",
                "email": "u1@test.dev",
                "name": "U1",
                "picture": None,
            },
        )
        with session_factory() as s:
            user = deps.get_current_user(authorization="Bearer abc", session=s)
            assert user.email == "u1@test.dev"
            assert user.google_subject_id == "u1"
