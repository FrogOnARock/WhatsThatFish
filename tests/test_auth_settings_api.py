"""Route-level tests (P1) for /auth/me (profile) and its PATCH (settings).

The auth gate is overridden in conftest's authed_client, so these focus on the
profile contract and the app-owned editable fields surviving partial PATCHes.
"""


class TestProfile:
    def test_me_returns_profile_with_app_fields(self, authed_client):
        body = authed_client.get("/auth/me").json()
        assert body["email"] == "diver@test.dev"
        assert body["display_name"] == "Test Diver"
        assert body["preferred_name"] is None
        assert body["unit_system"] == "metric"

    def test_me_401_unauthenticated(self, client):
        assert client.get("/auth/me").status_code == 401


class TestSettingsPatch:
    def test_set_preferred_name_and_units(self, authed_client):
        r = authed_client.patch(
            "/auth/me", json={"preferred_name": "Reef Diver", "unit_system": "imperial"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["preferred_name"] == "Reef Diver"
        assert r.json()["unit_system"] == "imperial"
        # Persisted: a fresh GET reflects it.
        assert authed_client.get("/auth/me").json()["unit_system"] == "imperial"

    def test_partial_patch_preserves_untouched_field(self, authed_client):
        authed_client.patch("/auth/me", json={"unit_system": "imperial"})
        authed_client.patch("/auth/me", json={"preferred_name": "Solo"})
        body = authed_client.get("/auth/me").json()
        assert body["preferred_name"] == "Solo"
        assert body["unit_system"] == "imperial"  # not reset by the 2nd PATCH

    def test_empty_preferred_name_clears_override(self, authed_client):
        authed_client.patch("/auth/me", json={"preferred_name": "Temp"})
        authed_client.patch("/auth/me", json={"preferred_name": ""})
        assert authed_client.get("/auth/me").json()["preferred_name"] is None

    def test_invalid_unit_system_rejected(self, authed_client):
        # Literal["metric","imperial"] → 422 on anything else.
        r = authed_client.patch("/auth/me", json={"unit_system": "furlongs"})
        assert r.status_code == 422
