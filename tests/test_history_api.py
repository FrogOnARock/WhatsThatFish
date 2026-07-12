"""Route-level tests (P1) for the dive → observation → photo → field-log API.

Exercise the real HTTP stack via the authed TestClient (get_current_user +
photo storage overridden in conftest). These guard the SPA contract: JSON shapes,
status codes, auth-gating, and the multipart photo round-trip.
"""

import io

import pytest

TAXA = [
    {
        "taxon_id": 1001,
        "zero_index": 0,
        "species": "Amphiprion ocellaris",
        "genus": "Amphiprion",
        "family": "Pomacentridae",
        "common_name": "Clown anemonefish",
    },
    {
        "taxon_id": 2001,
        "zero_index": 2,
        "zero_genus": 1,
        "zero_family": 1,
        "species": "Thalassoma lunare",
        "genus": "Thalassoma",
        "family": "Labridae",
        "common_name": "Moon wrasse",
    },
]


@pytest.fixture
def taxa(seed_taxa):
    return seed_taxa(TAXA)


def _make_dive(client, **body):
    r = client.post("/dives", json={"site_name": "Tulamben", **body})
    assert r.status_code == 200, r.text
    return r.json()


def _make_obs(client, dive_id, **body):
    payload = {"dive_id": dive_id, "predicted_species_index": 0, **body}
    r = client.post("/observations", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


# ─── auth gating ──────────────────────────────────────────────────────────────


class TestAuthGating:
    def test_protected_routes_401_without_token(self, client):
        # client = unauthenticated; get_current_user runs for real → 401.
        for path in ["/dives", "/history", "/me/stats", "/dive_sites", "/auth/me"]:
            assert client.get(path).status_code == 401, f"GET {path}"
        # A protected POST (valid body, so auth is what trips it).
        assert client.post("/dives", json={}).status_code == 401


# ─── dives ────────────────────────────────────────────────────────────────────


class TestDives:
    def test_create_and_list(self, authed_client, taxa):
        created = _make_dive(authed_client, site_name="blue hole")
        assert created["site_name"] == "Blue Hole"  # proper-cased
        assert created["observation_count"] == 0
        assert created["species"] == []

        listed = authed_client.get("/dives").json()
        assert len(listed) == 1
        assert listed[0]["id"] == created["id"]

    def test_list_enrichment_after_observations(self, authed_client, taxa):
        dive = _make_dive(authed_client)
        for idx in (0, 0, 2):
            _make_obs(authed_client, dive["id"], predicted_species_index=idx)
        out = authed_client.get("/dives").json()[0]
        assert out["observation_count"] == 3
        assert {sp["taxon_id"] for sp in out["species"]} == {1001, 2001}

    def test_patch_dive(self, authed_client, taxa):
        dive = _make_dive(authed_client)
        r = authed_client.patch(
            f"/dives/{dive['id']}", json={"notes": "viz 25m", "site_name": "New Site"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["notes"] == "viz 25m"
        assert r.json()["site_name"] == "New Site"

    def test_patch_unknown_dive_404(self, authed_client):
        import uuid

        r = authed_client.patch(f"/dives/{uuid.uuid4()}", json={"notes": "x"})
        assert r.status_code == 404


# ─── observations ─────────────────────────────────────────────────────────────


class TestObservations:
    def test_create_defaults_effective_to_predicted(self, authed_client, taxa):
        dive = _make_dive(authed_client)
        obs = _make_obs(authed_client, dive["id"])
        assert obs["predicted_taxon_id"] == 1001
        assert obs["corrected_taxon_id"] == 1001
        assert obs["label_status"] == "predicted"

    def test_create_unknown_dive_404(self, authed_client, taxa):
        import uuid

        r = authed_client.post(
            "/observations",
            json={"dive_id": str(uuid.uuid4()), "predicted_species_index": 0},
        )
        assert r.status_code == 404

    def test_patch_observation_relabel(self, authed_client, taxa):
        dive = _make_dive(authed_client)
        obs = _make_obs(authed_client, dive["id"], depth_m=5)
        r = authed_client.patch(
            f"/observations/{obs['id']}",
            json={
                "corrected_taxon_id": 2001,
                "label_status": "corrected",
                "depth_m": 18,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["corrected_taxon_id"] == 2001
        assert body["label_status"] == "corrected"
        assert body["depth_m"] == 18

    def test_patch_observation_404(self, authed_client):
        import uuid

        r = authed_client.patch(f"/observations/{uuid.uuid4()}", json={"depth_m": 1})
        assert r.status_code == 404


# ─── photos (multipart round-trip) ────────────────────────────────────────────


class TestPhotos:
    def test_upload_then_fetch_image(self, authed_client, taxa):
        dive = _make_dive(authed_client)
        obs = _make_obs(authed_client, dive["id"])

        files = {"img": ("photo.jpg", io.BytesIO(b"\xff\xd8jpegbytes"), "image/jpeg")}
        data = {
            "observation_id": obs["id"],
            "predicted_species_index": "0",
            "confidence": "0.9",
        }
        up = authed_client.post("/observation_photos", files=files, data=data)
        assert up.status_code == 200, up.text
        photo_id = up.json()["id"]

        # The image is served back from the tmp storage (FileResponse).
        img = authed_client.get(f"/observation_photos/{photo_id}/image")
        assert img.status_code == 200
        assert img.content == b"\xff\xd8jpegbytes"

    def test_fetch_unknown_photo_404(self, authed_client):
        import uuid

        r = authed_client.get(f"/observation_photos/{uuid.uuid4()}/image")
        assert r.status_code == 404


def _upload_photo(client, obs_id, blob=b"\xff\xd8jpeg", idx="0"):
    files = {"img": ("photo.jpg", io.BytesIO(blob), "image/jpeg")}
    r = client.post(
        "/observation_photos",
        files=files,
        data={"observation_id": obs_id, "predicted_species_index": idx},
    )
    assert r.status_code == 200, r.text
    return r.json()


# ─── deletion (blobs + rows, cascade) ─────────────────────────────────────────


class TestDeletion:
    def test_delete_photo_removes_row_and_blob(self, authed_client, taxa, tmp_path):
        dive = _make_dive(authed_client)
        obs = _make_obs(authed_client, dive["id"])
        photo = _upload_photo(authed_client, obs["id"])
        blob = tmp_path / photo["image_path"]
        assert blob.exists()

        r = authed_client.delete(f"/observation_photos/{photo['id']}")
        assert r.status_code == 204, r.text
        # Row gone (image 404) and the storage blob is cleaned up.
        assert authed_client.get(f"/observation_photos/{photo['id']}/image").status_code == 404
        assert not blob.exists()

    def test_delete_observation_cascades_photos(self, authed_client, taxa, tmp_path):
        dive = _make_dive(authed_client)
        obs = _make_obs(authed_client, dive["id"])
        p1 = _upload_photo(authed_client, obs["id"], blob=b"\xff\xd8a")
        p2 = _upload_photo(authed_client, obs["id"], blob=b"\xff\xd8b")

        r = authed_client.delete(f"/observations/{obs['id']}")
        assert r.status_code == 204, r.text
        # Observation gone → its photos' rows + blobs gone; field log empty.
        assert authed_client.patch(f"/observations/{obs['id']}", json={"depth_m": 1}).status_code == 404
        for p in (p1, p2):
            assert not (tmp_path / p["image_path"]).exists()
        assert authed_client.get("/history").json()["total_species"] == 0

    def test_delete_dive_cascades_everything(self, authed_client, taxa, tmp_path):
        dive = _make_dive(authed_client)
        obs = _make_obs(authed_client, dive["id"])
        photo = _upload_photo(authed_client, obs["id"])
        assert (tmp_path / photo["image_path"]).exists()

        r = authed_client.delete(f"/dives/{dive['id']}")
        assert r.status_code == 204, r.text
        assert authed_client.get("/dives").json() == []
        assert not (tmp_path / photo["image_path"]).exists()

    def test_delete_unknown_404(self, authed_client):
        import uuid

        rid = uuid.uuid4()
        assert authed_client.delete(f"/dives/{rid}").status_code == 404
        assert authed_client.delete(f"/observations/{rid}").status_code == 404
        assert authed_client.delete(f"/observation_photos/{rid}").status_code == 404


# ─── hero / card image ────────────────────────────────────────────────────────


def _photo_hero_map(client):
    """photo_id → is_hero across the whole field log."""
    log = client.get("/history").json()
    return {
        p["id"]: p["is_hero"]
        for sp in log["species"]
        for s in sp["sightings"]
        for p in s["photos"]
    }


class TestHeroPhoto:
    def test_set_hero_is_exclusive_per_species(self, authed_client, taxa):
        # Two sightings of the SAME species, one photo each.
        dive = _make_dive(authed_client)
        o1 = _make_obs(authed_client, dive["id"], predicted_species_index=0)
        o2 = _make_obs(authed_client, dive["id"], predicted_species_index=0)
        p1 = _upload_photo(authed_client, o1["id"])
        p2 = _upload_photo(authed_client, o2["id"])

        # Default: nothing is hero.
        assert _photo_hero_map(authed_client) == {p1["id"]: False, p2["id"]: False}

        assert authed_client.post(f"/observation_photos/{p1['id']}/hero").status_code == 204
        assert _photo_hero_map(authed_client) == {p1["id"]: True, p2["id"]: False}

        # Setting the other flips the hero exclusively (only one per species).
        assert authed_client.post(f"/observation_photos/{p2['id']}/hero").status_code == 204
        assert _photo_hero_map(authed_client) == {p1["id"]: False, p2["id"]: True}

    def test_set_hero_unknown_404(self, authed_client):
        import uuid

        assert authed_client.post(f"/observation_photos/{uuid.uuid4()}/hero").status_code == 404


# ─── field log / stats / site search ──────────────────────────────────────────


class TestHistoryAndStats:
    def test_history_groups_species(self, authed_client, taxa):
        dive = _make_dive(authed_client)
        for idx in (0, 0, 2):
            _make_obs(authed_client, dive["id"], predicted_species_index=idx)
        log = authed_client.get("/history").json()
        assert log["total_species"] == 2
        counts = {sp["taxon_id"]: sp["sighting_count"] for sp in log["species"]}
        assert counts == {1001: 2, 2001: 1}

    def test_me_stats(self, authed_client, taxa):
        dive = _make_dive(authed_client)
        _make_obs(authed_client, dive["id"])
        assert authed_client.get("/me/stats").json() == {
            "dives": 1,
            "observations": 1,
            "unique_species": 1,
        }

    def test_dive_sites_search(self, authed_client, taxa):
        _make_dive(authed_client, site_name="Coral Garden")
        hits = authed_client.get("/dive_sites", params={"q": "coral"}).json()
        assert [h["name"] for h in hits] == ["Coral Garden"]
