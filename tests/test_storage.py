"""Storage layer tests (P2) — the env-switched constructors (local vs GCS) and
the local contribution round-trip. GCS is mocked; we only assert the SWITCH
picks the right backend, not real cloud calls.
"""

from fastapi.responses import FileResponse

from whatsthatfish.serving import utils


class TestLocalContributionRoundTrip:
    def test_upload_writes_then_retrieve_serves_file(self, tmp_path):
        store = utils.LocalContribution(folder=tmp_path)
        key = "user-1/photo.jpg"
        returned = store.upload(key, b"\xff\xd8jpeg")
        assert returned == key
        # Written to disk under the nested key (dirs created).
        assert (tmp_path / key).read_bytes() == b"\xff\xd8jpeg"
        # Retrieve hands back a FileResponse pointing at that path.
        resp = store.retrieve_image(key)
        assert isinstance(resp, FileResponse)


class TestContributionConstructor:
    def test_local_backend_when_not_on_cloud_run(self, tmp_path, monkeypatch):
        monkeypatch.delenv("K_SERVICE", raising=False)
        store = utils.ContributionConstructor(folder=tmp_path).constructor()
        assert isinstance(store, utils.LocalContribution)

    def test_gcs_backend_on_cloud_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("K_SERVICE", "wtf-api")
        sentinel = object()
        monkeypatch.setattr(utils, "GCSContribution", lambda: sentinel)
        store = utils.ContributionConstructor(folder=tmp_path).constructor()
        assert store is sentinel


class TestStorageConstructor:
    def test_local_image_when_not_on_cloud_run(self, tmp_path, monkeypatch):
        monkeypatch.delenv("K_SERVICE", raising=False)
        store = utils.StorageConstructor(folder=tmp_path).constructor()
        assert isinstance(store, utils.LocalImage)

    def test_gcs_image_on_cloud_run(self, monkeypatch):
        monkeypatch.setenv("K_SERVICE", "wtf-api")
        sentinel = object()
        monkeypatch.setattr(utils, "GCSImage", lambda: sentinel)
        assert utils.StorageConstructor().constructor() is sentinel
