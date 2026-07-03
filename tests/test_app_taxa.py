"""
Tests for BuildAppTaxa.run() orchestration + _get_descriptions extraction.

Pure — no DB or live LLM. The builder is constructed via object.__new__ to skip
__init__, then its DB/LLM methods (_cte, _upsert, _get_descriptions) are stubbed,
so we exercise only the run-loop logic: the null-guard, failure resilience, and
batched upsert. _get_descriptions is tested against a faked anthropic client.
"""


from whatsthatfish.preprocessing.app_taxa import BuildAppTaxa


# ── Helpers ────────────────────────────────────────────────────────────────────


def _row(taxon_id: int, description=None) -> dict:
    return {
        "taxon_id": taxon_id,
        "zero_indexed_species": taxon_id,
        "filename": f"{taxon_id}.jpg",
        "species": f"Species {taxon_id}",
        "genus": "Genus",
        "family": "Family",
        "img_count": 10,
        "description": description,
        "location": None,
        "depth": None,
        "common_name": None,
    }


def _builder(rows, get_descriptions=None) -> BuildAppTaxa:
    """A BuildAppTaxa with DB/LLM methods stubbed.

    Records each upsert batch (as a list of taxon_ids) on `.upserted`, and each
    enrichment call's species on `.enrich_calls`.
    """
    b = object.__new__(BuildAppTaxa)
    b._cte = lambda: rows
    b.upserted = []
    b._upsert = lambda taxa: b.upserted.append([r["taxon_id"] for r in taxa])
    b.enrich_calls = []

    def _default_get(species):
        b.enrich_calls.append(species)
        return (f"cn-{species}", f"desc-{species}", ["loc"], "10m")

    b._get_descriptions = get_descriptions or _default_get
    return b


# ════════════════════════════════════════════════════════════════════════════════
# run() — the null-guard
# ════════════════════════════════════════════════════════════════════════════════


class TestEnrichmentGuard:
    def test_only_missing_descriptions_call_the_llm(self):
        # None and "" are "missing"; a real string is "present".
        # sorted() — enrichment now runs concurrently, so call order is arbitrary.
        rows = [_row(1, None), _row(2, "exists"), _row(3, "")]
        b = _builder(rows)
        b.run(batch_size=100)
        assert sorted(b.enrich_calls) == ["Species 1", "Species 3"]

    def test_present_description_is_preserved(self):
        rows = [_row(2, "exists")]
        b = _builder(rows)
        b.run(batch_size=100)
        assert rows[0]["description"] == "exists"
        assert b.enrich_calls == []

    def test_enriched_values_land_in_the_row(self):
        rows = [_row(1)]
        b = _builder(rows)
        b.run(batch_size=100)
        assert rows[0]["common_name"] == "cn-Species 1"
        assert rows[0]["description"] == "desc-Species 1"
        assert rows[0]["location"] == ["loc"]
        assert rows[0]["depth"] == "10m"


# ════════════════════════════════════════════════════════════════════════════════
# run() — resilience: one failure must not abort the pass or lose work
# ════════════════════════════════════════════════════════════════════════════════


def _flaky(bad_species: str):
    def _get(species):
        if species == bad_species:
            raise RuntimeError("api down")
        return ("cn", "desc", ["loc"], "10m")

    return _get


class TestResilience:
    def test_failed_row_keeps_null_for_retry(self):
        rows = [_row(1), _row(2), _row(3)]
        b = _builder(rows, get_descriptions=_flaky("Species 2"))
        b.run(batch_size=100)
        assert rows[1]["description"] is None  # left NULL → retried next run

    def test_failure_does_not_stop_neighbours(self):
        rows = [_row(1), _row(2), _row(3)]
        b = _builder(rows, get_descriptions=_flaky("Species 2"))
        b.run(batch_size=100)
        assert rows[0]["description"] == "desc"
        assert rows[2]["description"] == "desc"

    def test_all_rows_still_upserted_when_one_enrichment_fails(self):
        rows = [_row(i) for i in range(1, 4)]
        b = _builder(rows, get_descriptions=_flaky("Species 2"))
        b.run(batch_size=100)
        upserted_ids = [tid for batch in b.upserted for tid in batch]
        assert sorted(upserted_ids) == [1, 2, 3]


# ════════════════════════════════════════════════════════════════════════════════
# run() — batched upsert
# ════════════════════════════════════════════════════════════════════════════════


class TestBatching:
    def test_upserts_in_full_batches_plus_remainder(self):
        rows = [_row(i, "x") for i in range(120)]  # preset → no LLM calls
        b = _builder(rows)
        b.run(batch_size=50)
        assert [len(batch) for batch in b.upserted] == [50, 50, 20]

    def test_single_flush_when_under_batch_size(self):
        rows = [_row(i, "x") for i in range(10)]
        b = _builder(rows)
        b.run(batch_size=50)
        assert [len(batch) for batch in b.upserted] == [10]

    def test_exact_multiple_has_no_empty_trailing_batch(self):
        rows = [_row(i, "x") for i in range(100)]
        b = _builder(rows)
        b.run(batch_size=50)
        assert [len(batch) for batch in b.upserted] == [50, 50]

    def test_every_row_is_upserted_exactly_once(self):
        rows = [_row(i, "x") for i in range(57)]
        b = _builder(rows)
        b.run(batch_size=20)
        upserted_ids = [tid for batch in b.upserted for tid in batch]
        assert sorted(upserted_ids) == list(range(57))


# ════════════════════════════════════════════════════════════════════════════════
# _get_descriptions — tool_use extraction
# ════════════════════════════════════════════════════════════════════════════════


class _Block:
    def __init__(self, type_, input_=None):
        self.type = type_
        self.input = input_


class _Resp:
    def __init__(self, content):
        self.content = content


def _fake_client(content):
    """A stand-in for self.client whose messages.create() returns `content`."""

    class _Messages:
        def create(self, **kwargs):
            return _Resp(content)

    class _Client:
        messages = _Messages()

    return _Client()


_VALID_INPUT = {
    "common_name": "Clownfish",
    "description": "d",
    "location": ["reef"],
    "depth": "5m",
}


class TestGetDescriptions:
    def test_extracts_and_validates_tool_use(self):
        b = object.__new__(BuildAppTaxa)
        b.client = _fake_client([_Block("tool_use", _VALID_INPUT)])
        cn, desc, loc, depth = b._get_descriptions("Amphiprion ocellaris")
        assert (cn, desc, loc, depth) == ("Clownfish", "d", ["reef"], "5m")

    def test_skips_leading_text_block(self):
        # A text block before the tool_use block must be ignored.
        b = object.__new__(BuildAppTaxa)
        b.client = _fake_client([_Block("text"), _Block("tool_use", _VALID_INPUT)])
        cn, *_ = b._get_descriptions("X")
        assert cn == "Clownfish"
