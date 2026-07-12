/* History page — "Field Log". Species the signed-in user has logged, grouped by
   effective taxon (corrected_taxon_id) on the server, with a detail panel. */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import StatPill from "../components/history/StatPill";
import { SpeciesCard } from "../components/history/SpeciesCard";
import DetailPanel from "../components/history/DetailPanel";
import { getFieldLog, type FieldSpecies } from "../api/history";
import { prefetchPhotos } from "../api/photoCache";
import { useAuth } from "../auth/AuthContext";
import { ROUTES } from "../routes";

/** Every photo id across the whole field log — used to warm the cache before render. */
function allPhotoIds(species: FieldSpecies[]): string[] {
  return species.flatMap((s) =>
    s.sightings.flatMap((g) => g.photos.map((p) => p.id)),
  );
}

export default function HistoryPage() {
  const { status } = useAuth();
  const navigate = useNavigate();
  // The selected species is in the URL (/field-log/:taxonId) so browser
  // back/forward walk the selections and a card is deep-linkable.
  const { taxonId } = useParams();
  const activeId = taxonId != null ? Number(taxonId) : null;
  const selectSpecies = useCallback(
    (id: number) => navigate(`${ROUTES.fieldLog}/${id}`),
    [navigate],
  );
  const [species, setSpecies] = useState<FieldSpecies[] | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (status !== "signed-in") {
      setSpecies(null);
      return;
    }
    let cancelled = false;
    // Defer render until every photo is in the cache, so the log appears all at
    // once instead of images trickling in. prefetchPhotos always resolves (a
    // broken/slow photo can't hang the page).
    getFieldLog()
      .then(async (s) => {
        await prefetchPhotos(allPhotoIds(s));
        if (!cancelled) setSpecies(s);
      })
      .catch(() => !cancelled && setSpecies([]));
    return () => {
      cancelled = true;
    };
  }, [status]);

  // Re-fetch after an edit (user-triggered). New photos are rare here and any
  // already-seen ones are cache hits, so we don't gate this refresh on prefetch.
  const reload = useCallback(() => {
    getFieldLog()
      .then(async (s) => {
        await prefetchPhotos(allPhotoIds(s));
        setSpecies(s);
      })
      .catch(() => {});
  }, []);

  const filtered = useMemo(() => {
    const list = species ?? [];
    const q = query.trim().toLowerCase();
    if (!q) return list;
    return list.filter((s) =>
      [s.species, s.commonName, s.genus, s.family].some((v) =>
        v?.toLowerCase().includes(q),
      ),
    );
  }, [species, query]);

  if (status === "loading") return <main className="main" />;

  if (status !== "signed-in") {
    return (
      <main className="main">
        <div className="main__inner">
          <div className="auth-card">
            <h1 className="page-header__title">
              Your <em>field</em> log
            </h1>
            <p className="page-header__subtitle">
              Sign in to view the species you've logged.
            </p>
          </div>
        </div>
      </main>
    );
  }

  if (!species)
    return (
      <main className="main">
        <div className="main__inner">
          <div className="history-loading">
            <span className="analyzing__pulse" />
            Loading your field log…
          </div>
        </div>
      </main>
    );

  const active = filtered.find((s) => s.taxonId === activeId) ?? filtered[0];
  const totalSightings = species.reduce((n, s) => n + s.sightingCount, 0);

  return (
    <main className="main">
      <div className="main__inner">
        <header className="page-header">
          <div>
            <div className="page-header__crumb">Workspace · Field Log</div>
            <h1 className="page-header__title">
              Your <em>field</em> log
            </h1>
            <p className="page-header__subtitle">
              Every species you've identified, grouped by your confirmed ID and pinned
              to the dive site where you spotted it.
            </p>
          </div>
        </header>

        <div className="log-stats">
          <StatPill label="species" value={species.length} />
          <StatPill label="total sightings" value={totalSightings} />
        </div>

        <div className="log-toolbar">
          <div className="log-search">
            <span className="log-search__icon" aria-hidden>⌕</span>
            <input
              type="text"
              placeholder="Search common name, scientific, family…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
        </div>

        <div className="pdx-layout">
          <div className="pdx-grid">
            {filtered.map((sp, i) => (
              <SpeciesCard
                key={sp.taxonId}
                sp={sp}
                no={i + 1}
                active={sp.taxonId === active?.taxonId}
                onSelect={selectSpecies}
              />
            ))}
            {filtered.length === 0 && (
              <div className="pdx-empty">
                {query
                  ? `No matches in your log for “${query}”.`
                  : "No species logged yet — save an identification to start your field log."}
              </div>
            )}
          </div>

          {active && <DetailPanel sp={active} onChanged={reload} />}
        </div>
      </div>
    </main>
  );
}
