/* History page — "Field Log". Pokedex-style species grid + detail panel.
   Data comes from the API client (mock for now); sort comparators live in
   SORT_COMPARATORS below. */
import { useEffect, useMemo, useState } from "react";
import StatPill from "../components/history/StatPill.jsx";
import { SpeciesCard, GhostCard } from "../components/history/SpeciesCard.jsx";
import DetailPanel from "../components/history/DetailPanel.jsx";
import { getFieldLog } from "../api/client.js";

const SORTS = [
  { id: "recent", label: "Most recent" },
  { id: "seen",   label: "Most seen" },
  { id: "az",     label: "A → Z" },
  { id: "family", label: "By family" },
];

const lastSeen = (sp) => sp.sightings.map((s) => s.date).sort().pop();

const SORT_COMPARATORS = {
  recent: (a, b) => lastSeen(b).localeCompare(lastSeen(a)),
  seen:   (a, b) => b.sightings.length - a.sightings.length,
  az:     (a, b) => a.common.localeCompare(b.common),
  // TODO(you): "By family" ordering. The prototype sorts alphabetically by
  // family then species, but your inat_taxa ancestry gives you a real
  // taxonomic order — decide whether alphabetical is fine for display or
  // whether grouping should follow ancestry (e.g. sharks before teleosts).
  family: (_a, _b) => 0,
};

function uniqueSites(species) {
  const set = new Set();
  for (const sp of species) for (const s of sp.sightings) set.add(s.site);
  return set.size;
}

export default function HistoryPage() {
  const [log, setLog] = useState(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("recent");
  const [activeId, setActiveId] = useState(null);

  useEffect(() => {
    let cancelled = false;
    getFieldLog().then((l) => { if (!cancelled) setLog(l); });
    return () => { cancelled = true; };
  }, []);

  const all = log?.species ?? [];
  const ghosts = log?.ghosts ?? [];

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = all.filter(
      (sp) =>
        !q ||
        sp.common.toLowerCase().includes(q) ||
        sp.species.toLowerCase().includes(q) ||
        sp.family.toLowerCase().includes(q) ||
        sp.genus.toLowerCase().includes(q),
    );
    return list.slice().sort(SORT_COMPARATORS[sort]);
  }, [query, sort, all]);

  if (!log) return <main className="main" />;

  const active = all.find((sp) => sp.id === activeId) || filtered[0] || all[0];
  const totalSightings = all.reduce((n, sp) => n + sp.sightings.length, 0);
  const totalSites = uniqueSites(all);

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
              Every species you've identified, ranked by confidence and pinned to the dive site
              where you spotted it.
            </p>
          </div>
          <div className="page-header__model">
            <span className="page-header__model-pill">synced · 2 min ago</span>
            <span>local-first · ~62 KB on device</span>
          </div>
        </header>

        <div className="log-stats">
          <StatPill
            label="discovered"
            value={`${all.length} / ${log.totalSpecies.toLocaleString()}`}
            sub={`${((all.length / log.totalSpecies) * 100).toFixed(1)}% of catalogue`}
          />
          <StatPill label="total sightings" value={totalSightings} />
          <StatPill label="dive sites" value={totalSites} sub="6 regions · 4 countries" />
          <StatPill label="photos archived" value={totalSightings} sub="all geo-tagged" />
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
          <div className="log-sort">
            {SORTS.map((s) => (
              <button
                key={s.id}
                className={`log-sort__btn ${sort === s.id ? "log-sort__btn--active" : ""}`}
                onClick={() => setSort(s.id)}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div className="pdx-layout">
          <div className="pdx-grid">
            {filtered.map((sp) => (
              <SpeciesCard
                key={sp.id}
                sp={sp}
                active={sp.id === active?.id}
                onSelect={setActiveId}
              />
            ))}
            {sort === "recent" && !query && ghosts.map((g) => <GhostCard key={g.no} ghost={g} />)}
            {filtered.length === 0 && (
              <div className="pdx-empty">No matches in your log for “{query}”.</div>
            )}
          </div>

          {active && <DetailPanel sp={active} />}
        </div>
      </div>
    </main>
  );
}
