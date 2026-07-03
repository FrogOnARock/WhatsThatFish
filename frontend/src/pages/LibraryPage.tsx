import { useCallback, useEffect, useMemo, useState } from "react";
import { getSpeciesLibrary } from "../api/client";
import type { SpeciesEntry } from "../api/types";
import { API_BASE } from "../api/config";
import { prefetchImages, imagesReady } from "../api/imagePrefetch";
import StatPill from "../components/history/StatPill";
import {LibraryCard} from "../components/library/LibraryCard";
import LibraryPanel from "../components/library/LibraryPanel";


interface SortOption {
  id: string;
  label: string;
}

const SORTS: SortOption[] = [
  {id: "species", label: "By Species"},
  {id: "genus", label: "By Genus"},
  {id: "family", label: "By Family"},
  {id: "imgcount", label: "Image Count"}
]

const SORT_COMPARATORS: Record<string, (a: SpeciesEntry, b: SpeciesEntry) => number> = {
  "species": (a, b) => a.name.localeCompare(b.name),
  "genus": (a, b) => a.genus.localeCompare(b.genus),
  "family": (a, b) => a.family.localeCompare(b.family),
  "imgcount": (a, b) => a.imageCount - b.imageCount
};

// Stable empty reference. `entries ?? []` mints a new array every render while
// loading, which cascades filtered → urlsForPage → the prefetch effect into a
// setState loop (React "max update depth"). A module-level constant keeps the
// identity stable so the memo/effect chain settles.
const EMPTY_ENTRIES: SpeciesEntry[] = [];

export default function SpeciesLibrary() {
  const [entries, setEntries] = useState<SpeciesEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("species");
  const [activeId, setActiveId] = useState<string | null>(null);
  const PAGE_SIZE = 12;
  const [page, setPage] = useState(0);


  useEffect(() => {
    let cancelled = false;
    getSpeciesLibrary()
      .then((e) => { if (!cancelled) setEntries(e); })
      .catch((err) => { if (!cancelled) setError(String(err)); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => setPage(0), [query, sort]);

  const all = entries ?? EMPTY_ENTRIES;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = all.filter(
        (sp) =>
            !q ||
            sp.name.toLowerCase().includes(q) ||
            sp.genus.toLowerCase().includes(q) ||
            sp.family.toLowerCase().includes(q) ||
            sp.common.toLowerCase().includes(q)
    );
    return list.slice().sort(SORT_COMPARATORS[sort]);
  }, [query, sort, all]);

  // Image URLs for a given page of the current filtered/sorted list.
  const urlsForPage = useCallback(
    (p: number) =>
      filtered
        .slice(p * PAGE_SIZE, (p + 1) * PAGE_SIZE)
        .map((sp) => `${API_BASE}/image/${sp.filename}`),
    [filtered],
  );

  // The module-level prefetch cache isn't reactive; bump this to re-evaluate
  // pageReady once a batch settles.
  const [readyTick, setReadyTick] = useState(0);

  useEffect(() => {
    // Gate the VISIBLE page on its own images (so it appears all at once, no
    // trickle), then warm the next TWO pages in the background so a "Next" click
    // renders instantly — the GCS image fetch is the slow part, not the one-shot
    // metadata query, so paying it ahead of the click is the whole win.
    prefetchImages(urlsForPage(page)).then(() => setReadyTick((t) => t + 1));
    prefetchImages([...urlsForPage(page + 1), ...urlsForPage(page + 2)]);
  }, [page, urlsForPage]);

  const pageReady = useMemo(
    () => imagesReady(urlsForPage(page)),
    [urlsForPage, page, readyTick],
  );

  if (error) return (
      <main className="main">
        <div className="main__inner">
          <header className="page-header">
            <div>
              <div className="page-header__crumb">Species Library</div>
              <h1 className="page-header__title">
                Species <em>Library</em>
              </h1>
              <p className="page-header__subtitle">
                All species available in the library and the associated image counts trained on.
              </p>
            </div>
            <div className="page-header__model">
              <span className="page-header__model-pill">synced · 2 min ago</span>
              <span>local-first · ~62 KB on device</span>
            </div>
          </header>
          <div className="error">
            Error: {error} Please retry.
          </div>
        </div>
      </main>);

  if (!entries)
    return (
      <main className="main">
        <div className="main__inner">
          <div className="history-loading">
            <span className="analyzing__pulse" />
            Loading the species library…
          </div>
        </div>
      </main>
    );

  const pageCount = (Math.max(1, Math.ceil(filtered.length / PAGE_SIZE)));
  const paged = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);


  const totalSpecies = entries.length;
  const totalGenus = new Set(entries.map(entry => entry.genus)).size;
  const totalFamily = new Set(entries.map(entry => entry.family)).size;
  const active = all.find((sp) => `${sp.speciesId}` === activeId) || filtered[0] || all[0];

  return (
    <main className="main">
      <div className="main__inner">
        <header className="page-header">
          <div>
            <div className="page-header__crumb">Species Library</div>
            <h1 className="page-header__title">
              Species <em>Library</em>
            </h1>
            <p className="page-header__subtitle">
              All species available in the library and the associated image counts trained on.
            </p>
          </div>
          <div className="page-header__model">
            <span className="page-header__model-pill">synced · 2 min ago</span>
            <span>local-first · ~62 KB on device</span>
          </div>
        </header>

        <div className="log-stats">
          <StatPill label="Total Species" value={totalSpecies} />
          <StatPill label="Total Genus" value={totalGenus} />
          <StatPill label="Total Family" value={totalFamily} />
        </div>

        <div className="log-toolbar">
          <div className="log-search">
            <span className="log-search__icon" aria-hidden>⌕</span>
            <input
              type="text"
              placeholder="Search common name, family, genus…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <div className="log-sort">
            {SORTS.map((s) => (
              <button
                className={`log-sort__btn ${sort === s.id ? "log-sort__btn--active" : ""}`}
                key={s.id}
                onClick={() => setSort(s.id)}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div className="pdx-layout">
          <div className="pdx-grid">
            {filtered.length === 0 ? (
              <div className="pdx-empty">No matches in the library for “{query}”.</div>
            ) : !pageReady ? (
              // Only shows on a page whose images AREN'T already prefetched — i.e.
              // the very first view or a jumped page. Sequential Next/Prev lands on
              // a pre-warmed page and skips this entirely.
              <div className="pdx-empty">
                <span className="analyzing__pulse" /> Loading images…
              </div>
            ) : (
              paged.map((sp) => (
                <LibraryCard key={sp.speciesId} sp={sp} onSelect={setActiveId} />
              ))
            )}
          </div>

          {active && <LibraryPanel sp={active} />}
        </div>

        <div className="pdx-pager">
          <button disabled={page === 0} onClick={() => setPage(page - 1)}>← Prev</button>
          <span>Page {page + 1} of {pageCount}</span>
          <button disabled={page >= pageCount - 1} onClick={() => setPage(page + 1)}>Next →</button>
        </div>

      </div>
    </main>
  );
}