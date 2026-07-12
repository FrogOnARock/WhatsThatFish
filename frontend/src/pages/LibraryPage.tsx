import { useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getSpeciesLibrary } from "../api/client";
import type { SpeciesEntry } from "../api/types";
import StatPill from "../components/history/StatPill";
import LibraryPanel from "../components/library/LibraryPanel";
import { ROUTES } from "../routes";


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
  "imgcount": (a, b) => b.imageCount - a.imageCount
};

export default function SpeciesLibrary() {
  const navigate = useNavigate();
  const { speciesId } = useParams();
  const [entries, setEntries] = useState<SpeciesEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("species");
  const PAGE_SIZE = 20;
  const [page, setPage] = useState(0);

  useEffect(() => {
    let cancelled = false;
    getSpeciesLibrary()
      .then((e) => { if (!cancelled) setEntries(e); })
      .catch((err) => { if (!cancelled) setError(String(err)); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => setPage(0), [query, sort]);

  const all = entries ?? [];

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

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paged = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const totalSpecies = entries.length;
  const totalGenus = new Set(entries.map(entry => entry.genus)).size;
  const totalFamily = new Set(entries.map(entry => entry.family)).size;
  const active = all.find((sp) => `${sp.speciesId}` === speciesId) || filtered[0] || all[0];

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
          <div className="pdx-layout__list">
            {filtered.length === 0 ? (
              <div className="pdx-empty">No matches in the library for “{query}”.</div>
            ) : (
              <table className="pdx-table library-table">
                <thead>
                  <tr>
                    <th>Species</th>
                    <th>Common</th>
                    <th>Genus</th>
                    <th>Family</th>
                    <th className="pdx-table__num">Images</th>
                  </tr>
                </thead>
                <tbody>
                  {paged.map((sp) => (
                    <tr
                      key={sp.speciesId}
                      className={`library-table__row ${
                        sp.speciesId === active?.speciesId ? "library-table__row--active" : ""
                      }`}
                      onClick={() => navigate(`${ROUTES.library}/${sp.speciesId}`)}
                    >
                      <td className="library-table__sci">{sp.name}</td>
                      <td>{sp.common || "—"}</td>
                      <td>{sp.genus}</td>
                      <td>{sp.family}</td>
                      <td className="pdx-table__num">{sp.imageCount}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            <div className="pdx-pager">
              <button disabled={page === 0} onClick={() => setPage(page - 1)}>← Prev</button>
              <span>Page {page + 1} of {pageCount}</span>
              <button disabled={page >= pageCount - 1} onClick={() => setPage(page + 1)}>Next →</button>
            </div>
          </div>

          {active && <LibraryPanel sp={active} />}
        </div>
      </div>
    </main>
  );
}
