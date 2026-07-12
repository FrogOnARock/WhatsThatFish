/* Dive Log — tabular view of the signed-in user's dives, sortable by date,
   site, or sighting count. Each row opens an expanded detail popup; from there
   (or the row's Edit button) the dive's details can be PATCHed. */
import { useCallback, useEffect, useMemo, useState } from "react";
import { listDives, deleteDive, type Dive } from "../api/observations";
import DiveDetailModal from "../components/DiveDetailModal";
import DiveEditModal from "../components/DiveEditModal";
import ConfirmModal from "../components/ConfirmModal";
import StatPill from "../components/history/StatPill";
import { useAuth } from "../auth/AuthContext";

type SortKey = "date" | "site" | "count";
type SortDir = "asc" | "desc";

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

export default function DivesPage() {
  const { status } = useAuth();
  const [dives, setDives] = useState<Dive[] | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [detail, setDetail] = useState<Dive | null>(null);
  const [editing, setEditing] = useState<Dive | null>(null);
  const [deleting, setDeleting] = useState<Dive | null>(null);

  useEffect(() => {
    if (status !== "signed-in") {
      setDives(null);
      return;
    }
    let cancelled = false;
    listDives()
      .then((d) => !cancelled && setDives(d))
      .catch(() => !cancelled && setDives([]));
    return () => {
      cancelled = true;
    };
  }, [status]);

  const reload = useCallback(() => {
    listDives()
      .then((d) => {
        setDives(d);
        // Keep the open popup in sync with the freshly-saved data.
        setDetail((cur) => (cur ? d.find((x) => x.id === cur.id) ?? null : null));
      })
      .catch(() => {});
  }, []);

  // Clicking the active column toggles direction; a new column resets to its
  // natural default (newest dates / largest counts first, sites A→Z).
  const onSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(key === "site" ? "asc" : "desc");
    }
  };

  const sorted = useMemo(() => {
    const arr = [...(dives ?? [])];
    arr.sort((a, b) => {
      let cmp = 0;
      if (sortKey === "date") cmp = (a.divedAt ?? "").localeCompare(b.divedAt ?? "");
      else if (sortKey === "site")
        cmp = (a.siteName ?? "").localeCompare(b.siteName ?? "");
      else cmp = a.observationCount - b.observationCount;
      return sortDir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [dives, sortKey, sortDir]);

  if (status === "loading") return <main className="main" />;

  if (status !== "signed-in") {
    return (
      <main className="main">
        <div className="main__inner">
          <div className="auth-card">
            <h1 className="page-header__title">
              Your <em>dive</em> log
            </h1>
            <p className="page-header__subtitle">
              Sign in to view the dives you've logged.
            </p>
          </div>
        </div>
      </main>
    );
  }

  if (!dives)
    return (
      <main className="main">
        <div className="main__inner">
          <div className="history-loading">
            <span className="analyzing__pulse" />
            Loading your dive log…
          </div>
        </div>
      </main>
    );

  const arrow = (key: SortKey) =>
    key === sortKey ? (sortDir === "asc" ? " ▲" : " ▼") : "";
  const totalSightings = dives.reduce((n, d) => n + d.observationCount, 0);

  return (
    <main className="main">
      <div className="main__inner">
        <header className="page-header">
          <div>
            <div className="page-header__crumb">Workspace · Dive Log</div>
            <h1 className="page-header__title">
              Your <em>dive</em> log
            </h1>
            <p className="page-header__subtitle">
              Every dive you've logged. Click a row to see its details and the
              species you spotted.
            </p>
          </div>
        </header>

        <div className="log-stats">
          <StatPill label="dives" value={dives.length} />
          <StatPill label="total sightings" value={totalSightings} />
        </div>

        {dives.length === 0 ? (
          <div className="pdx-empty">
            No dives logged yet — save an identification to start your dive log.
          </div>
        ) : (
          <table className="pdx-table dive-table">
            <thead>
              <tr>
                <th>
                  <button className="dive-table__sort" onClick={() => onSort("date")}>
                    Date{arrow("date")}
                  </button>
                </th>
                <th>
                  <button className="dive-table__sort" onClick={() => onSort("site")}>
                    Site{arrow("site")}
                  </button>
                </th>
                <th className="pdx-table__num">
                  <button className="dive-table__sort" onClick={() => onSort("count")}>
                    Sightings{arrow("count")}
                  </button>
                </th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((d) => (
                <tr key={d.id} className="dive-table__row" onClick={() => setDetail(d)}>
                  <td className="pdx-table__date">{fmtDate(d.divedAt)}</td>
                  <td>
                    <div className="pdx-table__site">{d.siteName ?? "Untitled dive"}</div>
                  </td>
                  <td className="pdx-table__num">{d.observationCount}</td>
                  <td className="pdx-table__num">
                    <button
                      className="pdx-table__edit"
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditing(d);
                      }}
                    >
                      Edit
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {detail && (
        <DiveDetailModal
          dive={detail}
          onClose={() => setDetail(null)}
          onEdit={() => {
            setEditing(detail);
            setDetail(null);
          }}
          onDelete={() => {
            setDeleting(detail);
            setDetail(null);
          }}
        />
      )}

      {editing && (
        <DiveEditModal
          dive={editing}
          onClose={() => setEditing(null)}
          onSaved={reload}
        />
      )}

      {deleting && (
        <ConfirmModal
          title="Delete dive?"
          body="This permanently removes the dive and every sighting and photo logged on it, including the image files. This can't be undone."
          onConfirm={async () => {
            await deleteDive(deleting.id);
            reload();
          }}
          onClose={() => setDeleting(null)}
        />
      )}
    </main>
  );
}
