/* Sticky right-hand panel: hero, taxonomy crumb, stats, captures, sightings log. */
import FishPlaceholder from "../FishPlaceholder";
import StatPill from "./StatPill";
import type { LogSpecies } from "../../api/types";

export function fmtDate(iso: string): string {
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString("en-US", { day: "2-digit", month: "short", year: "numeric" });
}

interface DetailPanelProps {
  sp: LogSpecies;
}

export default function DetailPanel({ sp }: DetailPanelProps) {
  const sightings = sp.sightings;
  const total = sightings.length;
  const sites = new Set(sightings.map((s) => s.site)).size;
  const regions = new Set(sightings.map((s) => s.region)).size;
  const minDepth = Math.min(...sightings.map((s) => s.depth));
  const maxDepth = Math.max(...sightings.map((s) => s.depth));
  const byDate = [...sightings].sort((a, b) => a.date.localeCompare(b.date));
  const first = byDate[0];
  const last = byDate[byDate.length - 1];

  return (
    <aside className="pdx-detail">
      <div className="pdx-detail__hero">
        <FishPlaceholder hue={sp.hue} caption={sp.caption} large />
        <div className="pdx-detail__hero-no">№ {String(sp.no).padStart(3, "0")}</div>
      </div>

      <div className="pdx-detail__title-block">
        <div className="pdx-detail__crumb">
          <span>{sp.family}</span>
          <span className="pdx-detail__crumb-sep">›</span>
          <span>{sp.genus}</span>
          <span className="pdx-detail__crumb-sep">›</span>
          <span className="pdx-detail__crumb-sp">{sp.species.split(" ")[1]}</span>
        </div>
        <h2 className="pdx-detail__common">{sp.common}</h2>
        <div className="pdx-detail__sci">{sp.species}</div>
      </div>

      <p className="pdx-detail__habitat">{sp.habitat}</p>

      <div className="pdx-detail__stats">
        <StatPill label="sightings" value={total} />
        <StatPill label="dive sites" value={sites} sub={`${regions} regions`} />
        <StatPill label="depth" value={`${minDepth}–${maxDepth} m`} />
        <StatPill label="best conf." value={`${sp.bestConf.toFixed(1)}%`} />
      </div>

      <div className="pdx-detail__section">
        <div className="pdx-detail__section-head">
          <h4>Captures</h4>
          <span className="pdx-detail__section-meta">
            {total} photo{total === 1 ? "" : "s"}
          </span>
        </div>
        <div className="pdx-gallery">
          {sightings.map((s, i) => (
            <div key={i} className="pdx-gallery__cell">
              <FishPlaceholder
                hue={sp.hue + (i * 7) % 30}
                caption={fmtDate(s.date).split(" ")[0]}
              />
            </div>
          ))}
        </div>
      </div>

      <div className="pdx-detail__section">
        <div className="pdx-detail__section-head">
          <h4>Sightings log</h4>
          <span className="pdx-detail__section-meta">
            first {fmtDate(first.date)} · last {fmtDate(last.date)}
          </span>
        </div>
        <table className="pdx-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Site</th>
              <th className="pdx-table__num">Depth</th>
              <th className="pdx-table__num">Temp</th>
              <th className="pdx-table__num">Conf.</th>
            </tr>
          </thead>
          <tbody>
            {[...sightings].sort((a, b) => b.date.localeCompare(a.date)).map((s, i) => (
              <tr key={i}>
                <td className="pdx-table__date">{fmtDate(s.date)}</td>
                <td>
                  <div className="pdx-table__site">{s.site}</div>
                  <div className="pdx-table__region">{s.region}</div>
                </td>
                <td className="pdx-table__num">{s.depth} m</td>
                <td className="pdx-table__num">{s.tempC}°C</td>
                <td className="pdx-table__num pdx-table__conf">{s.conf.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </aside>
  );
}
