/* One taxonomy head's top-3: emphasised top row + two minor rows. */
export default function ClassificationCard({ taxon, predictions, common }) {
  const [top, ...rest] = predictions;
  return (
    <div className="cls-card">
      <div className="cls-card__head">
        <span className="cls-card__rank">{taxon.label}</span>
        <span className="cls-card__hint">{taxon.hint}</span>
      </div>
      <div className="cls-card__rows">
        <div className="cls-row cls-row--top">
          <span className="cls-row__rank">1</span>
          <div>
            <div className="cls-row__name">{top.name}</div>
            {taxon.key === "species" && common && (
              <div className="cls-row__common">{common}</div>
            )}
          </div>
          <span className="cls-row__conf">{top.conf.toFixed(1)}%</span>
        </div>
        {rest.map((p, i) => (
          <div key={p.name} className="cls-row cls-row--minor">
            <span className="cls-row__rank">{i + 2}</span>
            <div className="cls-row__name">{p.name}</div>
            <span className="cls-row__conf">{p.conf.toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
