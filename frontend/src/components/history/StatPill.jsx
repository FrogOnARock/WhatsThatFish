export default function StatPill({ label, value, sub }) {
  return (
    <div className="stat-pill">
      <div className="stat-pill__value">{value}</div>
      <div className="stat-pill__meta">
        <span className="stat-pill__label">{label}</span>
        {sub && <span className="stat-pill__sub">{sub}</span>}
      </div>
    </div>
  );
}
