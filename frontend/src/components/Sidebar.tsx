/* Sidebar: brand + nav. Active page lifted up to App. */

/** Every page the sidebar can route to (matches App's page state). */
export type PageId =
  | "whats-that-fish"
  | "history"
  | "saved"
  | "library"
  | "settings"
  | "about";

interface NavItem {
  id: PageId;
  label: string;
  group: "main" | "secondary";
  count?: number;
}

const NAV: NavItem[] = [
  { id: "whats-that-fish", label: "What's That Fish?", group: "main" },
  { id: "history",         label: "History",           group: "main", count: 12 },
  { id: "saved",           label: "Saved",             group: "main", count: 6 },
  { id: "library",         label: "Species Library",   group: "main" },
  { id: "settings",        label: "Settings",          group: "secondary" },
  { id: "about",           label: "About",             group: "secondary" },
];

interface SidebarProps {
  active: PageId;
  onNavigate: (id: PageId) => void;
}

export default function Sidebar({ active, onNavigate }: SidebarProps) {
  const main = NAV.filter((n) => n.group === "main");
  const secondary = NAV.filter((n) => n.group === "secondary");

  const renderItem = (item: NavItem) => (
    <button
      key={item.id}
      className={`nav__item ${active === item.id ? "nav__item--active" : ""}`}
      onClick={() => onNavigate(item.id)}
    >
      <span className="nav__dot" />
      <span>{item.label}</span>
      {item.count != null && <span className="nav__count">{item.count}</span>}
    </button>
  );

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <div className="sidebar__brand-mark">WTF · v0.1</div>
        <div className="sidebar__brand-word">
          What's <em>that</em>
          <br />
          Fish?
        </div>
      </div>

      <div className="sidebar__section-label">Workspace</div>
      <nav className="nav">{main.map(renderItem)}</nav>

      <div className="sidebar__section-label">Account</div>
      <nav className="nav">{secondary.map(renderItem)}</nav>

      <div className="sidebar__footer">
        <div className="sidebar__avatar">CM</div>
        <div>
          <div className="sidebar__user-name">Diver</div>
          <div className="sidebar__user-meta">offline · 24 ids</div>
        </div>
      </div>
    </aside>
  );
}
