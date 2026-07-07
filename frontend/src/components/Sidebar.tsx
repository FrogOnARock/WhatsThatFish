/* Sidebar: brand + nav + account footer. Active page lifted up to App. */
import { useAuth } from "../auth/AuthContext";

/** Every page the sidebar can route to (matches App's page state). `login` and
    `account` aren't nav items — they're reached from the footer. */
export type PageId =
  | "whats-that-fish"
  | "history"
  | "dives"
  | "library"
  | "settings"
  | "login"
  | "account";

function initials(name: string | null): string {
  if (!name) return "··";
  return name.split(" ").map((p) => p[0]).slice(0, 2).join("").toUpperCase();
}

interface NavItem {
  id: PageId;
  label: string;
  group: "main" | "secondary";
  count?: number;
}

const NAV: NavItem[] = [
  { id: "whats-that-fish", label: "What's That Fish?", group: "main" },
  { id: "history",         label: "Field Log",         group: "main" },
  { id: "dives",           label: "Dive Log",          group: "main" },
  { id: "library",         label: "Species Library",   group: "main" },
  { id: "settings",        label: "Settings",          group: "secondary" },
];

interface SidebarProps {
  active: PageId;
  onNavigate: (id: PageId) => void;
}

export default function Sidebar({ active, onNavigate }: SidebarProps) {
  const main = NAV.filter((n) => n.group === "main");
  const secondary = NAV.filter((n) => n.group === "secondary");
  const { user, status } = useAuth();

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

      {status === "signed-in" && user ? (
        <button
          className={`sidebar__footer sidebar__footer--button ${active === "account" ? "sidebar__footer--active" : ""}`}
          onClick={() => onNavigate("account")}
        >
          <div className="sidebar__avatar">
            {user.avatarUrl ? (
              <img src={user.avatarUrl} alt="" referrerPolicy="no-referrer" />
            ) : (
              initials(user.preferredName ?? user.displayName)
            )}
          </div>
          <div>
            <div className="sidebar__user-name">
              {user.preferredName ?? user.displayName ?? "Diver"}
            </div>
            <div className="sidebar__user-meta">{user.email ?? "signed in"}</div>
          </div>
        </button>
      ) : (
        <div className="sidebar__footer">
          <button
            className="sidebar__signin"
            onClick={() => onNavigate("login")}
            disabled={status === "loading"}
          >
            {status === "loading" ? "…" : "Sign in with Google"}
          </button>
        </div>
      )}
    </aside>
  );
}
