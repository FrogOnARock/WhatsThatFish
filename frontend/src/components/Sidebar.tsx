/* Sidebar: brand + nav + account footer. Routing is real now — nav items are
   NavLinks (active state driven by the URL), the footer uses useNavigate. */
import { NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { ROUTES } from "../routes";

function initials(name: string | null): string {
  if (!name) return "··";
  return name.split(" ").map((p) => p[0]).slice(0, 2).join("").toUpperCase();
}

interface NavItem {
  to: string;
  label: string;
  group: "main" | "secondary";
  /** `end` matches the path exactly — needed for "/" so it isn't active everywhere. */
  end?: boolean;
}

const NAV: NavItem[] = [
  { to: ROUTES.identify, label: "What's That Fish?", group: "main", end: true },
  { to: ROUTES.fieldLog, label: "Field Log",         group: "main" },
  { to: ROUTES.dives,    label: "Dive Log",          group: "main" },
  { to: ROUTES.library,  label: "Species Library",   group: "main" },
  { to: ROUTES.settings, label: "Settings",          group: "secondary" },
];

interface SidebarProps {
  /** Mobile drawer state — ignored by the always-visible desktop layout. */
  open?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ open = false, onClose }: SidebarProps) {
  const main = NAV.filter((n) => n.group === "main");
  const secondary = NAV.filter((n) => n.group === "secondary");
  const { user, status } = useAuth();
  const navigate = useNavigate();

  const renderItem = (item: NavItem) => (
    <NavLink
      key={item.to}
      to={item.to}
      end={item.end}
      className={({ isActive }) => `nav__item ${isActive ? "nav__item--active" : ""}`}
      onClick={onClose}
    >
      <span className="nav__dot" />
      <span>{item.label}</span>
    </NavLink>
  );

  const goAccount = () => {
    onClose?.();
    navigate(ROUTES.account);
  };
  const goLogin = () => {
    onClose?.();
    navigate(ROUTES.login);
  };

  return (
    <aside className={`sidebar ${open ? "sidebar--open" : ""}`}>
      <button
        className="sidebar__close"
        onClick={onClose}
        aria-label="Close navigation"
      >
        ×
      </button>
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
          className="sidebar__footer sidebar__footer--button"
          onClick={goAccount}
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
            onClick={goLogin}
            disabled={status === "loading"}
          >
            {status === "loading" ? "…" : "Sign in with Google"}
          </button>
        </div>
      )}
    </aside>
  );
}
