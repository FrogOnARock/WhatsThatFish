/* Account page — the signed-in user's profile + sign out. Falls back to a
   sign-in prompt when signed out (e.g. after a token expiry). */
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import GoogleSignInButton from "../components/GoogleSignInButton";
import { ROUTES } from "../routes";

function initials(name: string | null): string {
  if (!name) return "··";
  return name
    .split(" ")
    .map((p) => p[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export default function AccountPage() {
  const { user, status, signOut } = useAuth();
  const navigate = useNavigate();

  if (status !== "signed-in" || !user) {
    return (
      <main className="main">
        <div className="main__inner">
          <div className="auth-card">
            <h1 className="page-header__title">Your <em>account</em></h1>
            <p className="page-header__subtitle">You're signed out. Sign in to view your profile.</p>
            <div className="auth-card__action">
              <GoogleSignInButton />
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="main">
      <div className="main__inner">
        <header className="page-header">
          <div>
            <div className="page-header__crumb">Workspace · Account</div>
            <h1 className="page-header__title">Your <em>account</em></h1>
          </div>
        </header>

        <div className="account">
          <div className="account__avatar">
            {user.avatarUrl ? (
              <img src={user.avatarUrl} alt={user.displayName ?? "avatar"} referrerPolicy="no-referrer" />
            ) : (
              initials(user.preferredName ?? user.displayName)
            )}
          </div>
          <div className="account__info">
            <div className="account__name">{user.preferredName ?? user.displayName ?? "Diver"}</div>
            {user.email && <div className="account__email">{user.email}</div>}
          </div>
          <div className="account__actions">
            <button
              className="btn btn--ghost btn--sm"
              onClick={() => {
                signOut();
                navigate(ROUTES.identify);
              }}
            >
              Sign out
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}
