/* Login page — branding + the Google sign-in button. On successful sign-in
   (status flips to signed-in via AuthContext) we route to the account page. */
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import GoogleSignInButton from "../components/GoogleSignInButton";
import { useAuth } from "../auth/AuthContext";
import { ROUTES } from "../routes";

export default function LoginPage() {
  const { status } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (status === "signed-in") navigate(ROUTES.account, { replace: true });
  }, [status, navigate]);

  return (
    <main className="main">
      <div className="main__inner">
        <div className="auth-card">
          <div className="page-header__crumb">Workspace · Account</div>
          <h1 className="page-header__title">
            Sign <em>in</em>
          </h1>
          <p className="page-header__subtitle">
            Sign in with Google to save identifications to your field log, pin dive
            sites, and sync your history across devices.
          </p>
          <div className="auth-card__action">
            <GoogleSignInButton />
          </div>
        </div>
      </div>
    </main>
  );
}
