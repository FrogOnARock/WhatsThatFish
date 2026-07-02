/* Renders Google's official "Sign in with Google" button into a div once GIS is
   initialised. The sign-in result is handled centrally by AuthContext's callback,
   so this component is purely presentational — no per-button callback wiring. */
import { useEffect, useRef } from "react";
import { useAuth } from "../auth/AuthContext";

export default function GoogleSignInButton() {
  const ref = useRef<HTMLDivElement>(null);
  const { gisReady } = useAuth();

  useEffect(() => {
    if (gisReady && ref.current && window.google) {
      window.google.accounts.id.renderButton(ref.current, {
        theme: "outline",
        size: "large",
        shape: "pill",
        text: "continue_with",
      });
    }
  }, [gisReady]);

  return <div ref={ref} />;
}
