/* Minimal typing for the Google Identity Services client we load at runtime.
   Only the surface we actually call — id.initialize / renderButton / prompt /
   disableAutoSelect. The script is injected by AuthContext. */

interface GoogleCredentialResponse {
  credential: string; // the Google ID token (JWT)
}

interface GoogleIdConfig {
  client_id: string;
  callback: (response: GoogleCredentialResponse) => void;
  auto_select?: boolean;
}

interface GoogleButtonOptions {
  theme?: "outline" | "filled_blue" | "filled_black";
  size?: "small" | "medium" | "large";
  shape?: "rectangular" | "pill" | "circle" | "square";
  text?: "signin_with" | "signup_with" | "continue_with" | "signin";
  width?: number;
}

interface Window {
  google?: {
    accounts: {
      id: {
        initialize: (config: GoogleIdConfig) => void;
        renderButton: (parent: HTMLElement, options: GoogleButtonOptions) => void;
        prompt: () => void;
        disableAutoSelect: () => void;
      };
    };
  };
}
