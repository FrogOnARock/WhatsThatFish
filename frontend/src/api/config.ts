
export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8080";

// Google OAuth Web Client ID — same value as the backend's GOOGLE_OAUTH_CLIENT_ID.
// Set VITE_GOOGLE_CLIENT_ID in the frontend env (.env.local).
export const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? "";
