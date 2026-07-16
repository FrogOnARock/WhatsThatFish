
export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

// Google OAuth Web Client ID — same value as the backend's GOOGLE_OAUTH_CLIENT_ID.
// Set VITE_GOOGLE_CLIENT_ID in the frontend env (.env.local).
export const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID ?? "";

// Google Maps JS API key (Places Autocomplete for dive sites). Requires billing
// enabled on the GCP project. Unset → SiteAutocomplete falls back to plain
// free-text + backend suggestions. Set VITE_GOOGLE_MAPS_API_KEY in .env.local.
export const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY ?? "";
