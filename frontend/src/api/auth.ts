import { API_BASE } from "./config";
import { apiFetch, authedFetch, raiseForStatus, TIMEOUT } from "./http";
import type { UserProfile, UnitSystem } from "./types";

function mapProfile(raw: any): UserProfile {
  return {
    id: raw.id,
    email: raw.email,
    displayName: raw.display_name,
    avatarUrl: raw.avatar_url,
    preferredName: raw.preferred_name ?? null,
    unitSystem: raw.unit_system ?? "metric",
  };
}

/** Fetch the signed-in user's profile, sending the Google ID token as Bearer.
    Throws on a non-2xx (401 = token missing/invalid/expired) so the caller can
    treat it as signed-out. Maps the backend's snake_case to camelCase here, at
    the seam, like api/species.ts does for the catalogue. */
export async function getMe(token: string): Promise<UserProfile> {
  // Pass the token explicitly (this runs during hydration, possibly before
  // authedFetch's authToken() would see it). apiFetch throws AuthExpiredError on
  // 401 — callers already treat any throw here as signed-out.
  const res = await apiFetch(
    `${API_BASE}/auth/me`,
    { headers: { Authorization: `Bearer ${token}` } },
    { timeoutMs: TIMEOUT.META, retries: 1 },
  );
  await raiseForStatus(res, "Couldn't load your profile");
  return mapProfile(await res.json());
}

/** Update the app-owned profile fields. Send only the keys to change (backend
    uses exclude_unset); preferredName='' clears the override. */
export async function updateSettings(body: {
  preferredName?: string;
  unitSystem?: UnitSystem;
}): Promise<UserProfile> {
  const payload: Record<string, unknown> = {};
  if (body.preferredName !== undefined) payload.preferred_name = body.preferredName;
  if (body.unitSystem !== undefined) payload.unit_system = body.unitSystem;
  const res = await authedFetch(`${API_BASE}/auth/me`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await raiseForStatus(res, "Couldn't update your settings");
  return mapProfile(await res.json());
}
