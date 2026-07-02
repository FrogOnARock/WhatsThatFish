/* Per-user summary counts for the Settings page (GET /me/stats, auth-gated).
   Distinct from api/stats.ts, which is the model's trained-class counts. */
import { API_BASE } from "./config";
import { authedFetch, raiseForStatus } from "./http";

export interface UserStats {
  dives: number;
  observations: number;
  uniqueSpecies: number;
}

export async function getUserStats(): Promise<UserStats> {
  const res = await authedFetch(`${API_BASE}/me/stats`);
  await raiseForStatus(res, "Couldn't load your stats");
  const raw = await res.json();
  return {
    dives: raw.dives,
    observations: raw.observations,
    uniqueSpecies: raw.unique_species,
  };
}
