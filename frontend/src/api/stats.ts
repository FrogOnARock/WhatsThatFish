/* Trained-class counts from GET /stats — replaces hardcoded UI numbers.
   Wire shape is already flat ints with matching names, so no mapping needed. */
import { API_BASE } from "./config";
import { apiFetch, raiseForStatus, TIMEOUT } from "./http";

export interface ModelStats {
  species: number;
  genera: number;
  families: number;
}

export async function getStats(): Promise<ModelStats> {
  const res = await apiFetch(`${API_BASE}/stats`, {}, { timeoutMs: TIMEOUT.META, retries: 1 });
  await raiseForStatus(res, "Couldn't load model stats");
  return res.json();
}
