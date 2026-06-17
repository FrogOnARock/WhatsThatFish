/* Mock implementation of the API client interface.
   Returns prototype data with fake inference latency so the
   "analyzing" state is exercised exactly like the real thing. */
import {
  MOCK_PREDICTIONS,
  DEFAULT_PREDICTION_KEY,
  SAMPLE_FISH,
} from "./mock-predictions";
import { LOG_SPECIES, LOG_GHOSTS, LOG_TOTAL_SPECIES } from "./mock-log";
import type { FieldLog, Prediction } from "./types";

const FAKE_LATENCY_MS = 1100;

const sleep = (ms: number): Promise<void> =>
  new Promise((r) => setTimeout(r, ms));

/** identify(file) → Prediction — see client.ts for the contract. */
export async function identify(_file: File): Promise<Prediction | null> {
  await sleep(FAKE_LATENCY_MS);
  // Real backend routes on image content; the mock always answers clownfish.
  if (_file.name === "error.jpg") {
    throw new Error("Invalid image provided.")
  } else if (_file.name === "null.jpg") {
    return null
  }
  return MOCK_PREDICTIONS[DEFAULT_PREDICTION_KEY];
}

/** identifySample(id) → Prediction for one of the demo fish. */
export async function identifySample(id: string): Promise<Prediction | null> {
  await sleep(FAKE_LATENCY_MS);
  return MOCK_PREDICTIONS[id] ?? null;
}

/** getFieldLog() → { species, ghosts, totalSpecies } */
export async function getFieldLog(): Promise<FieldLog> {
  return {
    species: LOG_SPECIES,
    ghosts: LOG_GHOSTS,
    totalSpecies: LOG_TOTAL_SPECIES,
  };
}

export { SAMPLE_FISH, DEFAULT_PREDICTION_KEY };
