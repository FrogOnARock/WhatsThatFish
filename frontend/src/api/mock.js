/* Mock implementation of the API client interface.
   Returns prototype data with fake inference latency so the
   "analyzing" state is exercised exactly like the real thing. */
import {
  MOCK_PREDICTIONS,
  DEFAULT_PREDICTION_KEY,
  SAMPLE_FISH,
} from "./mock-predictions.js";
import { LOG_SPECIES, LOG_GHOSTS, LOG_TOTAL_SPECIES } from "./mock-log.js";

const FAKE_LATENCY_MS = 1100;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** identify(file) → Prediction — see client.js for the contract. */
export async function identify(_file) {
  await sleep(FAKE_LATENCY_MS);
  // Real backend routes on image content; the mock always answers clownfish.
  return MOCK_PREDICTIONS[DEFAULT_PREDICTION_KEY];
}

/** identifySample(id) → Prediction for one of the demo fish. */
export async function identifySample(id) {
  await sleep(FAKE_LATENCY_MS);
  return MOCK_PREDICTIONS[id] ?? null;
}

/** getFieldLog() → { species, ghosts, totalSpecies } */
export async function getFieldLog() {
  return {
    species: LOG_SPECIES,
    ghosts: LOG_GHOSTS,
    totalSpecies: LOG_TOTAL_SPECIES,
  };
}

export { SAMPLE_FISH, DEFAULT_PREDICTION_KEY };
