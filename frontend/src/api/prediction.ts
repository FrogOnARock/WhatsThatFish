import type { Prediction } from "./types";
import { API_BASE } from "./config";
import { apiFetch, raiseForStatus, TIMEOUT } from "./http";
// Override at build/deploy time with VITE_API_BASE (e.g. the Cloud Run URL).
// Defaults to the local uvicorn dev server.
//
// Inference gets the long PREDICT timeout (a cold instance loading ONNX + a real
// forward pass is slow) and one retry — the FormData body is reusable across the
// retry since it's not a consumed stream.

export async function getPrediction(file: File): Promise<Prediction> {
      const body = new FormData();
      body.append("img", file);
      const res = await apiFetch(
        `${API_BASE}/predict`,
        { method: "POST", body },
        { timeoutMs: TIMEOUT.PREDICT, retries: 1 },
      );
      await raiseForStatus(res, "Inference failed");
      return await res.json();
}

export async function getPredictionSample(file: string): Promise<Prediction> {
    const res = await apiFetch(
        `${API_BASE}/predict/sample/${file}`,
        {},
        { timeoutMs: TIMEOUT.PREDICT, retries: 1 },
    );
    await raiseForStatus(res, "Inference failed");
    return await res.json();
}


export interface SAMPLE {
  id: string,
  filename: string,
  label: string
}

export const DEFAULT_PREDICTION_KEY = "1"

export const SAMPLE_FISH: SAMPLE[] = [
    {
      id: "1",
      filename: "602438984.jpg",
      label: "Melon Butterflyfish"
    },
    {
      id: '2',
      filename: "329908445.jpeg",
      label: "New Guinea Wrasse"
    },
    {
      id: "3",
      filename: "32518515.jpg",
      label: "Anchor Tuskfish"
    },
    {
      id: "4",
      filename: "629269323.jpg",
      label: "Checkered Snapper"
    },
    {
      id: "5",
      filename: "399338408.jpg",
      label: "Camouflage Grouper"
    }
    ]