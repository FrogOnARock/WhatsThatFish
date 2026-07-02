import type { Prediction } from "./types";
import { API_BASE } from "./config";
// Override at build/deploy time with VITE_API_BASE (e.g. the Cloud Run URL).
// Defaults to the local uvicorn dev server.

export async function getPrediction(file: File): Promise<Prediction> {
      const body = new FormData();
      body.append("img", file);
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body });
      if (!res.ok) {
        throw new Error(`POST /prediction failed: ${res.status} ${res.statusText}`);
      }
      return await res.json();
}

export async function getPredictionSample(file: string): Promise<Prediction> {
    const res = await fetch(`${API_BASE}/predict/sample/${file}`);
    if (!res.ok) {
        throw new Error(`GET /prediction failed: ${res.status} ${res.statusText}`);
    }
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