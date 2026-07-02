import { API_BASE } from "./config";
import type {Prediction} from "./types";

export async function getPrediction(file: File): Promise<Prediction> {
      const body = new FormData();
      body.append("img", file);
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body });
      if (!res.ok) {
        throw new Error(`POST /prediction failed: ${res.status} ${res.statusText}`);
      }
      return await res.json();
}
