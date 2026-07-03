import { API_BASE } from "./config";
import { apiFetch, raiseForStatus, TIMEOUT } from "./http";
import type {Prediction} from "./types";

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
