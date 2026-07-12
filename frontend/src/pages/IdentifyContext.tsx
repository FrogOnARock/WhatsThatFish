/* Identify-session cache.

   With a real router, MainPage unmounts whenever the user navigates away (to the
   field log, a dive, etc.). Holding the current image + prediction in MainPage's
   own useState would therefore lose it on every trip. This provider lives ABOVE
   <Routes> (mounted in main.tsx), so the session survives navigation: classify a
   photo, browse the dive log, come back — the image and its result are still there.

   The object URL for an uploaded file is owned here for the provider's lifetime;
   we only revoke it on an explicit reset (new upload / "Try another"), never on
   MainPage unmount — that's the whole point. */
import { createContext, useCallback, useContext, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { ImageState } from "../components/ResultsView";
import type { Prediction } from "../api/types";

/** Mirrors MainPage's old RequestState. Re-exported for consumers. */
export type RequestState =
  | { status: "analyzing" }
  | { status: "success"; prediction: Prediction }
  | { status: "no-fish" }
  | { status: "error"; message: string };

/** The task that produced the current result — kept so "Retry" can re-run the
    exact same inference (sample filename or uploaded file). */
export type InferenceTask = () => Promise<Prediction | null>;

interface IdentifySession {
  image: ImageState | null;
  request: RequestState | null;
  lastTask: InferenceTask | null;
  /** Set the image being identified (does NOT clear a prior object URL — callers
      that replace an uploaded file should reset() first). */
  setImage: (image: ImageState | null) => void;
  setRequest: (request: RequestState | null) => void;
  setLastTask: (task: InferenceTask | null) => void;
  /** Run an inference task, driving request through analyzing → success/no-fish/
      error and remembering it as lastTask (so Retry re-runs the same one). Shared
      by MainPage and the field-log "re-run inference" action. */
  runInference: (task: InferenceTask) => Promise<void>;
  /** Clear the whole session and revoke a file object URL if one is live. */
  reset: () => void;
}

const IdentifyContext = createContext<IdentifySession | null>(null);

export function IdentifyProvider({ children }: { children: ReactNode }) {
  const [image, setImageState] = useState<ImageState | null>(null);
  const [request, setRequest] = useState<RequestState | null>(null);
  const lastTask = useRef<InferenceTask | null>(null);
  const setLastTask = useCallback((task: InferenceTask | null) => {
    lastTask.current = task;
  }, []);

  const setImage = useCallback((next: ImageState | null) => setImageState(next), []);

  const runInference = useCallback(async (task: InferenceTask) => {
    lastTask.current = task;
    setRequest({ status: "analyzing" });
    try {
      const result = await task();
      // A result with no detection still carries whole-frame guesses (detected=false),
      // so it's a success — ResultsView warns. Only a null result is a true no-op.
      if (result === null) setRequest({ status: "no-fish" });
      else setRequest({ status: "success", prediction: result });
    } catch (err) {
      console.error("inference failed", err);
      setRequest({ status: "error", message: `Inference failed: ${err}` });
    }
  }, []);

  const reset = useCallback(() => {
    setImageState((cur) => {
      if (cur?.kind === "file" && cur.url) URL.revokeObjectURL(cur.url);
      return null;
    });
    setRequest(null);
    lastTask.current = null;
  }, []);

  return (
    <IdentifyContext.Provider
      value={{
        image,
        request,
        lastTask: lastTask.current,
        setImage,
        setRequest,
        setLastTask,
        runInference,
        reset,
      }}
    >
      {children}
    </IdentifyContext.Provider>
  );
}

export function useIdentifySession(): IdentifySession {
  const ctx = useContext(IdentifyContext);
  if (!ctx) throw new Error("useIdentifySession must be used within IdentifyProvider");
  return ctx;
}
