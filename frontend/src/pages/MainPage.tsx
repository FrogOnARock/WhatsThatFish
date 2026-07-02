/* Main page — orchestrates idle / analyzing / results states.
   Unlike the prototype (setTimeout), the analyzing state is driven by the
   API promise resolving, so it behaves identically with the real backend. */
import { useCallback, useEffect, useState, useRef } from "react";
import DropZone from "../components/DropZone";
import ResultsView, { type ImageState } from "../components/ResultsView";
import { getPrediction, getPredictionSample } from "../api/client";
import type { Prediction } from "../api/types";
import { SAMPLE_FISH } from "../api/prediction";
import { API_BASE } from "../api/config";
import { getStats, type ModelStats } from "../api/stats";

export type RequestState =
    { status: "analyzing" } |
    { status: "success"; prediction: Prediction } |
    { status: "no-fish" } |
    { status: "error"; message: string }


export default function MainPage() {
  const lastTask = useRef<(() => Promise<Prediction | null>) | null>(null);
  const [image, setImage] = useState<ImageState | null>(null);
  const [request, setRequest] = useState<RequestState | null>(null);
  const [stats, setStats] = useState<ModelStats | null>(null);

  useEffect(() => {
    getStats().then(setStats).catch(() => {});
  }, []);

  const  runInference = useCallback(async (task: () => Promise<Prediction | null>) => {
    lastTask.current = task;
    setRequest({ status: "analyzing" });
    try {
      const result = await task();
      // A result with no detection still carries whole-frame guesses (detected=false),
      // so it's a success — ResultsView warns. Only a null result is a true no-op.
      if (result === null) {
        setRequest({status: "no-fish"})
      } else setRequest({status: "success", prediction: result});
    }
    catch (err) {
      console.error("inference failed", err);
      setRequest({ status: "error", message: `Inference failed: ${err}` });
    }
  }, []);

  const handleSample = useCallback((id: string) => {
    const sample = SAMPLE_FISH.find((s) => s.id === id);
    if (!sample) return;
    setImage({
      kind: "sample",
      filename: sample.filename,
      size: "2.3 MB",
      caption: sample.label,
      // Serve the real sample frame from the backend so the bbox overlays the
      // actual fish, not the abstract placeholder.
      url: `${API_BASE}/image/${sample.filename}`,
    });
    runInference(() => getPredictionSample(sample.filename));
  }, [runInference]);

  const handleUpload = useCallback((file: File) => {
    const url = URL.createObjectURL(file);
    setImage({
      kind: "file",
      filename: file.name,
      size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      url,
    });
    runInference(() => getPrediction(file));
  }, [runInference]);

  const handleRetry = useCallback(() => {
    if (lastTask.current) runInference(lastTask.current);},
    [runInference]);

  const handleReset = useCallback(() => {
    if (image?.kind === "file" && image.url) URL.revokeObjectURL(image.url);
    setImage(null);
    setRequest(null);
  }, [image]);

  return (
    <main className="main">
      <div className="main__inner">
        <header className="page-header">
          <div>
            <div className="page-header__crumb">Workspace · Identify</div>
            <h1 className="page-header__title">
              What's <em>that</em> fish?
            </h1>
            <p className="page-header__subtitle">
              Drop a dive photo and the model will return its three best guesses for
              species, genus and family — plus where you're most likely to see it.
            </p>
          </div>
          <div className="page-header__model">
            <span className="page-header__model-pill">model online · YOLO11 + CustomResnet</span>
            <span>
              {stats
                ? `${stats.species.toLocaleString()} species · ${stats.genera} genera · ${stats.families} families`
                : "loading classes…"}
            </span>
          </div>
        </header>

        {!image && <DropZone onUpload={handleUpload} onSample={handleSample} speciesCount={stats?.species} />}
        {image && (
          <ResultsView
            image={image}
            request={request}
            onReset={handleReset}
            onRetry={handleRetry}
          />
        )}
      </div>
    </main>
  );
}
