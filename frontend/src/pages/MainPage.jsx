/* Main page — orchestrates idle / analyzing / results states.
   Unlike the prototype (setTimeout), the analyzing state is driven by the
   API promise resolving, so it behaves identically with the real backend. */
import { useCallback, useState } from "react";
import DropZone from "../components/DropZone.jsx";
import ResultsView from "../components/ResultsView.jsx";
import { identify, identifySample, SAMPLE_FISH } from "../api/client.js";

export default function MainPage() {
  // image: { kind: 'sample' | 'file', filename, size, hue?, caption?, url? }
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  const runInference = useCallback(async (request) => {
    setAnalyzing(true);
    try {
      const result = await request;
      // TODO(you): design the no-detection state. The real pipeline can
      // legitimately return bbox=null (BoundingBoxInference found no fish)
      // or a coral negative — the prototype never designed this. Decide:
      // keep the image card with a "no fish detected" banner where the
      // classification cards would be? Or bounce back to the drop zone
      // with an inline message? `result === null` / `result.bbox === null`
      // are the cases to branch on here.
      setPrediction(result);
    } catch (err) {
      console.error("inference failed", err);
      // TODO(you): error state (backend unreachable / 5xx) — same decision
      // space as no-detection but probably wants a retry affordance.
      setPrediction(null);
      setImage(null);
    } finally {
      setAnalyzing(false);
    }
  }, []);

  const handleSample = useCallback((id) => {
    const sample = SAMPLE_FISH.find((s) => s.id === id);
    setImage({
      kind: "sample",
      filename: `${id}.jpg`,
      size: "2.3 MB",
      hue: sample.hue,
      caption: sample.caption,
    });
    runInference(identifySample(id));
  }, [runInference]);

  const handleUpload = useCallback((file) => {
    const url = URL.createObjectURL(file);
    setImage({
      kind: "file",
      filename: file.name,
      size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
      url,
    });
    runInference(identify(file));
  }, [runInference]);

  const handleReset = useCallback(() => {
    if (image?.kind === "file" && image.url) URL.revokeObjectURL(image.url);
    setImage(null);
    setPrediction(null);
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
              species, genus and subfamily — plus where you're most likely to see it.
            </p>
          </div>
          <div className="page-header__model">
            <span className="page-header__model-pill">model online · YOLO11 + CustomResnet</span>
            <span>1,247 classes · last trained 04 Apr 2026</span>
          </div>
        </header>

        {!image && <DropZone onUpload={handleUpload} onSample={handleSample} />}
        {image && (
          <ResultsView
            image={image}
            prediction={prediction}
            analyzing={analyzing}
            onReset={handleReset}
          />
        )}
      </div>
    </main>
  );
}
