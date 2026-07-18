/* Free-text dive-site input with suggestions from two sources:
   1. Google Places (when VITE_GOOGLE_MAPS_API_KEY is set) via the NEW
      AutocompleteSuggestion data API — available to new Maps customers, unlike
      the deprecated Autocomplete widget. Picking a place captures place_id +
      lat/lng via place.fetchFields (surfaced through onPlace).
   2. Backend existing-site suggestions (when no key, or Google errors) — so
      users reuse a logged site instead of a near-duplicate.
   The input stays fully controlled (value/onChange), so pre-filled edit values
   work; suggestions render in our own dropdown either way. */
import { useEffect, useRef, useState } from "react";
import { searchSites } from "../api/observations";
import { GOOGLE_MAPS_API_KEY } from "../api/config";

export interface PlacePick {
  placeId: string | null;
  lat: number | null;
  lng: number | null;
}

interface Props {
  value: string;
  onChange: (v: string) => void;
  onPlace?: (p: PlacePick) => void;
  placeholder?: string;
  className?: string;
}

// One Google suggestion (carries the prediction so we can fetch coords on pick)
// or a plain backend site (prediction undefined).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Suggestion = { id: string; name: string; prediction?: any };

/* Defines google.maps.importLibrary itself (Google's official bootstrap),
   rather than assuming it exists — a raw <script> append can't add it to a
   stale `google.maps` left by Vite HMR, which is what "importLibrary is not a
   function" means. Idempotent. */
function bootstrapMaps(key: string): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  const google = (w.google ??= {});
  const maps = (google.maps ??= {});
  if (maps.importLibrary) return; // real loader (or a prior bootstrap) present
  const CB = "__ib__";
  let loadPromise: Promise<void> | null = null;
  const libs = new Set<string>();
  const load = (): Promise<void> => {
    if (loadPromise) return loadPromise;
    loadPromise = new Promise<void>((resolve, reject) => {
      const params = new URLSearchParams({
        key,
        v: "weekly",
        loading: "async",
        libraries: [...libs].join(","),
        callback: "google.maps." + CB,
      });
      const s = document.createElement("script");
      s.src = "https://maps.googleapis.com/maps/api/js?" + params.toString();
      s.async = true;
      maps[CB] = () => resolve();
      s.onerror = () => reject(new Error("Google Maps failed to load"));
      document.head.append(s);
    });
    return loadPromise;
  };
  // Shim: once the script loads it REPLACES this with the real importLibrary,
  // so the .then() calls the real one (no recursion).
  maps.importLibrary = (name: string, ...rest: unknown[]) => {
    libs.add(name);
    return load().then(() =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (w.google.maps.importLibrary as any)(name, ...rest),
    );
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let placesPromise: Promise<any> | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function loadPlaces(): Promise<any> {
  if (!GOOGLE_MAPS_API_KEY) return Promise.reject(new Error("no maps key"));
  if (placesPromise) return placesPromise;
  bootstrapMaps(GOOGLE_MAPS_API_KEY);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const p = (window as any).google.maps.importLibrary("places");
  placesPromise = p;
  return p;
}

export default function SiteAutocomplete({
  value,
  onChange,
  onPlace,
  placeholder,
  className,
}: Props) {
  const [results, setResults] = useState<Suggestion[]>([]);
  const [open, setOpen] = useState(false);
  // True right after a pick, so we don't immediately re-search (and re-open)
  // for the value we just set.
  const justPicked = useRef(false);
  // Google Autocomplete session token — one session spans the keystrokes up to
  // a details fetch, which bills as a single session. Cleared after a pick.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tokenRef = useRef<any>(null);

  useEffect(() => {
    if (justPicked.current) {
      justPicked.current = false;
      return;
    }
    const q = value.trim();
    if (q.length < 1) {
      setResults([]);
      setOpen(false);
      return;
    }
    let cancelled = false;

    (async () => {
      // ── Google Places (new data API) ─────────────────────────────────────
      if (GOOGLE_MAPS_API_KEY) {
        try {
          const places = await loadPlaces();
          if (cancelled) return;
          if (!tokenRef.current) {
            tokenRef.current = new places.AutocompleteSessionToken();
          }
          const { suggestions } =
            await places.AutocompleteSuggestion.fetchAutocompleteSuggestions({
              input: q,
              sessionToken: tokenRef.current,
            });
          if (cancelled) return;
          const opts: Suggestion[] = suggestions
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            .filter((s: any) => s.placePrediction)
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            .map((s: any) => ({
              id: s.placePrediction.placeId,
              name: s.placePrediction.text.text,
              prediction: s.placePrediction,
            }));
          setResults(opts);
          setOpen(opts.length > 0);
          return;
        } catch (e) {
          // Surface WHY Google failed (API-not-enabled, referer/key restriction,
          // billing, quota) instead of silently degrading — then fall through to
          // backend suggestions so the field still works.
          console.warn(
            "[SiteAutocomplete] Google Places lookup failed; using backend " +
              "suggestions. Check that 'Places API (New)' is enabled, the key " +
              "allows this referrer, and billing is on. Error:",
            e,
          );
        }
      }

      // ── backend existing-site suggestions ────────────────────────────────
      try {
        const r = await searchSites(q);
        if (cancelled) return;
        setResults(r.map((s) => ({ id: s.id, name: s.name })));
        setOpen(r.length > 0);
      } catch {
        /* ignore */
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [value]);

  async function pick(opt: Suggestion) {
    justPicked.current = true;
    onChange(opt.name);
    setOpen(false);
    if (opt.prediction) {
      try {
        const place = opt.prediction.toPlace();
        await place.fetchFields({ fields: ["location", "id", "displayName"] });
        onPlace?.({
          placeId: place.id ?? null,
          lat: place.location?.lat() ?? null,
          lng: place.location?.lng() ?? null,
        });
      } catch (e) {
        console.warn("[SiteAutocomplete] place details (coords) fetch failed:", e);
        onPlace?.({ placeId: opt.id ?? null, lat: null, lng: null });
      }
      tokenRef.current = null; // session consumed by the details fetch
    } else {
      onPlace?.({ placeId: null, lat: null, lng: null });
    }
  }

  // Hide an exact-name match: no point suggesting the value already typed.
  const suggestions = results.filter(
    (s) => s.name.toLowerCase() !== value.trim().toLowerCase(),
  );

  return (
    <div className={`site-ac ${className ?? ""}`}>
      <input
        className="modal__input"
        placeholder={placeholder ?? "Dive site"}
        value={value}
        onChange={(e) => {
          // Manual typing invalidates any previously-picked place coords.
          onPlace?.({ placeId: null, lat: null, lng: null });
          onChange(e.target.value);
        }}
        onFocus={() => suggestions.length > 0 && setOpen(true)}
        // Delay so a suggestion's onClick fires before the list unmounts.
        onBlur={() => setTimeout(() => setOpen(false), 120)}
      />
      {open && suggestions.length > 0 && (
        <ul className="site-ac__list">
          {suggestions.map((s) => (
            <li key={s.id}>
              <button
                type="button"
                className="site-ac__item"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(s)}
              >
                {s.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
