/* Session-scoped cache of contribution-photo blob URLs.

   Photos sit behind the auth-gated image endpoint, so each one is a token'd
   fetch → blob → object URL. The field log shows many at once and they trickle
   in one-by-one, which looks janky. This cache lets us PREFETCH every photo up
   front (HistoryPage awaits it before rendering) and then serve each <img>
   instantly from memory.

   Object URLs are intentionally NOT revoked per-component — they live for the
   session so re-opening a card / lightbox is free. The set is bounded by how
   many photos a single user has logged, which is small. */
import { authedFetch } from "./http";
import { photoImageEndpoint } from "./history";

const cache = new Map<string, string>(); // photoId → object URL
const inflight = new Map<string, Promise<string | null>>(); // de-dupe concurrent loads

export function getCachedPhoto(id: string): string | undefined {
  return cache.get(id);
}

/** Fetch a photo once; subsequent calls (and concurrent ones) reuse the result.
    Resolves to null on failure so a single broken photo never blocks the rest. */
export function loadPhoto(id: string): Promise<string | null> {
  const hit = cache.get(id);
  if (hit) return Promise.resolve(hit);

  const pending = inflight.get(id);
  if (pending) return pending;

  const p = authedFetch(photoImageEndpoint(id))
    .then((r) => (r.ok ? r.blob() : Promise.reject(r.status)))
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      cache.set(id, url);
      inflight.delete(id);
      return url;
    })
    .catch(() => {
      inflight.delete(id);
      return null;
    });

  inflight.set(id, p);
  return p;
}

/** Warm the cache for a batch of photos. Always resolves (failures are absorbed
    in loadPhoto), so awaiting it can gate a render without risk of hanging on a
    single bad photo. */
export function prefetchPhotos(ids: string[]): Promise<void> {
  return Promise.all(ids.map(loadPhoto)).then(() => undefined);
}
