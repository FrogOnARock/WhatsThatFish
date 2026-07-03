/* Prefetch cache for PUBLIC catalogue images (bare <img src>, no auth).

   photoCache handles AUTH'd photos (token'd fetch → blob URL). These catalogue
   images need no token, so a normal <img> tag loads them fine — the only trick
   is warming the browser's HTTP cache BEFORE the tag mounts, so the image paints
   from disk instead of trickling in over the wire. With the backend's immutable
   Cache-Control, a preloaded image STAYS cached, so flipping to a page we've
   already prefetched is 0 network.

   Primary use is lookahead: while you view page N of the library, we prefetch
   N+1 / N+2 so a "Next" click renders instantly (the image fetch from GCS — not
   the one-shot metadata query — is the slow part). */

const settled = new Set<string>(); // urls whose preload has finished (loaded OR errored)
const inflight = new Map<string, Promise<void>>(); // de-dupe concurrent preloads

// Ceiling for a single preload. An <img> has no AbortController, so during a
// cold start a proxy-held socket would otherwise keep a URL inflight forever and
// hold a page gate closed. Generous enough to outlast a real ~30–60s boot (so we
// don't give up on an image that's about to arrive), but bounded so a truly stuck
// socket eventually settles-as-broken instead of pinning the skeleton.
const PRELOAD_TIMEOUT_MS = 60_000;

function preloadOne(url: string): Promise<void> {
  if (settled.has(url)) return Promise.resolve();
  const pending = inflight.get(url);
  if (pending) return pending;

  const p = new Promise<void>((resolve) => {
    const img = new Image();
    let timer: ReturnType<typeof setTimeout>;
    // Resolve on load, error, OR timeout, and mark settled every way — a single
    // broken/stuck image must never hang a batch or block the page gate (it'll
    // just render as a broken tile). Mirrors photoCache's "always resolves"
    // contract.
    const done = () => {
      clearTimeout(timer);
      settled.add(url);
      inflight.delete(url);
      resolve();
    };
    img.onload = img.onerror = done;
    timer = setTimeout(done, PRELOAD_TIMEOUT_MS);
    img.src = url;
  });
  inflight.set(url, p);
  return p;
}

/** Warm the browser cache for a batch of image URLs concurrently. Always
    resolves — a broken/slow image can't hang the caller. */
export function prefetchImages(urls: string[]): Promise<void> {
  return Promise.all(urls.map(preloadOne)).then(() => undefined);
}

/** True once every url in the batch has finished preloading (i.e. an <img> for
    it will paint from cache). An errored image counts as settled so a bad tile
    never keeps the gate closed forever. */
export function imagesReady(urls: string[]): boolean {
  return urls.every((u) => settled.has(u));
}
