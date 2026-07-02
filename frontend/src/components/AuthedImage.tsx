/* Renders a contribution photo that lives behind the auth-gated image endpoint.
   <img src> can't attach the Bearer token, so the bytes are fetched (with the
   header) into a blob object URL via the shared photoCache. When the field log
   prefetches, the URL is already cached and the image paints immediately; on a
   cache miss (e.g. a one-off lightbox) it loads on demand. The cache owns the
   object URLs for the session, so we don't revoke here. */
import { useEffect, useState } from "react";
import { getCachedPhoto, loadPhoto } from "../api/photoCache";

export default function AuthedImage({
  photoId,
  alt,
  className,
}: {
  photoId: string;
  alt?: string;
  className?: string;
}) {
  const [url, setUrl] = useState<string | null>(() => getCachedPhoto(photoId) ?? null);

  useEffect(() => {
    const hit = getCachedPhoto(photoId);
    if (hit) {
      setUrl(hit);
      return;
    }
    let cancelled = false;
    loadPhoto(photoId).then((u) => {
      if (!cancelled) setUrl(u);
    });
    return () => {
      cancelled = true;
    };
  }, [photoId]);

  if (!url) return <div className={className} aria-busy="true" />;
  return <img src={url} alt={alt ?? ""} className={className} />;
}
