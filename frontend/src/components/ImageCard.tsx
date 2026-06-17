/* ImageCard — the dumb, state-UNAWARE shell: photo viewport + filename bar.
   It knows nothing about RequestState. Anything state-specific (the bbox overlay,
   the analyzing pulse, the bbox Toggle) is INJECTED by the caller through the
   `overlay` and `barAction` slots — so this stays reusable anywhere a
   photo-with-bar is shown (results, history detail, library preview). */
import type { ReactNode } from "react";
import FishPlaceholder from "./FishPlaceholder";
import type { ImageState } from "./ResultsView";

interface ImageCardProps {
  image: ImageState;
  /** Layered over the photo in the viewport: bbox node (success),
      pulse (analyzing), or omitted (no-fish). A falsy value renders nothing. */
  overlay?: ReactNode;
  /** Right side of the bottom bar: the bounding-box Toggle (success only),
      otherwise omitted. */
  barAction?: ReactNode;
}

export default function ImageCard({ image, overlay, barAction }: ImageCardProps) {
  return (
    <div className="image-card">
      <div className="image-card__viewport">
        {image.kind === "sample" ? (
          <FishPlaceholder hue={image.hue} caption={image.caption} large />
        ) : (
          <img src={image.url} alt="uploaded" />
        )}
        {overlay}
      </div>
      <div className="image-card__bar">
        <div className="image-card__bar-left">
          <span className="image-card__filename">{image.filename}</span>
          <span>·</span>
          <span>{image.size}</span>
        </div>
        {barAction}
      </div>
    </div>
  );
}
