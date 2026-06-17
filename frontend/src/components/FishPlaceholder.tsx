/* Stripey aquatic placeholder, tinted by hue so each sample reads distinct.
   Stands in for real fish photos until samples are hosted. */
import type { CSSProperties } from "react";

interface FishPlaceholderProps {
  hue?: number;
  caption?: string;
  large?: boolean;
}

export default function FishPlaceholder({
  hue = 200,
  caption,
  large = false,
}: FishPlaceholderProps) {
  // Custom CSS properties aren't part of CSSProperties' known keys, so cast.
  const style = {
    "--ph-h": hue,
    "--ph-l": large ? 0.82 : 0.86,
    "--ph-l-2": large ? 0.9 : 0.92,
  } as CSSProperties;
  return (
    <div className="placeholder" style={style}>
      <span className="placeholder__caption">{caption}</span>
    </div>
  );
}
