/* Stripey aquatic placeholder, tinted by hue so each sample reads distinct.
   Stands in for real fish photos until samples are hosted. */
export default function FishPlaceholder({ hue = 200, caption, large = false }) {
  const style = {
    "--ph-h": hue,
    "--ph-l": large ? 0.82 : 0.86,
    "--ph-l-2": large ? 0.9 : 0.92,
  };
  return (
    <div className="placeholder" style={style}>
      <span className="placeholder__caption">{caption}</span>
    </div>
  );
}
