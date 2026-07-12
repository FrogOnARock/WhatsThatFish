/* Shared "Search on Wikipedia" link-out — used by the species library panel and
   the post-inference description card so the control is identical in both. */
import { wikipediaSearchUrl } from "../lib/wikipedia";

export default function WikipediaButton({
  name,
  className,
}: {
  name: string;
  className?: string;
}) {
  return (
    <a
      className={`btn btn--ghost btn--sm wiki-btn ${className ?? ""}`}
      href={wikipediaSearchUrl(name)}
      target="_blank"
      rel="noopener noreferrer"
    >
      Search on Wikipedia ↗
    </a>
  );
}
