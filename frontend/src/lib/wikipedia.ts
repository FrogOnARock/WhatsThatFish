/* Wikipedia link-out for a species. Uses the search endpoint (not a direct
   /wiki/<title> article) so a scientific name without an exact article title
   lands on Wikipedia's suggestions rather than a dead article page. */
export function wikipediaSearchUrl(name: string): string {
  return `https://en.wikipedia.org/w/index.php?search=${encodeURIComponent(
    name.trim(),
  )}`;
}
