/* Free-text dive-site input with existing-site suggestions. The user can type a
   brand-new site name, but matching sites surface as a dropdown so they reuse
   one instead of creating a near-duplicate. Backend match is substring on the
   normalized name_key (case/spacing-insensitive). */
import { useEffect, useRef, useState } from "react";
import { searchSites, type SiteOption } from "../api/observations";

interface Props {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
}

export default function SiteAutocomplete({
  value,
  onChange,
  placeholder,
  className,
}: Props) {
  const [results, setResults] = useState<SiteOption[]>([]);
  const [open, setOpen] = useState(false);
  // True only right after a suggestion click, so we don't immediately re-search
  // (and re-open the dropdown) for the value we just set.
  const justPicked = useRef(false);

  useEffect(() => {
    if (justPicked.current) {
      justPicked.current = false;
      return;
    }
    if (value.trim().length < 1) {
      setResults([]);
      return;
    }
    let cancelled = false;
    searchSites(value)
      .then((r) => {
        if (cancelled) return;
        setResults(r);
        setOpen(r.length > 0);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [value]);

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
        onChange={(e) => onChange(e.target.value)}
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
                onClick={() => {
                  justPicked.current = true;
                  onChange(s.name);
                  setOpen(false);
                }}
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
