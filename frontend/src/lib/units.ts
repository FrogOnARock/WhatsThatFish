/* Unit conversion for user-facing depths. The DB stores depth in METERS
   (depth_m); the UI shows + accepts it in the signed-in user's preferred system.
   Keep all m↔ft logic here so display and input stay symmetric (a value shown
   as "59 ft" must round-trip back to the same meters on save). */
import type { UnitSystem } from "../api/types";

const M_TO_FT = 3.28084;

/** Unit suffix for the system: "m" or "ft". */
export function unitLabel(unitSystem: UnitSystem): string {
  return unitSystem === "imperial" ? "ft" : "m";
}

/** Stored meters → the user's unit, rounded to a whole number for display. */
export function toDisplayDepth(meters: number, unitSystem: UnitSystem): number {
  const value = unitSystem === "imperial" ? meters * M_TO_FT : meters;
  return Math.round(value);
}

/** Stored meters → a display string, e.g. "18 m" / "59 ft". Null → em dash. */
export function formatDepth(
  meters: number | null | undefined,
  unitSystem: UnitSystem,
): string {
  if (meters == null) return "—";
  return `${toDisplayDepth(meters, unitSystem)} ${unitLabel(unitSystem)}`;
}

/** A user-entered depth (in their unit) → meters for storage, 2dp so an
    imperial round-trip is stable (59 ft → 17.98 m → 59 ft). */
export function toMeters(value: number, unitSystem: UnitSystem): number {
  const meters = unitSystem === "imperial" ? value / M_TO_FT : value;
  return Math.round(meters * 100) / 100;
}
