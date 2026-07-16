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

/* ── extended dive-log quantities ────────────────────────────────────────────
   Same store-metric / display-and-input-in-preferred-unit contract as depth.
   Distance (visibility) shares the m↔ft logic above. */

const KG_TO_LB = 2.20462;
const BAR_TO_PSI = 14.5038;

// Visibility is a distance — reuse the depth m↔ft conversions, own label.
export const distanceLabel = unitLabel;
export const formatDistance = formatDepth;
export const toDisplayDistance = toDisplayDepth;
export const distanceToMeters = toMeters;

/** Temperature: metric °C, imperial °F. */
export function tempLabel(unitSystem: UnitSystem): string {
  return unitSystem === "imperial" ? "°F" : "°C";
}
export function toDisplayTemp(celsius: number, unitSystem: UnitSystem): number {
  const v = unitSystem === "imperial" ? celsius * 9 / 5 + 32 : celsius;
  return Math.round(v);
}
export function toCelsius(value: number, unitSystem: UnitSystem): number {
  const c = unitSystem === "imperial" ? (value - 32) * 5 / 9 : value;
  return Math.round(c * 100) / 100;
}
export function formatTemp(
  celsius: number | null | undefined,
  unitSystem: UnitSystem,
): string {
  if (celsius == null) return "—";
  return `${toDisplayTemp(celsius, unitSystem)} ${tempLabel(unitSystem)}`;
}

/** Weight: metric kg, imperial lb. */
export function weightLabel(unitSystem: UnitSystem): string {
  return unitSystem === "imperial" ? "lb" : "kg";
}
export function toDisplayWeight(kg: number, unitSystem: UnitSystem): number {
  const v = unitSystem === "imperial" ? kg * KG_TO_LB : kg;
  return Math.round(v * 10) / 10;
}
export function toKg(value: number, unitSystem: UnitSystem): number {
  const kg = unitSystem === "imperial" ? value / KG_TO_LB : value;
  return Math.round(kg * 100) / 100;
}
export function formatWeight(
  kg: number | null | undefined,
  unitSystem: UnitSystem,
): string {
  if (kg == null) return "—";
  return `${toDisplayWeight(kg, unitSystem)} ${weightLabel(unitSystem)}`;
}

/** Tank pressure: metric bar, imperial psi. */
export function pressureLabel(unitSystem: UnitSystem): string {
  return unitSystem === "imperial" ? "psi" : "bar";
}
export function toDisplayPressure(bar: number, unitSystem: UnitSystem): number {
  const v = unitSystem === "imperial" ? bar * BAR_TO_PSI : bar;
  return Math.round(v);
}
export function toBar(value: number, unitSystem: UnitSystem): number {
  const bar = unitSystem === "imperial" ? value / BAR_TO_PSI : value;
  return Math.round(bar * 100) / 100;
}
export function formatPressure(
  bar: number | null | undefined,
  unitSystem: UnitSystem,
): string {
  if (bar == null) return "—";
  return `${toDisplayPressure(bar, unitSystem)} ${pressureLabel(unitSystem)}`;
}
