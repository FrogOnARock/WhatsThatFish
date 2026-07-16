/* Edit a dive's dive-level details: site (with existing-site autocomplete),
   date, the full dive-log record, and notes. All are shared by every sighting
   on the dive, so a disclaimer mirrors the one in EditSightingModal. Numeric
   fields are shown + entered in the user's unit system and stored metric. */
import { useState } from "react";
import type { Dive } from "../api/observations";
import { updateDive } from "../api/observations";
import SiteAutocomplete, { type PlacePick } from "./SiteAutocomplete";
import { useAuth } from "../auth/AuthContext";
import {
  distanceLabel,
  toDisplayDistance,
  distanceToMeters,
  tempLabel,
  toDisplayTemp,
  toCelsius,
  weightLabel,
  toDisplayWeight,
  toKg,
  pressureLabel,
  toDisplayPressure,
  toBar,
} from "../lib/units";

interface Props {
  dive: Dive;
  onClose: () => void;
  onSaved: () => void;
}

// Stored metric → the editable display string in the user's unit (or "").
const show = (
  metric: number | null,
  toDisplay: (v: number, u: "metric" | "imperial") => number,
  units: "metric" | "imperial",
) => (metric == null ? "" : String(toDisplay(metric, units)));

// Editable string → metric number for storage (null when blank).
const store = (
  s: string,
  toMetric: (v: number, u: "metric" | "imperial") => number,
  units: "metric" | "imperial",
) => {
  const t = s.trim();
  if (t === "") return null;
  const n = Number(t);
  return Number.isNaN(n) ? null : toMetric(n, units);
};

const asInt = (s: string): number | null => {
  const t = s.trim();
  if (t === "") return null;
  const n = parseInt(t, 10);
  return Number.isNaN(n) ? null : n;
};

export default function DiveEditModal({ dive, onClose, onSaved }: Props) {
  const { user } = useAuth();
  const units = user?.unitSystem ?? "metric";

  const [siteName, setSiteName] = useState(dive.siteName ?? "");
  // Coords/place from a Places pick. Null until the user picks (or after manual
  // typing), so a save that didn't touch the site never overwrites the dive's
  // existing coordinates.
  const [place, setPlace] = useState<PlacePick>({
    placeId: null,
    lat: null,
    lng: null,
  });
  const [divedAt, setDivedAt] = useState(dive.divedAt ? dive.divedAt.slice(0, 10) : "");
  const [startTime, setStartTime] = useState(
    dive.startedAt ? dive.startedAt.slice(11, 16) : "",
  );
  const [visibility, setVisibility] = useState(
    show(dive.visibilityM, toDisplayDistance, units),
  );
  const [airTemp, setAirTemp] = useState(show(dive.airTempC, toDisplayTemp, units));
  const [waterTemp, setWaterTemp] = useState(show(dive.waterTempC, toDisplayTemp, units));
  const [weight, setWeight] = useState(show(dive.weightKg, toDisplayWeight, units));
  const [exposureSuit, setExposureSuit] = useState(dive.exposureSuit ?? "");
  const [depthAvg, setDepthAvg] = useState(show(dive.depthAvgM, toDisplayDistance, units));
  const [depthMax, setDepthMax] = useState(show(dive.depthMaxM, toDisplayDistance, units));
  const [bottomTime, setBottomTime] = useState(
    dive.bottomTimeMin != null ? String(dive.bottomTimeMin) : "",
  );
  const [totalTime, setTotalTime] = useState(
    dive.totalTimeMin != null ? String(dive.totalTimeMin) : "",
  );
  const [endPressure, setEndPressure] = useState(
    show(dive.endPressureBar, toDisplayPressure, units),
  );
  const [diveShop, setDiveShop] = useState(dive.diveShop ?? "");
  const [notes, setNotes] = useState(dive.notes ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const dist = distanceLabel(units);
  const temp = tempLabel(units);

  async function handleSubmit() {
    setBusy(true);
    setError(null);
    try {
      await updateDive(dive.id, {
        siteName: siteName || undefined,
        // undefined (not null) when unset → PATCH leaves existing values intact.
        googlePlaceId: place.placeId ?? undefined,
        gpsLat: place.lat ?? undefined,
        gpsLng: place.lng ?? undefined,
        divedAt,
        // started_at is a full datetime; combine the dive date with the time.
        startedAt: divedAt && startTime ? `${divedAt}T${startTime}:00` : null,
        visibilityM: store(visibility, distanceToMeters, units),
        airTempC: store(airTemp, toCelsius, units),
        waterTempC: store(waterTemp, toCelsius, units),
        weightKg: store(weight, toKg, units),
        exposureSuit: exposureSuit.trim() || null,
        depthAvgM: store(depthAvg, distanceToMeters, units),
        depthMaxM: store(depthMax, distanceToMeters, units),
        bottomTimeMin: asInt(bottomTime),
        totalTimeMin: asInt(totalTime),
        endPressureBar: store(endPressure, toBar, units),
        diveShop: diveShop.trim() || null,
        notes,
      });
      onSaved();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const field = (
    label: string,
    value: string,
    setter: (v: string) => void,
    type = "number",
    placeholder = "",
  ) => (
    <div className="modal__field">
      <label className="modal__label">{label}</label>
      <input
        className="modal__input"
        type={type}
        inputMode={type === "number" ? "decimal" : undefined}
        value={value}
        placeholder={placeholder}
        onChange={(e) => setter(e.target.value)}
      />
    </div>
  );

  return (
    <div className="modal__backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="modal__title">Edit dive</h3>

        <p className="modal__disclaimer">
          These details are dive-level — changing them updates every sighting logged
          on this dive.
        </p>

        <div className="modal__field">
          <label className="modal__label">Dive site</label>
          <SiteAutocomplete
            value={siteName}
            onChange={setSiteName}
            onPlace={setPlace}
            placeholder="Dive site (e.g. Tulamben, Bali)"
          />
        </div>

        <div className="modal__field">
          <label className="modal__label">Date</label>
          <input
            className="modal__input"
            type="date"
            value={divedAt}
            onChange={(e) => setDivedAt(e.target.value)}
          />
        </div>

        {field("Start time", startTime, setStartTime, "time")}
        {field(`Visibility (${dist})`, visibility, setVisibility)}
        {field(`Air temp (${temp})`, airTemp, setAirTemp)}
        {field(`Water temp (${temp})`, waterTemp, setWaterTemp)}
        {field(`Weight (${weightLabel(units)})`, weight, setWeight)}
        {field("Exposure suit", exposureSuit, setExposureSuit, "text", "e.g. 5mm wetsuit")}
        {field(`Avg depth (${dist})`, depthAvg, setDepthAvg)}
        {field(`Max depth (${dist})`, depthMax, setDepthMax)}
        {field("Bottom time (min)", bottomTime, setBottomTime)}
        {field("Total dive time (min)", totalTime, setTotalTime)}
        {field(`Ending pressure (${pressureLabel(units)})`, endPressure, setEndPressure)}
        {field("Dive shop", diveShop, setDiveShop, "text", "")}

        <div className="modal__field">
          <label className="modal__label">Notes</label>
          <textarea
            className="modal__input modal__textarea"
            rows={4}
            placeholder="Conditions, buddies, anything memorable…"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
          />
        </div>

        {error && <div className="modal__error">{error}</div>}
        <div className="modal__actions">
          <button className="btn btn--ghost btn--sm" onClick={onClose} disabled={busy}>
            Cancel
          </button>
          <button className="btn btn--foam btn--sm" onClick={handleSubmit} disabled={busy}>
            {busy ? "Saving…" : "Save changes"}
          </button>
        </div>
      </div>
    </div>
  );
}
