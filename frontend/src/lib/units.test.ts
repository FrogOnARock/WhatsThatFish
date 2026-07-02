import { describe, it, expect } from "vitest";
import { unitLabel, toDisplayDepth, formatDepth, toMeters } from "./units";

describe("unitLabel", () => {
  it("returns m / ft", () => {
    expect(unitLabel("metric")).toBe("m");
    expect(unitLabel("imperial")).toBe("ft");
  });
});

describe("toDisplayDepth", () => {
  it("passes meters through (rounded) for metric", () => {
    expect(toDisplayDepth(18, "metric")).toBe(18);
    expect(toDisplayDepth(17.98, "metric")).toBe(18);
  });
  it("converts meters → feet for imperial", () => {
    expect(toDisplayDepth(12.19, "imperial")).toBe(40);
  });
});

describe("formatDepth", () => {
  it("formats with the unit suffix", () => {
    expect(formatDepth(18, "metric")).toBe("18 m");
    expect(formatDepth(12.19, "imperial")).toBe("40 ft");
  });
  it("renders an em dash for null/undefined", () => {
    expect(formatDepth(null, "metric")).toBe("—");
    expect(formatDepth(undefined, "imperial")).toBe("—");
  });
});

describe("toMeters", () => {
  it("passes metric entry through", () => {
    expect(toMeters(18, "metric")).toBe(18);
  });
  it("converts imperial entry to meters (2dp)", () => {
    expect(toMeters(40, "imperial")).toBe(12.19);
  });
});

describe("round-trip stability", () => {
  it("imperial entry → meters → display is lossless", () => {
    for (const ft of [5, 18, 40, 59, 100]) {
      const stored = toMeters(ft, "imperial");
      expect(toDisplayDepth(stored, "imperial")).toBe(ft);
    }
  });
});
