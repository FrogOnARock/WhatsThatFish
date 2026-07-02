/* Tests for the snake_case → camelCase mapping at the API seam. fetch is mocked
   so these are pure data-shape tests (no network). */
import { describe, it, expect, beforeEach, vi } from "vitest";
import { listDives } from "./observations";
import { getMe } from "./auth";
import { getFieldLog } from "./history";
import { getUserStats } from "./userStats";

function mockFetch(payload: unknown) {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => payload,
  }) as unknown as typeof fetch;
}

beforeEach(() => {
  localStorage.setItem("wtf_token", "test-token");
});

describe("listDives → mapDive", () => {
  it("maps snake_case + nested species, with defaults", async () => {
    mockFetch([
      {
        id: "d1", site_id: "s1", site_name: "Blue Hole",
        gps_lat: 1.5, gps_lng: 2.5, dived_at: "2026-01-01", notes: "viz 30m",
        created_at: "2026-01-02", observation_count: 3,
        species: [{ taxon_id: 1001, name: "Amphiprion ocellaris", common_name: "Clownfish" }],
      },
    ]);
    const [d] = await listDives();
    expect(d.siteName).toBe("Blue Hole");
    expect(d.gpsLat).toBe(1.5);
    expect(d.observationCount).toBe(3);
    expect(d.species[0]).toEqual({
      taxonId: 1001, name: "Amphiprion ocellaris", commonName: "Clownfish",
    });
  });

  it("defaults missing count/species", async () => {
    mockFetch([
      {
        id: "d1", site_id: null, site_name: null, gps_lat: null, gps_lng: null,
        dived_at: null, notes: null, created_at: "2026-01-02",
        // observation_count + species intentionally absent
      },
    ]);
    const [d] = await listDives();
    expect(d.observationCount).toBe(0);
    expect(d.species).toEqual([]);
  });
});

describe("getMe → mapProfile", () => {
  it("maps app-owned fields with unit default", async () => {
    mockFetch({
      id: "u1", email: "a@test.dev", display_name: "Diver",
      avatar_url: "http://x/a.png", preferred_name: "Reef", unit_system: "imperial",
    });
    const p = await getMe("tok");
    expect(p.displayName).toBe("Diver");
    expect(p.preferredName).toBe("Reef");
    expect(p.unitSystem).toBe("imperial");
  });

  it("defaults unit_system to metric when absent", async () => {
    mockFetch({ id: "u1", email: null, display_name: null, avatar_url: null });
    const p = await getMe("tok");
    expect(p.unitSystem).toBe("metric");
    expect(p.preferredName).toBeNull();
  });
});

describe("getFieldLog mapping", () => {
  it("maps nested sightings + photos to camelCase", async () => {
    mockFetch({
      total_species: 1,
      species: [
        {
          taxon_id: 1001, species: "Amphiprion ocellaris", genus: "Amphiprion",
          family: "Pomacentridae", common_name: "Clownfish", sighting_count: 1,
          sightings: [
            {
              observation_id: "o1", dive_id: "d1", dived_at: "2026-01-01",
              site_name: "Reef", depth_m: 12, label_status: "predicted",
              photos: [{ id: "p1", bbox: null, width: 100, height: 80 }],
            },
          ],
        },
      ],
    });
    const log = await getFieldLog();
    expect(log[0].taxonId).toBe(1001);
    expect(log[0].commonName).toBe("Clownfish");
    const s = log[0].sightings[0];
    expect(s.observationId).toBe("o1");
    expect(s.diveId).toBe("d1");
    expect(s.depthM).toBe(12);
    expect(s.photos[0].id).toBe("p1");
  });
});

describe("getUserStats mapping", () => {
  it("maps unique_species → uniqueSpecies", async () => {
    mockFetch({ dives: 2, observations: 5, unique_species: 3 });
    const stats = await getUserStats();
    expect(stats).toEqual({ dives: 2, observations: 5, uniqueSpecies: 3 });
  });
});
