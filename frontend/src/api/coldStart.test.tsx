/* Cold-start UX simulation, executed deterministically against the REAL
   BackendStatusProvider + ColdStartBanner (no browser). We drive a controllable
   fetch: a slow /health (cold boot) then recovery, and a hard-503 (failure tail),
   and assert the actual DOM the user would see. */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act, waitFor } from "@testing-library/react";
import { BackendStatusProvider } from "./backendStatus";
import ColdStartBanner from "../components/ColdStartBanner";

function renderBanner() {
  return render(
    <BackendStatusProvider>
      <ColdStartBanner />
    </BackendStatusProvider>,
  );
}

afterEach(() => vi.useRealTimers());

describe("cold start (slow /health then recovery)", () => {
  beforeEach(() => {
    // /health hangs until aborted OR resolves late — here we control it with a
    // deferred promise so we can assert the "warming" window explicitly.
    let resolveHealth: (r: Response) => void;
    const healthPromise = new Promise<Response>((res) => (resolveHealth = res));
    (globalThis as any).__resolveHealth = () =>
      resolveHealth({ ok: true, status: 200, json: async () => ({ status: "ok" }) } as Response);
    global.fetch = vi.fn(() => healthPromise) as unknown as typeof fetch;
  });

  it("shows the waking-the-model banner while the instance boots, then hides it", async () => {
    renderBanner();

    // During boot: banner is visible with the cold-start copy + accessible role.
    const banner = await screen.findByRole("status");
    expect(banner).toHaveTextContent(/waking the model/i);
    expect(banner).toHaveTextContent(/can take ~30s/i);

    // Instance finishes booting → /health resolves 200 → status flips to online.
    await act(async () => {
      (globalThis as any).__resolveHealth();
    });

    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
  });
});

describe("failure tail (persistent 503)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // Every ping 503s. apiFetch returns the 503 (no throw); the provider re-pings
    // with backoff and, after exhausting them, lands on "down".
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ detail: "cold" }),
    }) as unknown as typeof fetch;
  });

  it("falls back to the offline banner after retries are exhausted", async () => {
    renderBanner();

    // Drain the ping/backoff loop (backoffs are 1s,2s,4s,8s → advance generously).
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });

    const banner = screen.getByRole("status");
    expect(banner).toHaveTextContent(/can't reach the model service/i);
    expect(banner.className).toMatch(/cold-banner--down/);
  });
});
