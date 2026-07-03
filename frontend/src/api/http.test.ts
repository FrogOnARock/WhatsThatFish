/* apiFetch cold-start behavior: transient-5xx retry, timeout → typed error, and
   the BACKEND_OK/SLOW signals the status context relies on. fetch is mocked; a
   signal-aware mock lets us exercise the AbortController timeout path. */
import { describe, it, expect, vi } from "vitest";
import { apiFetch, BackendTimeoutError } from "./http";
import { BACKEND_OK_EVENT, BACKEND_SLOW_EVENT } from "./events";

function okResponse(status = 200) {
  return { ok: status < 400, status, json: async () => ({}) } as unknown as Response;
}

describe("apiFetch retry", () => {
  it("retries once on 503 then returns the 200", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(okResponse(503))
      .mockResolvedValueOnce(okResponse(200));
    global.fetch = fetchMock as unknown as typeof fetch;

    const res = await apiFetch("http://x/stats", {}, { timeoutMs: 5_000, retries: 1 });

    expect(res.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("gives up after exhausting retries and returns the final 503", async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse(503));
    global.fetch = fetchMock as unknown as typeof fetch;

    const res = await apiFetch("http://x/stats", {}, { timeoutMs: 5_000, retries: 1 });

    expect(res.status).toBe(503);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("apiFetch timeout", () => {
  it("aborts and throws BackendTimeoutError when the socket hangs", async () => {
    // Mock that never resolves on its own — only the AbortController can end it,
    // exactly like a Cloud Run cold start holding the connection open.
    global.fetch = vi.fn(
      (_url: string, init: RequestInit = {}) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    ) as unknown as typeof fetch;

    await expect(
      apiFetch("http://x/predict", {}, { timeoutMs: 40, retries: 0 }),
    ).rejects.toBeInstanceOf(BackendTimeoutError);
  });
});

describe("apiFetch events", () => {
  it("emits BACKEND_OK on a 2xx", async () => {
    global.fetch = vi.fn().mockResolvedValue(okResponse(200)) as unknown as typeof fetch;
    const ok = vi.fn();
    window.addEventListener(BACKEND_OK_EVENT, ok);
    await apiFetch("http://x/health", {}, { timeoutMs: 5_000 });
    window.removeEventListener(BACKEND_OK_EVENT, ok);
    expect(ok).toHaveBeenCalled();
  });

  it("emits BACKEND_SLOW on a timeout", async () => {
    global.fetch = vi.fn(
      (_url: string, init: RequestInit = {}) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () =>
            reject(new DOMException("aborted", "AbortError")),
          );
        }),
    ) as unknown as typeof fetch;
    const slow = vi.fn();
    window.addEventListener(BACKEND_SLOW_EVENT, slow);
    await expect(
      apiFetch("http://x/predict", {}, { timeoutMs: 40, retries: 0 }),
    ).rejects.toBeInstanceOf(BackendTimeoutError);
    window.removeEventListener(BACKEND_SLOW_EVENT, slow);
    expect(slow).toHaveBeenCalled();
  });
});
