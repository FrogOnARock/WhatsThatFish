import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Separate from vite.config.ts so the app's `tsc --noEmit` (which type-checks
// vite.config.ts) isn't tripped by vitest pulling in its own vite types.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    // Concrete origin so localStorage is available (jsdom's about:blank is opaque).
    environmentOptions: { jsdom: { url: "http://localhost:5173" } },
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
});
