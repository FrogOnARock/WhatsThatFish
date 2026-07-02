import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import SiteAutocomplete from "./SiteAutocomplete";
import * as api from "../api/observations";

// Mock the network call; the component logic (debounced search → dropdown →
// pick) is what's under test.
vi.mock("../api/observations", () => ({
  searchSites: vi.fn(),
}));

const mockedSearch = vi.mocked(api.searchSites);

/** Controlled wrapper so typing drives the value prop like the real callers. */
function Harness() {
  const [value, setValue] = useState("");
  return (
    <div>
      <SiteAutocomplete value={value} onChange={setValue} placeholder="Dive site" />
      <span data-testid="value">{value}</span>
    </div>
  );
}

beforeEach(() => {
  mockedSearch.mockReset();
});

describe("SiteAutocomplete", () => {
  it("suggests existing sites as you type and fills the field on pick", async () => {
    mockedSearch.mockResolvedValue([{ id: "s1", name: "Blue Hole" }]);
    render(<Harness />);

    await userEvent.type(screen.getByPlaceholderText("Dive site"), "blue");

    const option = await screen.findByRole("button", { name: "Blue Hole" });
    await userEvent.click(option);

    // Field now holds the picked site; dropdown is gone.
    expect(screen.getByTestId("value")).toHaveTextContent("Blue Hole");
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Blue Hole" })).not.toBeInTheDocument(),
    );
  });

  it("hides an exact-name match (nothing new to suggest)", async () => {
    mockedSearch.mockResolvedValue([{ id: "s1", name: "Reef" }]);
    render(<Harness />);
    await userEvent.type(screen.getByPlaceholderText("Dive site"), "Reef");
    // The only result equals the typed value → no suggestion button.
    await waitFor(() => expect(mockedSearch).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: "Reef" })).not.toBeInTheDocument();
  });
});
