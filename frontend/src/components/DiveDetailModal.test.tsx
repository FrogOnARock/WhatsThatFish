import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DiveDetailModal from "./DiveDetailModal";
import type { Dive } from "../api/observations";

const DIVE: Dive = {
  id: "d1",
  siteId: "s1",
  siteName: "Blue Hole",
  gpsLat: 1.2345,
  gpsLng: 2.3456,
  divedAt: "2026-03-12",
  notes: "Great viz, saw a turtle.",
  createdAt: "2026-03-13",
  observationCount: 2,
  species: [
    { taxonId: 1001, name: "Amphiprion ocellaris", commonName: "Clownfish" },
    { taxonId: 2001, name: "Thalassoma lunare", commonName: null },
  ],
};

describe("DiveDetailModal", () => {
  it("renders site, notes, and species (prefers common name)", () => {
    render(<DiveDetailModal dive={DIVE} onClose={() => {}} onEdit={() => {}} />);
    expect(screen.getByText("Blue Hole")).toBeInTheDocument();
    expect(screen.getByText("Great viz, saw a turtle.")).toBeInTheDocument();
    // common name in brackets when present; bare scientific name otherwise.
    expect(screen.getByText("Amphiprion ocellaris (Clownfish)")).toBeInTheDocument();
    expect(screen.getByText("Thalassoma lunare")).toBeInTheDocument();
  });

  it("shows a placeholder when there are no notes", () => {
    render(
      <DiveDetailModal dive={{ ...DIVE, notes: null }} onClose={() => {}} onEdit={() => {}} />,
    );
    expect(screen.getByText("No notes for this dive.")).toBeInTheDocument();
  });

  it("fires onEdit and onClose", async () => {
    const onEdit = vi.fn();
    const onClose = vi.fn();
    render(<DiveDetailModal dive={DIVE} onClose={onClose} onEdit={onEdit} />);
    await userEvent.click(screen.getByRole("button", { name: "Edit dive" }));
    expect(onEdit).toHaveBeenCalledOnce();
    await userEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
