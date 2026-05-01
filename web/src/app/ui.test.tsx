import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ActionButton, PageHeader, Panel, StatusBadge } from "./ui";

describe("admin UI primitives", () => {
  it("renders a panel with a named region", () => {
    render(
      <Panel title="Registry health" description="Current workflow registry state.">
        <p>12 workflows loaded</p>
      </Panel>,
    );

    expect(screen.getByRole("region", { name: "Registry health" })).toBeTruthy();
    expect(screen.getByText("Current workflow registry state.")).toBeTruthy();
    expect(screen.getByText("12 workflows loaded")).toBeTruthy();
  });

  it("renders page header metadata and actions", () => {
    render(
      <PageHeader
        eyebrow="Operations"
        title="Workflows"
        description="Review workflow templates."
        actions={<ActionButton variant="primary">Reload</ActionButton>}
      />,
    );

    expect(screen.getByText("Operations")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Workflows" })).toBeTruthy();
    expect(screen.getByText("Review workflow templates.")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Reload" })).toBeTruthy();
  });

  it("marks status badges with their visual tone", () => {
    render(<StatusBadge tone="success">Connected</StatusBadge>);

    expect(screen.getByText("Connected").getAttribute("data-tone")).toBe("success");
  });
});
