import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ServerPathPicker, type PathListing } from "./ServerPathPicker";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const listing = (overrides: Partial<PathListing> = {}): PathListing => ({
  root: "/",
  path: "/workspace",
  parent: "/",
  canGoUp: true,
  entries: [
    { name: "alpha", path: "/workspace/alpha", type: "directory", selectable: true },
    { name: "beta", path: "/workspace/beta", type: "directory", selectable: true },
  ],
  ...overrides,
});

const deferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe("ServerPathPicker", () => {
  it("renders modal path context and parent row when parent exists", async () => {
    const listEntries = vi.fn(async () => listing());

    render(
      <ServerPathPicker
        title="Pick path"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(await screen.findByRole("dialog", { name: "Pick path" })).toBeTruthy();
    expect(await screen.findByText("/workspace")).toBeTruthy();
    expect(screen.getByRole("option", { name: "../" })).toBeTruthy();
  });

  it("renders simplified folder-first rows without duplicated open controls", async () => {
    const listEntries = vi.fn(async () =>
      listing({
        parent: null,
        canGoUp: false,
        entries: [
          { name: "readme.md", path: "/workspace/readme.md", type: "file", selectable: true },
          { name: "src", path: "/workspace/src", type: "directory", selectable: true },
          { name: "docs", path: "/workspace/docs", type: "directory", selectable: true },
          { name: "license", path: "/workspace/license", type: "file", selectable: true },
        ],
      }),
    );

    render(
      <ServerPathPicker
        title="Pick path"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    const rows = await screen.findAllByRole("option");

    expect(rows.map((row) => row.textContent)).toEqual(["src/", "docs/", "readme.md", "license"]);
    expect(screen.queryByRole("button", { name: "Open src" })).toBeNull();
    expect(screen.queryByText("/workspace/src")).toBeNull();
    expect(screen.queryByText("directory")).toBeNull();
  });

  it("clicking parent row loads parent path", async () => {
    const listEntries = vi
      .fn<({ path }: { path?: string }) => Promise<PathListing>>()
      .mockResolvedValueOnce(listing())
      .mockResolvedValueOnce(listing({ path: "/", parent: null, canGoUp: false, entries: [] }));

    render(
      <ServerPathPicker
        title="Pick path"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    fireEvent.click(await screen.findByRole("option", { name: "../" }));

    await waitFor(() => {
      expect(listEntries).toHaveBeenNthCalledWith(2, {
        path: "/",
        selectionType: "folder",
        extensions: undefined,
      });
    });
  });

  it("folder mode selects a directory row and confirms", async () => {
    const listEntries = vi.fn(async () => listing());
    const onSelect = vi.fn();

    render(
      <ServerPathPicker
        title="Pick folder"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={onSelect}
        onCancel={vi.fn()}
      />,
    );

    fireEvent.click(await screen.findByRole("option", { name: "alpha/" }));
    fireEvent.click(screen.getByRole("button", { name: "Use selection" }));

    expect(onSelect).toHaveBeenCalledWith("/workspace/alpha");
  });

  it("file mode opens directories and selects matching files", async () => {
    const first = listing({
      entries: [
        { name: "src", path: "/workspace/src", type: "directory", selectable: false },
        { name: "readme.md", path: "/workspace/readme.md", type: "file", selectable: true },
      ],
    });
    const second = listing({
      path: "/workspace/src",
      parent: "/workspace",
      entries: [
        { name: "main.ts", path: "/workspace/src/main.ts", type: "file", selectable: true },
      ],
    });
    const listEntries = vi.fn().mockResolvedValueOnce(first).mockResolvedValueOnce(second);
    const onSelect = vi.fn();

    render(
      <ServerPathPicker
        title="Pick file"
        selectionMode="file"
        allowedExtensions={[".ts", ".md"]}
        listEntries={listEntries}
        onSelect={onSelect}
        onCancel={vi.fn()}
      />,
    );

    fireEvent.doubleClick(await screen.findByRole("option", { name: "src/" }));
    await screen.findByText("/workspace/src");

    fireEvent.click(screen.getByRole("option", { name: "main.ts" }));
    fireEvent.click(screen.getByRole("button", { name: "Use selection" }));

    expect(onSelect).toHaveBeenCalledWith("/workspace/src/main.ts");
  });

  it("supports ArrowDown Enter and Escape keyboard behavior", async () => {
    const listEntries = vi.fn(async () => listing());
    const onCancel = vi.fn();

    render(
      <ServerPathPicker
        title="Pick path"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={vi.fn()}
        onCancel={onCancel}
      />,
    );

    const listbox = await screen.findByRole("listbox", { name: "Server path entries" });

    fireEvent.keyDown(listbox, { key: "ArrowDown" });
    fireEvent.keyDown(listbox, { key: "Enter" });
    fireEvent.keyDown(listbox, { key: "Escape" });

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("falls back from invalid startPath and shows warning", async () => {
    const listEntries = vi
      .fn()
      .mockRejectedValueOnce(new Error("missing"))
      .mockResolvedValueOnce(listing({ path: "/" }));

    render(
      <ServerPathPicker
        title="Pick path"
        rootPath="/"
        startPath="/missing"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(
      await screen.findByText("Could not open /missing; showing the default browsing root."),
    ).toBeTruthy();
    expect(listEntries).toHaveBeenCalledTimes(2);
  });

  it("keeps the latest listing when start path requests resolve out of order", async () => {
    const first = deferred<PathListing>();
    const second = deferred<PathListing>();
    const listEntries = vi.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);

    const { rerender } = render(
      <ServerPathPicker
        title="Pick path"
        startPath="/first"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(listEntries).toHaveBeenCalledTimes(1);
    });

    rerender(
      <ServerPathPicker
        title="Pick path"
        startPath="/second"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(listEntries).toHaveBeenCalledTimes(2);
    });

    second.resolve(listing({ path: "/second", parent: "/", entries: [] }));
    expect(await screen.findByText("/second")).toBeTruthy();

    await act(async () => {
      first.resolve(listing({ path: "/first", parent: "/", entries: [] }));
      await first.promise;
    });

    await waitFor(() => {
      expect(screen.getByText("/second")).toBeTruthy();
      expect(screen.queryByText("/first")).toBeNull();
    });
  });

  it("shows loading, empty, and error states", async () => {
    const delayed = new Promise<PathListing>((resolve) => {
      setTimeout(() => resolve(listing({ entries: [], parent: null, canGoUp: false })), 20);
    });
    const listEntries = vi.fn().mockImplementationOnce(async () => delayed);

    render(
      <ServerPathPicker
        title="Pick path"
        selectionMode="folder"
        listEntries={listEntries}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText("Loading paths…")).toBeTruthy();
    expect(await screen.findByText("No entries in this directory.")).toBeTruthy();

    cleanup();

    const failingListEntries = vi.fn().mockRejectedValueOnce(new Error("boom"));

    render(
      <ServerPathPicker
        title="Pick path"
        selectionMode="folder"
        listEntries={failingListEntries}
        onSelect={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(await screen.findByText("Unable to load directory listing.")).toBeTruthy();
  });
});
