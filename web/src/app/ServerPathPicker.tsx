import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react";

export type PathEntry = {
  name: string;
  path: string;
  type: "directory" | "file";
  selectable: boolean;
};

export type PathListing = {
  root: string;
  path: string;
  parent: string | null;
  canGoUp: boolean;
  entries: PathEntry[];
};

export type ServerPathPickerProps = {
  title: string;
  rootPath?: string;
  startPath?: string;
  selectionMode: "folder" | "file";
  allowedExtensions?: string[];
  listEntries(options?: {
    path?: string;
    selectionType?: "folder" | "file";
    extensions?: string[];
  }): Promise<PathListing>;
  onSelect(path: string): void;
  onCancel(): void;
};

type PickerRow =
  | { kind: "parent"; key: string; label: string; path: string }
  | { kind: "entry"; key: string; entry: PathEntry };

const entryLabel = (entry: PathEntry): string =>
  entry.type === "directory" ? `${entry.name}/` : entry.name;

const nonBlank = (value?: string): string | undefined => {
  const trimmed = value?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : undefined;
};

export function ServerPathPicker({
  title,
  rootPath,
  startPath,
  selectionMode,
  allowedExtensions,
  listEntries,
  onSelect,
  onCancel,
}: ServerPathPickerProps) {
  const [listing, setListing] = useState<PathListing | null>(null);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [warning, setWarning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const requestIdRef = useRef(0);
  const dialogRef = useRef<HTMLElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const titleId = useId();
  const descriptionId = useId();

  const effectiveRoot = nonBlank(rootPath);
  const effectiveStart = nonBlank(startPath);
  const fallbackPath = effectiveRoot;
  const initialPath = effectiveStart ?? fallbackPath;

  const loadPath = useCallback(
    async (path?: string): Promise<PathListing | null> => {
      const requestId = requestIdRef.current + 1;
      requestIdRef.current = requestId;
      setIsLoading(true);
      setError(null);
      let next: PathListing;
      try {
        next = await listEntries({
          path,
          selectionType: selectionMode,
          extensions: allowedExtensions,
        });
      } catch (loadError) {
        if (requestId === requestIdRef.current) {
          setIsLoading(false);
        }
        throw loadError;
      }
      if (requestId !== requestIdRef.current) {
        return null;
      }
      setListing(next);
      setSelectedPath(null);
      setActiveIndex(0);
      setIsLoading(false);
      return next;
    },
    [allowedExtensions, listEntries, selectionMode],
  );

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      try {
        await loadPath(initialPath);
        if (!cancelled) {
          setWarning(null);
        }
      } catch {
        const canFallback = effectiveStart && effectiveStart !== fallbackPath;
        if (!canFallback) {
          if (!cancelled) {
            setError("Unable to load directory listing.");
            setIsLoading(false);
          }
          return;
        }

        try {
          await loadPath(fallbackPath);
          if (!cancelled) {
            setWarning(`Could not open ${effectiveStart}; showing the default browsing root.`);
          }
        } catch {
          if (!cancelled) {
            setError("Unable to load directory listing.");
            setIsLoading(false);
          }
        }
      }
    };

    void initialize();

    return () => {
      cancelled = true;
      requestIdRef.current += 1;
    };
  }, [effectiveStart, fallbackPath, initialPath, loadPath]);

  useEffect(() => {
    dialogRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!isLoading && !error) {
      listRef.current?.focus();
    }
  }, [error, isLoading, listing?.path]);

  const showParent = Boolean(
    listing?.parent && (!effectiveRoot || listing.path !== effectiveRoot),
  );

  const rows = useMemo<PickerRow[]>(() => {
    if (!listing) {
      return [];
    }

    const nextRows: PickerRow[] = [];
    if (showParent && listing.parent) {
      nextRows.push({ kind: "parent", key: "parent", label: "../", path: listing.parent });
    }
    const sortedEntries = [...listing.entries].sort((left, right) => {
      if (left.type === right.type) {
        return 0;
      }
      return left.type === "directory" ? -1 : 1;
    });
    for (const entry of sortedEntries) {
      nextRows.push({ kind: "entry", key: entry.path, entry });
    }
    return nextRows;
  }, [listing, showParent]);

  const openPath = useCallback(
    async (path: string) => {
      try {
        await loadPath(path);
      } catch {
        setError("Unable to load directory listing.");
        setIsLoading(false);
      }
    },
    [loadPath],
  );

  const activateRow = useCallback(
    (row: PickerRow) => {
      if (row.kind === "parent") {
        void openPath(row.path);
        return;
      }

      const { entry } = row;
      if (entry.selectable) {
        setSelectedPath(entry.path);
        return;
      }
      if (entry.type === "directory") {
        void openPath(entry.path);
      }
    },
    [openPath],
  );

  const openRow = useCallback(
    (row: PickerRow) => {
      if (row.kind === "parent") {
        void openPath(row.path);
        return;
      }
      if (row.entry.type === "directory") {
        void openPath(row.entry.path);
      }
    },
    [openPath],
  );

  const onDialogKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      onCancel();
    }
  };

  const onListKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (rows.length === 0) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((current) => Math.min(current + 1, rows.length - 1));
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((current) => Math.max(current - 1, 0));
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      activateRow(rows[activeIndex] ?? rows[0]);
      return;
    }

    if (event.key === "ArrowRight") {
      event.preventDefault();
      openRow(rows[activeIndex] ?? rows[0]);
    }
  };

  return (
    <div
      className="server-path-picker__backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onCancel();
        }
      }}
    >
      <section
        ref={dialogRef}
        className="server-path-picker"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        tabIndex={-1}
        onKeyDown={onDialogKeyDown}
      >
        <header className="server-path-picker__header">
          <div>
            <p className="server-path-picker__eyebrow">
              {selectionMode === "folder" ? "Server folder" : "Server file"}
            </p>
            <h2 id={titleId}>{title}</h2>
          </div>
          <button type="button" className="server-path-picker__close" onClick={onCancel}>
            Close
          </button>
        </header>

        <div id={descriptionId} className="server-path-picker__pathbar">
          <span>Current path</span>
          <code>{listing?.path ?? "Loading..."}</code>
        </div>

        {warning ? <p className="server-path-picker__warning">{warning}</p> : null}
        {error ? <p className="server-path-picker__error">{error}</p> : null}
        {isLoading ? <p className="server-path-picker__status">Loading paths…</p> : null}

        {!isLoading && !error ? (
          <div
            ref={listRef}
            className="server-path-picker__list"
            role="listbox"
            aria-label="Server path entries"
            tabIndex={0}
            onKeyDown={onListKeyDown}
          >
            {rows.length === 0 ? <p className="server-path-picker__empty">No entries in this directory.</p> : null}

            {rows.map((row, index) => {
              if (row.kind === "parent") {
                return (
                  <div
                    key={row.key}
                    role="option"
                    aria-selected={false}
                    className={`server-path-picker__row${index === activeIndex ? " is-active" : ""}`}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => void openPath(row.path)}
                    onDoubleClick={() => void openPath(row.path)}
                  >
                    <span className="server-path-picker__name">{row.label}</span>
                  </div>
                );
              }

              const isSelected = selectedPath === row.entry.path;
              const isActive = index === activeIndex;
              const label = entryLabel(row.entry);
              const isDisabled = !row.entry.selectable && row.entry.type === "file";
              return (
                <div
                  key={row.key}
                  role="option"
                  aria-label={label}
                  aria-selected={isSelected}
                  aria-disabled={isDisabled || undefined}
                  className={`server-path-picker__row${isActive ? " is-active" : ""}${isSelected ? " is-selected" : ""}${isDisabled ? " is-disabled" : ""}`}
                  onClick={() => {
                    if (row.entry.selectable) {
                      setSelectedPath(row.entry.path);
                      return;
                    }
                    if (row.entry.type === "directory") {
                      void openPath(row.entry.path);
                    }
                  }}
                  onDoubleClick={() => {
                    if (row.entry.type === "directory") {
                      void openPath(row.entry.path);
                    }
                  }}
                  onMouseEnter={() => setActiveIndex(index)}
                >
                  <span className="server-path-picker__name">{label}</span>
                </div>
              );
            })}
          </div>
        ) : null}

        <footer className="server-path-picker__footer">
          <div className="server-path-picker__selection" aria-live="polite">
            <span>Selected</span>
            <code>{selectedPath ?? "Choose an entry"}</code>
          </div>
          <div className="server-path-picker__actions">
            <button type="button" className="secondary-button" onClick={onCancel}>
              Cancel
            </button>
            <button type="button" onClick={() => selectedPath && onSelect(selectedPath)} disabled={!selectedPath}>
              Use selection
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}
