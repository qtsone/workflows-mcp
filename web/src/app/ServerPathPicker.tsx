import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";

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

  const showParent = Boolean(
    listing?.parent && (!effectiveRoot || listing.path !== effectiveRoot),
  );

  const rows = useMemo<PickerRow[]>(() => {
    if (!listing) {
      return [];
    }

    const nextRows: PickerRow[] = [];
    if (showParent && listing.parent) {
      nextRows.push({ kind: "parent", key: "parent", label: "..", path: listing.parent });
    }
    for (const entry of listing.entries) {
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

  const onListKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      onCancel();
      return;
    }

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
    }
  };

  return (
    <section className="server-path-picker" aria-label={title}>
      <header className="server-path-picker__header">
        <h2>{title}</h2>
        <code>{`$ ls ${listing?.path ?? "..."}`}</code>
      </header>

      {warning ? <p className="server-path-picker__warning">{warning}</p> : null}
      {error ? <p className="server-path-picker__error">{error}</p> : null}
      {isLoading ? <p>Loading paths…</p> : null}

      {!isLoading && !error ? (
        <div
          className="server-path-picker__list"
          role="listbox"
          aria-label="Server path entries"
          tabIndex={0}
          onKeyDown={onListKeyDown}
        >
          {rows.length === 0 ? <p>No entries in this directory.</p> : null}

          {rows.map((row, index) => {
            if (row.kind === "parent") {
              return (
                <div
                  key={row.key}
                  role="option"
                  aria-selected={index === activeIndex}
                  className="server-path-picker__row"
                >
                  <span className="server-path-picker__name">{row.label}</span>
                  <button type="button" onClick={() => void openPath(row.path)}>
                    Open parent directory
                  </button>
                </div>
              );
            }

            const isSelected = selectedPath === row.entry.path;
            const isActive = index === activeIndex;
            return (
              <div
                key={row.key}
                role="option"
                aria-label={row.entry.name}
                aria-selected={isSelected || isActive}
                className={`server-path-picker__row${isSelected ? " is-selected" : ""}`}
                onClick={() => {
                  if (row.entry.selectable) {
                    setSelectedPath(row.entry.path);
                  }
                }}
              >
                <button
                  type="button"
                  className="server-path-picker__entry"
                  onClick={() => {
                    if (row.entry.selectable) {
                      setSelectedPath(row.entry.path);
                    }
                  }}
                >
                  {row.entry.name}
                </button>

                {row.entry.type === "directory" ? (
                  <button type="button" onClick={() => void openPath(row.entry.path)}>
                    Open {row.entry.name}
                  </button>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}

      <div className="server-path-picker__actions">
        <button type="button" className="secondary-button" onClick={onCancel}>
          Cancel
        </button>
        <button type="button" onClick={() => selectedPath && onSelect(selectedPath)} disabled={!selectedPath}>
          Use selection
        </button>
      </div>
    </section>
  );
}
