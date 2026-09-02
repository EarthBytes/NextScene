"use client";

import { X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useGenres, usePreferences, useUpdatePreferences } from "@/hooks/use-api";
import { cn } from "@/lib/utils";

export function GenrePreferences() {
  const { data: genres } = useGenres();
  const { data: preferences } = usePreferences();
  const updatePreferences = useUpdatePreferences();

  const availableGenres = new Set(genres?.map((entry) => entry.genre) ?? []);
  const selected = new Set(
    (preferences?.preferred_genres ?? []).filter((genre) => availableGenres.has(genre)),
  );
  const busy = updatePreferences.isPending;

  const toggleGenre = (genre: string) => {
    const next = new Set(selected);
    if (next.has(genre)) {
      next.delete(genre);
    } else {
      next.add(genre);
    }
    updatePreferences.mutate(Array.from(next));
  };

  const clear = () => updatePreferences.mutate([]);

  if (!genres?.length) return null;

  return (
    <section className="mx-4 rounded-xl border border-zinc-800 bg-zinc-950 p-4 sm:mx-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-white">Recommendation genres</h2>
          <p className="mt-1 text-xs text-zinc-500">
            Pick genres to focus your picks. Leave empty to use your library taste.
          </p>
        </div>
        {selected.size > 0 ? (
          <Button variant="ghost" size="sm" disabled={busy} onClick={clear}>
            <X className="h-3 w-3" />
            Clear
          </Button>
        ) : null}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {genres.map(({ genre }) => {
          const active = selected.has(genre);
          return (
            <button
              key={genre}
              type="button"
              disabled={busy}
              onClick={() => toggleGenre(genre)}
              className={cn(
                "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                active
                  ? "border-red-600 bg-red-600/20 text-red-300"
                  : "border-zinc-700 text-zinc-400 hover:border-zinc-500 hover:text-white",
              )}
            >
              {genre}
            </button>
          );
        })}
      </div>
    </section>
  );
}
