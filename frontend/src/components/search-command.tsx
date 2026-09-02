"use client";

import { Command } from "cmdk";
import { Search, Tag } from "lucide-react";
import { useEffect, useState } from "react";

import { MovieCard } from "@/components/movie-card";
import { CommandDialog, CommandDialogContent } from "@/components/ui/command-dialog";
import { useAddMovie, useGenres, useItemSearch, useTags } from "@/hooks/use-api";
import { useUiStore } from "@/lib/store";
import { cn } from "@/lib/utils";

type SearchMode = "title" | "genre" | "tag";

export function SearchCommand() {
  const open = useUiStore((s) => s.searchOpen);
  const setSearchOpen = useUiStore((s) => s.setSearchOpen);
  const setSelectedItemId = useUiStore((s) => s.setSelectedItemId);
  const [mode, setMode] = useState<SearchMode>("title");
  const [query, setQuery] = useState("");
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const { data: genres } = useGenres();
  const { data: tags } = useTags();
  const { data, isFetching } = useItemSearch({
    q: mode === "title" ? query : undefined,
    genre: mode === "genre" ? selectedGenre ?? undefined : undefined,
    tag: mode === "tag" ? selectedTag ?? query : undefined,
  });
  const addMovie = useAddMovie();
  const [addingId, setAddingId] = useState<number | null>(null);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen(true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [setSearchOpen]);

  const handleAdd = (itemId: number) => {
    setAddingId(itemId);
    addMovie.mutate(itemId, { onSettled: () => setAddingId(null) });
  };

  const resetMode = (next: SearchMode) => {
    setMode(next);
    setQuery("");
    setSelectedGenre(null);
    setSelectedTag(null);
  };

  const hasActiveSearch =
    (mode === "title" && query.trim().length >= 2) ||
    (mode === "genre" && Boolean(selectedGenre)) ||
    (mode === "tag" && Boolean(selectedTag || query.trim().length >= 2));

  return (
    <CommandDialog open={open} onOpenChange={setSearchOpen}>
      <CommandDialogContent title="Search movies">
        <Command className="rounded-xl bg-black">
          <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2">
            {(["title", "genre", "tag"] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => resetMode(tab)}
                className={cn(
                  "rounded-md px-3 py-1 text-xs font-medium capitalize",
                  mode === tab ? "bg-red-600 text-white" : "text-zinc-500 hover:text-white",
                )}
              >
                {tab}
              </button>
            ))}
          </div>

          {mode === "title" ? (
            <div className="flex items-center gap-3 border-b border-zinc-800 px-4 py-3">
              <Search className="h-4 w-4 text-zinc-500" />
              <Command.Input
                value={query}
                onValueChange={setQuery}
                placeholder="Search by movie title…"
                className="w-full bg-transparent text-sm text-white outline-none placeholder:text-zinc-600"
              />
            </div>
          ) : null}

          {mode === "genre" ? (
            <div className="max-h-36 overflow-y-auto border-b border-zinc-800 p-4 no-scrollbar">
              <div className="flex flex-wrap gap-2">
                {genres?.map(({ genre }) => (
                  <button
                    key={genre}
                    type="button"
                    onClick={() => setSelectedGenre(genre)}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs",
                      selectedGenre === genre
                        ? "border-red-600 bg-red-600/20 text-red-300"
                        : "border-zinc-700 text-zinc-400 hover:text-white",
                    )}
                  >
                    {genre}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {mode === "tag" ? (
            <div className="border-b border-zinc-800 p-4">
              <div className="flex items-center gap-3">
                <Tag className="h-4 w-4 text-zinc-500" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search tags like pixar, superhero…"
                  className="w-full bg-transparent text-sm text-white outline-none placeholder:text-zinc-600"
                />
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {tags?.slice(0, 12).map(({ tag }) => (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => {
                      setSelectedTag(tag);
                      setQuery(tag);
                    }}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs",
                      selectedTag === tag
                        ? "border-red-600 bg-red-600/20 text-red-300"
                        : "border-zinc-700 text-zinc-400 hover:text-white",
                    )}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <Command.List className="max-h-[420px] overflow-y-auto p-4">
            {!hasActiveSearch ? (
              <p className="px-2 py-6 text-center text-sm text-zinc-600">
                {mode === "title"
                  ? "Type at least 2 characters to search"
                  : mode === "genre"
                    ? "Select a genre to browse movies"
                    : "Pick or type a tag to browse movies"}
              </p>
            ) : isFetching ? (
              <p className="px-2 py-4 text-sm text-zinc-500">Searching…</p>
            ) : data?.items.length === 0 ? (
              <p className="px-2 py-4 text-sm text-zinc-500">No movies found</p>
            ) : (
              <div className="flex gap-4 overflow-x-auto pb-2 no-scrollbar">
                {data?.items.map((item) => (
                  <MovieCard
                    key={item.item_id}
                    item={{ ...item, title: item.title }}
                    onOpen={() => {
                      setSelectedItemId(item.item_id);
                      setSearchOpen(false);
                    }}
                    onAdd={handleAdd}
                    showAdd
                    adding={addingId === item.item_id}
                  />
                ))}
              </div>
            )}
          </Command.List>
        </Command>
      </CommandDialogContent>
    </CommandDialog>
  );
}
