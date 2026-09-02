"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Bookmark, BookmarkX, Check, EyeOff, X } from "lucide-react";

import { PosterImage } from "@/components/poster-image";
import { StarRating } from "@/components/star-rating";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useClearRating,
  useDismissMovie,
  useExplanation,
  useItem,
  useMarkWatched,
  useMovieStatus,
  useRateMovie,
  useRemoveWatched,
  useRemoveWatchlist,
  useWatchlist,
} from "@/hooks/use-api";
import { useUiStore } from "@/lib/store";

export function MovieDetailModal() {
  const itemId = useUiStore((s) => s.selectedItemId);
  const setSelectedItemId = useUiStore((s) => s.setSelectedItemId);
  const { data: item, isLoading } = useItem(itemId);
  const { data: status, refetch: refetchStatus } = useMovieStatus(itemId);
  const { data: explanation, isLoading: explanationLoading } = useExplanation(itemId);

  const watchlist = useWatchlist();
  const removeWatchlist = useRemoveWatchlist();
  const markWatched = useMarkWatched();
  const removeWatched = useRemoveWatched();
  const rateMovie = useRateMovie();
  const clearRating = useClearRating();
  const dismissMovie = useDismissMovie();

  const close = () => setSelectedItemId(null);

  const busy =
    watchlist.isPending ||
    removeWatchlist.isPending ||
    markWatched.isPending ||
    removeWatched.isPending ||
    rateMovie.isPending ||
    clearRating.isPending ||
    dismissMovie.isPending;

  const refresh = () => refetchStatus();

  return (
    <Dialog.Root open={itemId != null} onOpenChange={(open) => !open && close()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm" />
        <Dialog.Content
          className="fixed left-1/2 top-1/2 z-50 w-[min(94vw,860px)] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-zinc-800 bg-black shadow-2xl outline-none"
          onOpenAutoFocus={(event) => event.preventDefault()}
        >
          <Dialog.Close className="absolute right-3 top-3 z-10 rounded-full bg-black/80 p-2 text-zinc-400 hover:text-white">
            <X className="h-4 w-4" />
          </Dialog.Close>

          {!item && isLoading ? (
            <div className="p-6">
              <Skeleton className="h-[280px] w-full rounded-lg" />
            </div>
          ) : item ? (
            <div className="grid max-h-[min(85vh,640px)] grid-cols-1 md:grid-cols-[200px_1fr]">
              <div className="relative h-[280px] overflow-hidden rounded-t-xl md:h-auto md:rounded-l-xl md:rounded-tr-none">
                <PosterImage item={item} className="absolute inset-0" priority />
              </div>

              <div className="flex flex-col gap-3 overflow-hidden p-5 md:p-6">
                <div>
                  <Dialog.Title className="pr-8 text-xl font-bold text-white">{item.title}</Dialog.Title>
                  <p className="mt-1 text-sm text-zinc-500">
                    {[item.year, ...(item.genres ?? []).slice(0, 3)].filter(Boolean).join(" · ")}
                  </p>
                </div>

                <p className="line-clamp-3 text-sm leading-relaxed text-zinc-300">
                  {item.description ?? "No synopsis available."}
                </p>

                <div className="space-y-1">
                  <p className="text-xs uppercase tracking-wide text-zinc-500">Your rating</p>
                  <StarRating
                    value={status?.rating ?? null}
                    disabled={busy}
                    onChange={(rating) => {
                      rateMovie.mutate({ itemId: item.item_id, rating }, { onSuccess: refresh });
                    }}
                    onClear={() => {
                      clearRating.mutate(item.item_id, { onSuccess: refresh });
                    }}
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant={status?.in_watchlist ? "default" : "secondary"}
                    size="sm"
                    disabled={busy}
                    onClick={() => {
                      if (status?.in_watchlist) {
                        removeWatchlist.mutate(item.item_id, { onSuccess: refresh });
                      } else {
                        watchlist.mutate(item.item_id, { onSuccess: refresh });
                      }
                    }}
                  >
                    {status?.in_watchlist ? (
                      <>
                        <BookmarkX className="h-4 w-4" />
                        Remove watchlist
                      </>
                    ) : (
                      <>
                        <Bookmark className="h-4 w-4" />
                        Watchlist
                      </>
                    )}
                  </Button>
                  <Button
                    variant={status?.in_library ? "default" : "secondary"}
                    size="sm"
                    disabled={busy}
                    onClick={() => {
                      if (status?.in_library) {
                        removeWatched.mutate(item.item_id, { onSuccess: refresh });
                      } else {
                        markWatched.mutate(item.item_id, { onSuccess: refresh });
                      }
                    }}
                  >
                    {status?.in_library ? (
                      <>
                        <X className="h-4 w-4" />
                        Remove watched
                      </>
                    ) : (
                      <>
                        <Check className="h-4 w-4" />
                        Watched it
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={busy || status?.dismissed}
                    className="col-span-2"
                    onClick={() =>
                      dismissMovie.mutate(item.item_id, {
                        onSuccess: () => {
                          refresh();
                          close();
                        },
                      })
                    }
                  >
                    <EyeOff className="h-4 w-4" />
                    {status?.dismissed ? "Removed from recs" : "Not interested"}
                  </Button>
                </div>

                <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-3">
                  <p className="text-xs font-medium uppercase tracking-wide text-red-500">
                    Why we recommended this
                  </p>
                  {explanationLoading ? (
                    <Skeleton className="mt-2 h-10 w-full" />
                  ) : explanation ? (
                    <div className="mt-2 space-y-2">
                      <p className="line-clamp-3 text-sm leading-relaxed text-zinc-200">
                        {explanation.explanation}
                      </p>
                      {explanation.reasons.length > 0 ? (
                        <ul className="space-y-1">
                          {explanation.reasons.slice(0, 2).map((reason) => (
                            <li key={reason} className="line-clamp-2 text-xs text-zinc-500">
                              • {reason}
                            </li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  ) : (
                    <p className="mt-2 text-xs text-zinc-500">
                      Add more movies to unlock personalized explanations.
                    </p>
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
