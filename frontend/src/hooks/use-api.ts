"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { RecommendationsResponse, UserPreferences } from "@/lib/types";
import { useAuthStore } from "@/lib/store";

export function useRecommendations(k = 12, genres?: string[]) {
  const token = useAuthStore((s) => s.token);
  const genreKey = (genres ?? []).join(",");
  return useQuery({
    queryKey: ["recommendations", k, genreKey],
    queryFn: () => api.recommendations(k, genres?.length ? genres : undefined),
    enabled: Boolean(token),
    placeholderData: keepPreviousData,
  });
}

export function useMyMovies() {
  const token = useAuthStore((s) => s.token);
  return useQuery({
    queryKey: ["my-movies"],
    queryFn: () => api.myMovies(),
    enabled: Boolean(token),
  });
}

export function useWatchlistMovies() {
  const token = useAuthStore((s) => s.token);
  return useQuery({
    queryKey: ["watchlist"],
    queryFn: () => api.watchlist(),
    enabled: Boolean(token),
  });
}

export function useItemSearch(params: { q?: string; genre?: string; tag?: string }) {
  const hasQuery = (params.q?.trim().length ?? 0) >= 2;
  const hasGenre = Boolean(params.genre?.trim());
  const hasTag = Boolean(params.tag?.trim());
  return useQuery({
    queryKey: ["search", params.q ?? "", params.genre ?? "", params.tag ?? ""],
    queryFn: () =>
      api.searchItems({
        q: params.q,
        genre: params.genre,
        tag: params.tag,
        limit: 12,
      }),
    enabled: hasQuery || hasGenre || hasTag,
  });
}

export function useGenres() {
  return useQuery({
    queryKey: ["genres"],
    queryFn: () => api.genres(),
  });
}

export function useTags() {
  return useQuery({
    queryKey: ["tags"],
    queryFn: () => api.tags(20),
  });
}

export function usePreferences() {
  const token = useAuthStore((s) => s.token);
  return useQuery({
    queryKey: ["preferences"],
    queryFn: () => api.preferences(),
    enabled: Boolean(token),
  });
}

export function useUpdatePreferences() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (preferred_genres: string[]) => {
      return api.updatePreferences(preferred_genres);
    },
    onMutate: async (preferred_genres) => {
      await queryClient.cancelQueries({ queryKey: ["preferences"] });
      const previous = queryClient.getQueryData<UserPreferences>(["preferences"]);
      queryClient.setQueryData(["preferences"], { preferred_genres });
      return { previous };
    },
    onError: (_error, _genres, context) => {
      if (context?.previous) {
        queryClient.setQueryData(["preferences"], context.previous);
      }
    },
    onSettled: async () => {
      await queryClient.invalidateQueries({ queryKey: ["preferences"] });
      await queryClient.invalidateQueries({ queryKey: ["recommendations"] });
    },
  });
}

export function useExplanation(itemId: number | null) {
  const token = useAuthStore((s) => s.token);
  return useQuery({
    queryKey: ["explanation", itemId],
    queryFn: () => api.explanation(itemId!),
    enabled: Boolean(token) && itemId != null,
  });
}

export function useItem(itemId: number | null) {
  return useQuery({
    queryKey: ["item", itemId],
    queryFn: () => api.item(itemId!),
    enabled: itemId != null,
  });
}

export function useMovieStatus(itemId: number | null) {
  const token = useAuthStore((s) => s.token);
  return useQuery({
    queryKey: ["movie-status", itemId],
    queryFn: () => api.movieStatus(itemId!),
    enabled: Boolean(token) && itemId != null,
  });
}

function removeFromRecommendationsCache(
  queryClient: ReturnType<typeof useQueryClient>,
  itemId: number,
) {
  queryClient.setQueriesData<RecommendationsResponse>(
    { queryKey: ["recommendations"] },
    (old) => {
      if (!old) return old;
      const filtered = old.recommendations.filter((rec) => rec.item_id !== itemId);
      return {
        ...old,
        recommendations: filtered.map((rec, index) => ({ ...rec, rank: index + 1 })),
      };
    },
  );
}

async function refreshFeed(queryClient: ReturnType<typeof useQueryClient>) {
  await Promise.all([
    queryClient.refetchQueries({ queryKey: ["recommendations"], type: "active" }),
    queryClient.refetchQueries({ queryKey: ["my-movies"], type: "active" }),
    queryClient.refetchQueries({ queryKey: ["watchlist"], type: "active" }),
  ]);
}

function invalidateMovieQueries(queryClient: ReturnType<typeof useQueryClient>) {
  queryClient.invalidateQueries({ queryKey: ["my-movies"] });
  queryClient.invalidateQueries({ queryKey: ["watchlist"] });
  queryClient.invalidateQueries({ queryKey: ["recommendations"] });
  queryClient.invalidateQueries({ queryKey: ["movie-status"] });
}

async function afterMovieMutation(
  queryClient: ReturnType<typeof useQueryClient>,
  itemId?: number | null,
) {
  if (itemId != null) {
    removeFromRecommendationsCache(queryClient, itemId);
  }
  invalidateMovieQueries(queryClient);
  await refreshFeed(queryClient);
}

function useFeedMutation<TArgs, TResult>(
  mutationFn: (args: TArgs) => Promise<TResult>,
  options?: { removeFromFeed?: (args: TArgs) => number | null },
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (args: TArgs) => {
      const itemId = options?.removeFromFeed?.(args) ?? null;
      if (itemId != null) {
        removeFromRecommendationsCache(queryClient, itemId);
      }
      try {
        const result = await mutationFn(args);
        invalidateMovieQueries(queryClient);
        await refreshFeed(queryClient);
        return result;
      } catch (error) {
        invalidateMovieQueries(queryClient);
        await refreshFeed(queryClient);
        throw error;
      }
    },
  });
}

export function useAddMovie() {
  return useFeedMutation((itemId: number) => api.addMovie(itemId));
}

export function useWatchlist() {
  return useFeedMutation((itemId: number) => api.addToWatchlist(itemId));
}

export function useMarkWatched() {
  return useFeedMutation((itemId: number) => api.markWatched(itemId), {
    removeFromFeed: (itemId) => itemId,
  });
}

export function useRateMovie() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ itemId, rating }: { itemId: number; rating: number }) => {
      const result = await api.rateMovie(itemId, rating);
      await afterMovieMutation(queryClient);
      return result;
    },
  });
}

export function useDismissMovie() {
  return useFeedMutation((itemId: number) => api.dismissMovie(itemId), {
    removeFromFeed: (itemId) => itemId,
  });
}

export function useRemoveWatchlist() {
  return useFeedMutation((itemId: number) => api.removeFromWatchlist(itemId));
}

export function useRemoveWatched() {
  return useFeedMutation((itemId: number) => api.removeFromWatched(itemId));
}

export function useClearRating() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (itemId: number) => {
      const result = await api.clearRating(itemId);
      await afterMovieMutation(queryClient);
      return result;
    },
  });
}
