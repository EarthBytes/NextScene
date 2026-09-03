"use client";

import { LogOut, Plus, Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { AccountMenu } from "@/components/account-menu";
import { Logo } from "@/components/logo";
import { GenrePreferences } from "@/components/genre-preferences";
import { HeroBanner } from "@/components/hero-banner";
import { MovieCarousel } from "@/components/movie-carousel";
import { MovieDetailModal } from "@/components/movie-detail-modal";
import { SearchCommand } from "@/components/search-command";
import { Button } from "@/components/ui/button";
import { useGenres, useMyMovies, usePreferences, useRecommendations, useWatchlistMovies } from "@/hooks/use-api";
import { useAuthStore, useUiStore } from "@/lib/store";

export function HomePage() {
  const router = useRouter();
  const token = useAuthStore((s) => s.token);
  const refreshProfile = useAuthStore((s) => s.refreshProfile);
  const logout = () => {
    useAuthStore.getState().logout();
    router.replace("/login");
  };
  const setSearchOpen = useUiStore((s) => s.setSearchOpen);
  const setSelectedItemId = useUiStore((s) => s.setSelectedItemId);

  const preferences = usePreferences();
  const { data: genreCatalog } = useGenres();
  const availableGenres = new Set(genreCatalog?.map((entry) => entry.genre) ?? []);
  const preferredGenres = (preferences.data?.preferred_genres ?? []).filter((genre) =>
    availableGenres.has(genre),
  );
  const recommendations = useRecommendations(12, preferredGenres);
  const myMovies = useMyMovies();
  const watchlist = useWatchlistMovies();

  useEffect(() => {
    if (!token) router.replace("/login");
  }, [token, router]);

  useEffect(() => {
    if (token) void refreshProfile();
  }, [token, refreshProfile]);

  if (!token) return null;

  const topPick = recommendations.data?.recommendations[0];
  const rest = recommendations.data?.recommendations.slice(1) ?? [];
  const libraryCount = myMovies.data?.length ?? 0;
  const needsMore = recommendations.data?.needs_more_movies ?? libraryCount < 3;

  const showRecommendationsLoading = recommendations.isLoading && !recommendations.data;
  const recommendationsRefreshing = recommendations.isFetching && !!recommendations.data;

  return (
    <div className="min-h-screen bg-black text-white">
      <header className="border-b border-zinc-900 bg-black">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-4 sm:px-8">
          <Logo href="/" />
          <div className="flex items-center gap-2">
            <AccountMenu />
            <Button variant="ghost" size="sm" onClick={() => setSearchOpen(true)}>
              <Search className="h-4 w-4" />
              <span className="hidden sm:inline">Add movies</span>
            </Button>
            <Button variant="ghost" size="icon" onClick={logout} aria-label="Log out">
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-10 py-8">
        {libraryCount === 0 ? (
          <section className="mx-4 rounded-xl border border-zinc-800 bg-zinc-950 p-8 text-center sm:mx-8">
            <h2 className="text-2xl font-semibold text-white">Build your movie library</h2>
            <p className="mx-auto mt-3 max-w-lg text-sm text-zinc-400">
              Search and add movies you&apos;ve watched. Once you&apos;ve added a few, we&apos;ll predict
              what you&apos;ll enjoy next.
            </p>
            <Button className="mt-6" onClick={() => setSearchOpen(true)}>
              <Plus className="h-4 w-4" />
              Add your first movie
            </Button>
          </section>
        ) : needsMore ? (
          <section className="mx-4 rounded-xl border border-red-900/50 bg-red-950/20 p-6 sm:mx-8">
            <p className="text-sm text-zinc-300">
              Add {3 - libraryCount} more movie{3 - libraryCount === 1 ? "" : "s"} to unlock
              personalized recommendations.
            </p>
            <Button variant="secondary" size="sm" className="mt-3" onClick={() => setSearchOpen(true)}>
              Add more movies
            </Button>
          </section>
        ) : null}

        <GenrePreferences />

        {recommendations.isError ? (
          <div className="mx-4 rounded-xl border border-red-900 bg-red-950/30 p-6 text-sm text-red-300 sm:mx-8">
            Could not load recommendations.
            {recommendations.error instanceof Error && recommendations.error.message
              ? ` ${recommendations.error.message}`
              : " Check that the backend is running and you have at least 3 movies in your library."}
          </div>
        ) : null}

        {topPick || showRecommendationsLoading ? (
          <HeroBanner
            key={topPick?.item_id ?? "loading"}
            item={topPick}
            loading={showRecommendationsLoading}
            refreshing={recommendationsRefreshing}
            onOpen={(item) => setSelectedItemId(item.item_id)}
          />
        ) : null}

        <MovieCarousel
          title={
            preferredGenres.length > 0
              ? `Recommended in ${preferredGenres.join(", ")}`
              : "Recommended for you"
          }
          subtitle={needsMore ? "Add more movies for better picks" : undefined}
          items={rest}
          loading={showRecommendationsLoading}
          refreshing={recommendationsRefreshing}
          onOpen={(item) => setSelectedItemId(item.item_id)}
        />

        <MovieCarousel
          title="Your watchlist"
          subtitle={`${watchlist.data?.length ?? 0} saved`}
          items={watchlist.data?.map((m) => ({ ...m, rank: 0 }))}
          loading={watchlist.isLoading}
          onOpen={(item) => setSelectedItemId(item.item_id)}
        />

        <MovieCarousel
          title="Your movies"
          subtitle={`${libraryCount} in your library`}
          items={myMovies.data?.map((m) => ({ ...m, rank: 0 }))}
          loading={myMovies.isLoading}
          onOpen={(item) => setSelectedItemId(item.item_id)}
        />
      </main>

      <MovieDetailModal />
      <SearchCommand />
    </div>
  );
}
