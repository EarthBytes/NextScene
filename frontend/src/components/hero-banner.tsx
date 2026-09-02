"use client";

import { PosterImage } from "@/components/poster-image";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { RecommendationItem } from "@/lib/types";

interface HeroBannerProps {
  item?: RecommendationItem;
  loading?: boolean;
  refreshing?: boolean;
  onOpen: (item: RecommendationItem) => void;
}

export function HeroBanner({ item, loading, refreshing, onOpen }: HeroBannerProps) {
  if (loading) {
    return (
      <section className="mx-4 overflow-hidden rounded-xl border border-zinc-800 sm:mx-8">
        <Skeleton className="h-[300px] w-full sm:h-[380px]" />
      </section>
    );
  }

  if (!item) return null;

  return (
    <section
      className={`relative mx-4 overflow-hidden rounded-xl border border-zinc-800 bg-black transition-opacity duration-200 sm:mx-8 ${
        refreshing ? "opacity-80" : "opacity-100"
      }`}
    >
      <div className="absolute inset-0 opacity-20">
        <PosterImage item={item} className="h-full w-full blur-md" />
      </div>
      <div className="absolute inset-0 bg-gradient-to-r from-black via-black/95 to-transparent" />

      <div className="relative grid gap-6 p-6 sm:grid-cols-[180px_1fr] sm:p-10">
        <div className="relative mx-auto aspect-[2/3] w-[150px] overflow-hidden rounded-lg border border-zinc-800 sm:mx-0 sm:w-full">
          <PosterImage item={item} className="absolute inset-0" priority />
        </div>
        <div className="flex flex-col justify-center">
          <p className="text-sm font-medium uppercase tracking-wider text-red-500">Top pick for you</p>
          <h1 className="mt-2 text-3xl font-bold text-white sm:text-4xl">{item.title}</h1>
          <p className="mt-3 text-sm text-zinc-400">
            {[item.year, ...(item.genres ?? []).slice(0, 3)].filter(Boolean).join(" · ")}
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button onClick={() => onOpen(item)}>View details</Button>
          </div>
        </div>
      </div>
    </section>
  );
}
