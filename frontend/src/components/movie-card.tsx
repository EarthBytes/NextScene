"use client";

import { Plus } from "lucide-react";

import { PosterImage } from "@/components/poster-image";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import type { RecommendationItem } from "@/lib/types";
import { cn } from "@/lib/utils";

interface MovieCardProps {
  item: RecommendationItem | { item_id: number; title: string; genres?: string[]; year?: number | null; poster_url?: string | null; image_url?: string | null; rank?: number };
  onOpen: (item: MovieCardProps["item"]) => void;
  onAdd?: (itemId: number) => void;
  showAdd?: boolean;
  adding?: boolean;
  className?: string;
}

export function MovieCard({ item, onOpen, onAdd, showAdd, adding, className }: MovieCardProps) {
  return (
    <article
      className={cn("group relative w-[160px] shrink-0 cursor-pointer snap-start sm:w-[180px]", className)}
      onClick={() => onOpen(item)}
    >
      <div className="relative aspect-[2/3] overflow-hidden rounded-lg border border-zinc-800 bg-black transition group-hover:border-red-600">
        <PosterImage item={item} className="absolute inset-0" />
        {"rank" in item && item.rank === 1 ? (
          <span className="absolute left-2 top-2 rounded bg-red-600 px-2 py-0.5 text-xs font-semibold text-white">
            Top pick
          </span>
        ) : null}
      </div>
      <div className="mt-2 space-y-1">
        <h3 className="line-clamp-2 text-sm font-medium text-white">{item.title}</h3>
        <p className="text-xs text-zinc-500">
          {[item.year, ...(item.genres ?? []).slice(0, 2)].filter(Boolean).join(" · ")}
        </p>
      </div>
      <div className="mt-2 flex gap-2 opacity-0 transition group-hover:opacity-100">
        {showAdd && onAdd ? (
          <Button
            variant="secondary"
            size="sm"
            className="h-7 flex-1 text-xs"
            disabled={adding}
            onClick={(e) => {
              e.stopPropagation();
              onAdd(item.item_id);
            }}
          >
            <Plus className="h-3 w-3" />
            Add
          </Button>
        ) : null}
      </div>
    </article>
  );
}

export function MovieCardSkeleton() {
  return (
    <div className="w-[160px] shrink-0 sm:w-[180px]">
      <Skeleton className="aspect-[2/3] w-full rounded-lg" />
      <Skeleton className="mt-2 h-4 w-3/4" />
    </div>
  );
}
