"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { useRef } from "react";

import { MovieCard, MovieCardSkeleton } from "@/components/movie-card";
import { Button } from "@/components/ui/button";

interface MovieCarouselProps {
  title: string;
  subtitle?: string;
  items?: Array<Parameters<typeof MovieCard>[0]["item"]>;
  loading?: boolean;
  refreshing?: boolean;
  onOpen: Parameters<typeof MovieCard>[0]["onOpen"];
  onAdd?: (itemId: number) => void;
  showAdd?: boolean;
  addingId?: number | null;
}

export function MovieCarousel({
  title,
  subtitle,
  items,
  loading,
  refreshing,
  onOpen,
  onAdd,
  showAdd,
  addingId,
}: MovieCarouselProps) {
  const scrollerRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: "left" | "right") => {
    const node = scrollerRef.current;
    if (!node) return;
    node.scrollBy({ left: direction === "left" ? -node.clientWidth * 0.8 : node.clientWidth * 0.8, behavior: "smooth" });
  };

  if (!loading && (!items || items.length === 0)) return null;

  const showSkeletons = loading && (!items || items.length === 0);

  return (
    <section className={`space-y-4 transition-opacity duration-200 ${refreshing ? "opacity-80" : "opacity-100"}`}>
      <div className="flex items-end justify-between gap-4 px-4 sm:px-8">
        <div>
          <h2 className="text-xl font-semibold text-white">{title}</h2>
          {subtitle ? <p className="mt-1 text-sm text-zinc-500">{subtitle}</p> : null}
        </div>
        <div className="hidden gap-2 sm:flex">
          <Button variant="secondary" size="icon" onClick={() => scroll("left")} aria-label="Scroll left">
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button variant="secondary" size="icon" onClick={() => scroll("right")} aria-label="Scroll right">
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <div ref={scrollerRef} className="no-scrollbar flex gap-4 overflow-x-auto px-4 pb-2 sm:px-8 snap-x snap-mandatory">
        {showSkeletons
          ? Array.from({ length: 6 }).map((_, i) => <MovieCardSkeleton key={i} />)
          : items?.map((item) => (
              <MovieCard
                key={item.item_id}
                item={item}
                onOpen={onOpen}
                onAdd={onAdd}
                showAdd={showAdd}
                adding={addingId === item.item_id}
              />
            ))}
      </div>
    </section>
  );
}
