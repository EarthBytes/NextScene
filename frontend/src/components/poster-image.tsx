"use client";

import { useState } from "react";

import { posterSrc } from "@/lib/api";
import { cn } from "@/lib/utils";

interface PosterImageProps {
  item: { title?: string | null; poster_url?: string | null; image_url?: string | null };
  className?: string;
  priority?: boolean;
}

export function PosterImage({ item, className, priority }: PosterImageProps) {
  const initial = posterSrc(item);
  const [src, setSrc] = useState<string | null>(initial);
  const [failed, setFailed] = useState(false);

  if (!src || failed) {
    return (
      <div
        className={cn(
          "flex items-center justify-center bg-gradient-to-br from-zinc-900 to-black p-4 text-center text-sm text-zinc-500",
          className,
        )}
      >
        {item.title ?? "No poster"}
      </div>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={item.title ?? "Movie poster"}
      className={cn("h-full w-full object-cover", className)}
      loading={priority ? "eager" : "lazy"}
      onError={() => {
        if (item.image_url && src !== item.image_url) {
          setSrc(item.image_url);
          return;
        }
        setFailed(true);
      }}
    />
  );
}
