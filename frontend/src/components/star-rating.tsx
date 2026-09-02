"use client";

import { Star } from "lucide-react";

import { cn } from "@/lib/utils";

interface StarRatingProps {
  value?: number | null;
  onChange: (rating: number) => void;
  onClear?: () => void;
  disabled?: boolean;
}

export function StarRating({ value, onChange, onClear, disabled }: StarRatingProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex gap-1">
        {Array.from({ length: 5 }).map((_, index) => {
          const rating = index + 1;
          const active = value != null && rating <= value;
          return (
            <button
              key={rating}
              type="button"
              tabIndex={-1}
              disabled={disabled}
              onClick={(event) => {
                event.stopPropagation();
                if (value === rating && onClear) {
                  onClear();
                } else {
                  onChange(rating);
                }
              }}
              className={cn(
                "rounded p-0.5 transition hover:scale-110 disabled:opacity-50",
                active ? "text-red-500" : "text-zinc-600 hover:text-red-400",
              )}
              aria-label={`Rate ${rating} stars`}
            >
              <Star className={cn("h-5 w-5", active && "fill-current")} />
            </button>
          );
        })}
      </div>
      {value != null ? (
        <span className="text-xs text-zinc-500">{value}/5</span>
      ) : (
        <span className="text-xs text-zinc-600">Not rated</span>
      )}
    </div>
  );
}
