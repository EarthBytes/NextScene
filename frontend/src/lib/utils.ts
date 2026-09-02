import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatScore(score: number): string {
  if (score <= 1) {
    return `${Math.round(Math.max(score, 0) * 100)}%`;
  }
  return score.toFixed(0);
}

export function formatMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return `${ms.toFixed(1)}ms`;
}
