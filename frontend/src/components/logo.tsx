import Link from "next/link";

import { cn } from "@/lib/utils";

interface LogoProps {
  className?: string;
  href?: string;
  showWordmark?: boolean;
}

export function Logo({ className, href = "/", showWordmark = true }: LogoProps) {
  const content = (
    <span className={cn("inline-flex items-center gap-2", className)}>
      <span className="relative flex h-8 w-8 items-center justify-center rounded-lg border border-red-600/80 bg-zinc-950">
        <svg viewBox="0 0 24 24" className="h-4 w-4 text-red-600" aria-hidden="true">
          <path fill="currentColor" d="M8 5v14l11-7z" />
        </svg>
      </span>
      {showWordmark ? (
        <span className="text-lg font-bold tracking-tight">
          <span className="text-white">Next</span>
          <span className="text-red-600">Scene</span>
        </span>
      ) : null}
    </span>
  );

  if (href) {
    return (
      <Link href={href} className="inline-flex">
        {content}
      </Link>
    );
  }

  return content;
}
