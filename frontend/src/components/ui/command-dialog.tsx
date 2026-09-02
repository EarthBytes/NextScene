"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";

import { cn } from "@/lib/utils";

export const CommandDialog = Dialog.Root;

export function CommandDialogContent({
  children,
  title = "Search",
}: {
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" />
      <Dialog.Content
        className={cn(
          "fixed left-1/2 top-[15%] z-50 w-[min(92vw,640px)] -translate-x-1/2 rounded-2xl border border-white/10 bg-zinc-950 shadow-2xl outline-none",
        )}
      >
        <Dialog.Title className="sr-only">{title}</Dialog.Title>
        <Dialog.Close className="absolute right-4 top-4 rounded-md p-2 text-zinc-400 hover:bg-white/10 hover:text-white">
          <X className="h-4 w-4" />
        </Dialog.Close>
        {children}
      </Dialog.Content>
    </Dialog.Portal>
  );
}
