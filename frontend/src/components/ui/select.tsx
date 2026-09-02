"use client";

import * as Select from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";

export function SelectRoot({ children, value, onValueChange }: {
  children: React.ReactNode;
  value: string;
  onValueChange: (value: string) => void;
}) {
  return (
    <Select.Root value={value} onValueChange={onValueChange}>
      {children}
    </Select.Root>
  );
}

export function SelectTrigger({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <Select.Trigger
      className={cn(
        "flex h-10 items-center justify-between gap-2 rounded-lg border border-white/10 bg-white/5 px-3 text-sm text-white outline-none hover:bg-white/10",
        className,
      )}
    >
      {children}
      <Select.Icon>
        <ChevronDown className="h-4 w-4 opacity-60" />
      </Select.Icon>
    </Select.Trigger>
  );
}

export const SelectValue = Select.Value;

export function SelectContent({ children }: { children: React.ReactNode }) {
  return (
    <Select.Portal>
      <Select.Content className="z-50 overflow-hidden rounded-lg border border-white/10 bg-zinc-900 shadow-xl">
        <Select.Viewport className="p-1">{children}</Select.Viewport>
      </Select.Content>
    </Select.Portal>
  );
}

export function SelectItem({ value, children }: { value: string; children: React.ReactNode }) {
  return (
    <Select.Item
      value={value}
      className="relative flex cursor-pointer select-none items-center rounded-md px-8 py-2 text-sm text-zinc-200 outline-none data-[highlighted]:bg-white/10"
    >
      <Select.ItemIndicator className="absolute left-2 inline-flex items-center">
        <Check className="h-4 w-4" />
      </Select.ItemIndicator>
      <Select.ItemText>{children}</Select.ItemText>
    </Select.Item>
  );
}
