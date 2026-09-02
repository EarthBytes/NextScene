"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Trash2, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/lib/store";

export function AccountMenu() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const deleteAccount = useAuthStore((s) => s.deleteAccount);
  const logout = useAuthStore((s) => s.logout);

  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const close = () => {
    setOpen(false);
    setConfirming(false);
    setError(null);
  };

  const handleDelete = async () => {
    setDeleting(true);
    setError(null);
    try {
      await deleteAccount();
      close();
      router.replace("/login");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete account");
    } finally {
      setDeleting(false);
    }
  };

  if (!user) return null;

  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) {
          setConfirming(false);
          setError(null);
        }
      }}
    >
      <Dialog.Trigger asChild>
        <button
          type="button"
          className="rounded-md px-2 py-1 text-sm text-zinc-400 transition-colors hover:bg-zinc-900 hover:text-white"
        >
          {user.username}
        </button>
      </Dialog.Trigger>

      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-[min(94vw,400px)] -translate-x-1/2 -translate-y-1/2 rounded-xl border border-zinc-800 bg-black p-6 shadow-2xl outline-none">
          <div className="flex items-start justify-between gap-4">
            <div>
              <Dialog.Title className="text-lg font-semibold text-white">Account</Dialog.Title>
              <Dialog.Description className="mt-1 text-sm text-zinc-500">
                Signed in as {user.username}
              </Dialog.Description>
            </div>
            <Dialog.Close className="rounded-full p-1 text-zinc-500 hover:text-white">
              <X className="h-4 w-4" />
            </Dialog.Close>
          </div>

          {confirming ? (
            <div className="mt-6 space-y-4">
              <p className="text-sm text-zinc-300">
                Delete your account permanently? Your library, watchlist, and ratings will be removed.
                This cannot be undone.
              </p>
              {error ? <p className="text-sm text-red-400">{error}</p> : null}
              <div className="flex gap-2">
                <Button
                  variant="secondary"
                  className="flex-1"
                  disabled={deleting}
                  onClick={() => {
                    setConfirming(false);
                    setError(null);
                  }}
                >
                  Cancel
                </Button>
                <Button
                  className="flex-1 bg-red-700 hover:bg-red-600"
                  disabled={deleting}
                  onClick={() => void handleDelete()}
                >
                  {deleting ? "Deleting…" : "Delete account"}
                </Button>
              </div>
            </div>
          ) : (
            <div className="mt-6 space-y-2">
              <Button
                variant="outline"
                className="w-full justify-start text-red-400 hover:text-red-300"
                onClick={() => setConfirming(true)}
              >
                <Trash2 className="h-4 w-4" />
                Delete account
              </Button>
              <Button
                variant="ghost"
                className="w-full"
                onClick={() => {
                  logout();
                  close();
                  router.replace("/login");
                }}
              >
                Log out
              </Button>
            </div>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
