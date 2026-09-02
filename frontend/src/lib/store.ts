import { create } from "zustand";
import { persist } from "zustand/middleware";

import { api } from "@/lib/api";
import type { UserProfile } from "@/lib/types";

interface AuthState {
  token: string | null;
  user: UserProfile | null;
  setAuth: (token: string, user: UserProfile) => void;
  setUser: (user: UserProfile) => void;
  logout: () => void;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  refreshProfile: () => Promise<void>;
  deleteAccount: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      setAuth: (token, user) => {
        localStorage.setItem("auth_token", token);
        set({ token, user });
      },
      setUser: (user) => set({ user }),
      logout: () => {
        localStorage.removeItem("auth_token");
        set({ token: null, user: null });
      },
      login: async (username, password) => {
        const response = await api.login(username, password);
        localStorage.setItem("auth_token", response.access_token);
        set({ token: response.access_token, user: response.user });
      },
      register: async (username, password) => {
        const response = await api.register(username, password);
        localStorage.setItem("auth_token", response.access_token);
        set({ token: response.access_token, user: response.user });
      },
      refreshProfile: async () => {
        if (!get().token) return;
        const user = await api.me();
        set({ user });
      },
      deleteAccount: async () => {
        await api.deleteAccount();
        localStorage.removeItem("auth_token");
        set({ token: null, user: null });
      },
    }),
    {
      name: "nextscene-auth",
      partialize: (state) => ({ token: state.token, user: state.user }),
    },
  ),
);

interface UiState {
  selectedItemId: number | null;
  setSelectedItemId: (id: number | null) => void;
  explainItemId: number | null;
  setExplainItemId: (id: number | null) => void;
  searchOpen: boolean;
  setSearchOpen: (open: boolean) => void;
}

export const useUiStore = create<UiState>((set) => ({
  selectedItemId: null,
  setSelectedItemId: (selectedItemId) => set({ selectedItemId }),
  explainItemId: null,
  setExplainItemId: (explainItemId) => set({ explainItemId }),
  searchOpen: false,
  setSearchOpen: (searchOpen) => set({ searchOpen }),
}));
