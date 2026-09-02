import type {
  AuthResponse,
  ExplanationResponse,
  GenreCount,
  ItemDetail,
  ItemSearchResponse,
  LibraryItem,
  MovieStatus,
  RecommendationsResponse,
  TagCount,
  UserPreferences,
  UserProfile,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("auth_token");
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // ignore
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export function posterSrc(item: {
  poster_url?: string | null;
  image_url?: string | null;
}): string | null {
  if (item.poster_url) {
    if (item.poster_url.startsWith("http")) return item.poster_url;
    return `${API_BASE}${item.poster_url}`;
  }
  if (item.image_url) return item.image_url;
  return null;
}

export const api = {
  register: (username: string, password: string) =>
    request<AuthResponse>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  login: (username: string, password: string) =>
    request<AuthResponse>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  me: () => request<UserProfile>("/api/auth/me"),

  deleteAccount: () => request<void>("/api/auth/me", { method: "DELETE" }),

  recommendations: (k = 12, genres?: string[]) => {
    const search = new URLSearchParams({ k: String(k) });
    for (const genre of genres ?? []) {
      search.append("genres", genre);
    }
    return request<RecommendationsResponse>(`/api/me/recommendations?${search}`);
  },

  preferences: () => request<UserPreferences>("/api/me/preferences"),

  updatePreferences: (preferred_genres: string[]) =>
    request<UserPreferences>("/api/me/preferences", {
      method: "PUT",
      body: JSON.stringify({ preferred_genres }),
    }),

  genres: () => request<GenreCount[]>("/api/items/genres"),

  tags: (limit = 20) => request<TagCount[]>(`/api/items/tags?limit=${limit}`),

  myMovies: () => request<LibraryItem[]>("/api/me/movies"),

  addMovie: (item_id: number) =>
    request<LibraryItem>("/api/me/movies", {
      method: "POST",
      body: JSON.stringify({ item_id }),
    }),

  item: (itemId: number) => request<ItemDetail>(`/api/items/${itemId}`),

  searchItems: (params: { q?: string; genre?: string; genres?: string[]; tag?: string; limit?: number }) => {
    const search = new URLSearchParams();
    if (params.q) search.set("q", params.q);
    if (params.genre) search.set("genre", params.genre);
    for (const genre of params.genres ?? []) {
      search.append("genres", genre);
    }
    if (params.tag) search.set("tag", params.tag);
    if (params.limit) search.set("limit", String(params.limit));
    return request<ItemSearchResponse>(`/api/items/search?${search}`);
  },

  explanation: (itemId: number) =>
    request<ExplanationResponse>(`/api/me/explanations/${itemId}`),

  movieStatus: (itemId: number) =>
    request<MovieStatus>(`/api/me/movies/${itemId}/status`),

  watchlist: () => request<LibraryItem[]>("/api/me/watchlist"),

  addToWatchlist: (item_id: number) =>
    request<LibraryItem>("/api/me/watchlist", {
      method: "POST",
      body: JSON.stringify({ item_id }),
    }),

  markWatched: (item_id: number) =>
    request<LibraryItem>("/api/me/watched", {
      method: "POST",
      body: JSON.stringify({ item_id }),
    }),

  rateMovie: (item_id: number, rating: number) =>
    request<MovieStatus>("/api/me/ratings", {
      method: "POST",
      body: JSON.stringify({ item_id, rating }),
    }),

  dismissMovie: (item_id: number) =>
    request<MovieStatus>("/api/me/dismissed", {
      method: "POST",
      body: JSON.stringify({ item_id }),
    }),

  removeFromWatchlist: (item_id: number) =>
    request<MovieStatus>(`/api/me/watchlist/${item_id}`, { method: "DELETE" }),

  removeFromWatched: (item_id: number) =>
    request<MovieStatus>(`/api/me/watched/${item_id}`, { method: "DELETE" }),

  clearRating: (item_id: number) =>
    request<MovieStatus>(`/api/me/ratings/${item_id}`, { method: "DELETE" }),
};

export { ApiError, API_BASE, getToken };
