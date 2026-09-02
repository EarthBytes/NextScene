export interface UserProfile {
  id: number;
  username: string;
}

export interface UserPreferences {
  preferred_genres: string[];
}

export interface GenreCount {
  genre: string;
  count: number;
}

export interface TagCount {
  tag: string;
  count: number;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: UserProfile;
}

export interface LibraryItem {
  item_id: number;
  title: string;
  genres: string[];
  year: number | null;
  poster_url: string | null;
  image_url: string | null;
  added_at: string;
}

export interface RecommendationItem {
  item_id: number;
  title: string | null;
  rank: number;
  genres: string[];
  year: number | null;
  poster_url: string | null;
  image_url: string | null;
}

export interface RecommendationsResponse {
  recommendations: RecommendationItem[];
  library_count: number;
  needs_more_movies: boolean;
  active_genres?: string[];
}

export interface ItemDetail {
  item_id: number;
  title: string;
  description: string | null;
  genres: string[];
  year: number | null;
  image_url: string | null;
  poster_url: string | null;
  imdb_id: string | null;
  metadata_json: Record<string, unknown>;
}

export interface ItemSummary {
  item_id: number;
  title: string;
  genres: string[];
  year: number | null;
  image_url: string | null;
  poster_url: string | null;
}

export interface ItemSearchResponse {
  items: ItemSummary[];
  total: number;
  limit: number;
  offset: number;
  query?: string | null;
  genre?: string | null;
  genres?: string[];
  tag?: string | null;
}

export interface ExplanationResponse {
  item_id: number;
  title: string | null;
  explanation: string;
  related_titles: string[];
  shared_genres: string[];
  reasons: string[];
}

export interface MovieStatus {
  item_id: number;
  in_library: boolean;
  in_watchlist: boolean;
  dismissed: boolean;
  rating: number | null;
}
