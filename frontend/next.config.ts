import type { NextConfig } from "next";

// Server-only: used for rewrites and image remote patterns (not exposed to the browser).
const apiUrl =
  process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function apiPosterPattern() {
  try {
    const url = new URL(apiUrl);
    return {
      protocol: url.protocol.replace(":", "") as "http" | "https",
      hostname: url.hostname,
      ...(url.port ? { port: url.port } : {}),
      pathname: "/posters/**",
    };
  } catch {
    return null;
  }
}

const apiPattern = apiPosterPattern();

const nextConfig: NextConfig = {
  // Standalone is for Docker self-hosting; Next 16.3 + Vercel's adapter breaks on it.
  ...(process.env.VERCEL ? {} : { output: "standalone" as const }),
  images: {
    remotePatterns: [
      ...(apiPattern ? [apiPattern] : []),
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/posters/**",
      },
      {
        protocol: "https",
        hostname: "image.tmdb.org",
        pathname: "/t/p/**",
      },
      {
        protocol: "https",
        hostname: "m.media-amazon.com",
        pathname: "/**",
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
