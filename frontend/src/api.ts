export const getDefaultApiBase = () =>
  import.meta.env.VITE_API_BASE ||
  (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

export const apiFetch = async <T>(
  url: string,
  options: RequestInit = {},
): Promise<T> => {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
};
