import { useEffect, useState } from "react";

export function useStoredState<T>(
  key: string,
  initial: T,
  parser: (raw: string) => T = (raw) => JSON.parse(raw) as T,
  serializer: (val: T) => string = (val) => JSON.stringify(val),
) {
  const [value, setValue] = useState<T>(() => {
    if (typeof window === "undefined") return initial;
    const saved = localStorage.getItem(key);
    if (saved == null) return initial;
    try {
      return parser(saved);
    } catch {
      return initial;
    }
  });

  useEffect(() => {
    if (typeof window === "undefined") return;
    localStorage.setItem(key, serializer(value));
  }, [key, serializer, value]);

  return [value, setValue] as const;
}
