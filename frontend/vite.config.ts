import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  const https =
    env.HTTPS === "true"
      ? (() => {
          const cert = env.SSL_CRT_FILE ? fs.readFileSync(path.resolve(env.SSL_CRT_FILE)) : undefined;
          const key = env.SSL_KEY_FILE ? fs.readFileSync(path.resolve(env.SSL_KEY_FILE)) : undefined;
          if (cert && key) return { cert, key };
          return true; // self-signed by Vite
        })()
      : false;

  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: true,
      https,
    },
  };
});
