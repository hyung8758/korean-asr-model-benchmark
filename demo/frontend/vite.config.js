import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import { defineConfig } from 'vite';

function httpsOptions() {
  if (process.env.DEMO_SSL_ENABLED !== '1') {
    return undefined;
  }
  return {
    cert: fs.readFileSync(process.env.DEMO_SSL_CERT_FILE),
    key: fs.readFileSync(process.env.DEMO_SSL_KEY_FILE),
  };
}

function backendProxyTarget() {
  return process.env.VITE_BACKEND_TARGET || 'http://127.0.0.1:16000';
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 16010,
    https: httpsOptions(),
    proxy: {
      '/api': {
        target: backendProxyTarget(),
        changeOrigin: true,
        secure: false,
        ws: true,
      },
    },
  },
});
