import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',
  build: {
    outDir: resolve(__dirname, '../src/dlightrag/web/static'),
    emptyOutDir: false,
    rollupOptions: {
      input: resolve(__dirname, 'main.ts'),
      output: {
        entryFileNames: 'js/[name].js',
        chunkFileNames: 'js/[name]-[hash].js',
        assetFileNames: (assetInfo) => {
          if (assetInfo.name?.endsWith('.css')) {
            return '[name].[ext]';
          }
          return '[name]-[hash].[ext]';
        },
      },
      external: [],
    },
    target: 'es2022',
    modulePreload: false,
  },
  css: {
    postcss: {
      plugins: [],
    },
  },
});
