import { defineConfig } from 'vite';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: '.',
  build: {
    outDir: resolve(__dirname, '../src/dlightrag/web/static'),
    emptyOutDir: false,
    rollupOptions: {
      input: resolve(__dirname, 'ui/main.ts'),
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
    cssCodeSplit: false,
    target: 'es2022',
    modulePreload: false,
  },
  css: {
    postcss: {
      plugins: [],
    },
  },
});
