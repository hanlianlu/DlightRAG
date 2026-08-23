import {defineConfig, type Plugin} from 'vite';
import {resolve, dirname} from 'path';
import {fileURLToPath} from 'url';
import {readFileSync, writeFileSync} from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outputDirectory = resolve(__dirname, '../src/dlightrag/web/static/app');

function devWebPageFallback(): Plugin {
  return {
    name: 'dlightrag-dev-web-page-fallback',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((request, _response, next) => {
        const pathname = new URL(request.url ?? '/', 'http://vite.local').pathname;
        if (pathname === '/static/app/__THEME_INIT__') {
          request.url = '/static/app/theme-init.ts';
        } else if (pathname === '/web' || pathname === '/web/'
            || /^\/web\/conversations\/[^/]+\/?$/.test(pathname)) {
          request.url = '/static/app/';
        } else if (pathname === '/web/design-system') {
          request.url = '/static/app/design-system.html';
        }
        next();
      });
    },
  };
}

function hashedThemeInit(): Plugin {
  return {
    name: 'dlightrag-hashed-theme-init',
    apply: 'build',
    writeBundle(options, bundle) {
      const theme = Object.values(bundle).find(
        (item) => item.type === 'chunk' && item.name === 'theme-init',
      );
      if (!theme || theme.type !== 'chunk' || !options.dir) {
        throw new Error('Vite did not emit the pre-paint theme entry');
      }
      for (const filename of ['index.html', 'login.html', 'design-system.html']) {
        const path = resolve(options.dir, filename);
        const source = readFileSync(path, 'utf8');
        writeFileSync(path, source.replace('__THEME_INIT__', theme.fileName));
      }
    },
  };
}

export default defineConfig({
  root: '.',
  base: '/static/app/',
  appType: 'spa',
  plugins: [devWebPageFallback(), hashedThemeInit()],
  build: {
    outDir: outputDirectory,
    emptyOutDir: true,
    rollupOptions: {
      input: {
        app: resolve(__dirname, 'index.html'),
        login: resolve(__dirname, 'login.html'),
        'design-system': resolve(__dirname, 'design-system.html'),
        'theme-init': resolve(__dirname, 'theme-init.ts'),
      },
      output: {
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
      external: [],
    },
    cssCodeSplit: false,
    target: 'es2022',
    modulePreload: false,
  },
  server: {
    proxy: {
      '/web/api': 'http://127.0.0.1:8100',
      '/web/login': 'http://127.0.0.1:8100',
      '/web/logout': 'http://127.0.0.1:8100',
      '/static/vendor': 'http://127.0.0.1:8100',
      '/static/pygments.css': 'http://127.0.0.1:8100',
    },
  },
  css: {
    postcss: {
      plugins: [],
    },
  },
});
