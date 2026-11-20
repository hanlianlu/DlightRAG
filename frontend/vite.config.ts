import {defineConfig, type Plugin} from 'vite';
import {resolve, dirname, basename, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync} from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outputDirectory = resolve(__dirname, '../src/dlightrag/adapters/http/browser/static/app');
// Where a build's source maps are kept: never in the served directory, always beside the sources
// that produced them (gitignored). A reported stack names a hashed asset; its map is
// .sourcemaps/<build>/assets/<asset>.map. The newest builds are kept, enough to symbolicate a stack from
// a page that has not been reloaded since a few deploys ago.
const sourcemapDirectory = resolve(__dirname, '.sourcemaps');
const KEPT_BUILDS = 10;

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

/**
 * Finishes a build's output directory, whichever one it is (`options.dir`, so `--outDir` works):
 * every HTML page names the hashed pre-paint theme script, and the source maps leave the served
 * directory for `.sourcemaps/<build>`, named for the application entry's hash. A map names its sources
 * relative to where it is written, so each is rewritten for where it now lives; the oldest builds go.
 */
function finishBuildOutput(): Plugin {
  return {
    name: 'dlightrag-finish-build-output',
    apply: 'build',
    writeBundle(options, bundle) {
      const directory = options.dir;
      const chunk = (name: string) => Object.values(bundle).find(
        (item) => item.type === 'chunk' && item.isEntry && item.name === name,
      );
      const theme = chunk('theme-init');
      const application = chunk('app');
      if (!directory || theme?.type !== 'chunk' || application?.type !== 'chunk') {
        throw new Error('Vite did not emit the application and pre-paint theme entries');
      }

      const pages = Object.keys(bundle).filter((fileName) => fileName.endsWith('.html'));
      if (pages.length === 0) throw new Error('Vite did not emit any HTML page');
      for (const fileName of pages) {
        const path = resolve(directory, fileName);
        writeFileSync(path, readFileSync(path, 'utf8').replaceAll('__THEME_INIT__', theme.fileName));
      }

      const maps = Object.keys(bundle).filter((name) => name.endsWith('.map'));
      if (maps.length === 0) return;
      const build = resolve(sourcemapDirectory, basename(application.fileName, '.js'));
      rmSync(build, {recursive: true, force: true});
      for (const fileName of maps) {
        const from = resolve(directory, fileName);
        const to = resolve(build, fileName);
        const map = JSON.parse(readFileSync(from, 'utf8')) as {sources: string[]; sourceRoot?: string};
        const root = resolve(dirname(from), map.sourceRoot ?? '');
        map.sources = map.sources.map((source) => relative(dirname(to), resolve(root, source)));
        delete map.sourceRoot;
        mkdirSync(dirname(to), {recursive: true});
        // Write then remove: the output directory may be on another filesystem.
        writeFileSync(to, JSON.stringify(map));
        rmSync(from);
      }
      const builds = readdirSync(sourcemapDirectory, {withFileTypes: true})
        .filter((entry) => entry.isDirectory())
        .map((entry) => resolve(sourcemapDirectory, entry.name))
        .sort((first, second) => statSync(second).mtimeMs - statSync(first).mtimeMs);
      for (const old of builds.slice(KEPT_BUILDS)) rmSync(old, {recursive: true, force: true});
    },
  };
}

export default defineConfig({
  root: '.',
  base: '/static/app/',
  appType: 'spa',
  plugins: [devWebPageFallback(), finishBuildOutput()],
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
    },
    target: 'es2022',
    // Hidden: the maps exist for symbolicating reported stacks, but no bundle points at one.
    sourcemap: 'hidden',
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
});
