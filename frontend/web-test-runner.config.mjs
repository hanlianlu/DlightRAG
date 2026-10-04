import {readFile} from 'node:fs/promises';
import {join, relative} from 'node:path';
import {esbuildPlugin} from '@web/dev-server-esbuild';
import {playwrightLauncher} from '@web/test-runner-playwright';
import {build} from 'vite';

// Vite projects CSS Modules into class-name objects. The test server otherwise
// serves an imported .module.css file as raw CSS, which browsers reject as a JS
// module. Preserve raw stylesheet links while projecting module imports.
const cssModulePlugin = {
  name: 'css-module-projection',
  transformImport({source}) {
    if (source.endsWith('.module.css')) return `${source}?wtr-css-module`;
    if (source.endsWith('.css')) return `${source}?wtr-css-empty`;
    return undefined;
  },
  async serve(context) {
    if (context.path.endsWith('.css') && 'wtr-css-empty' in context.query) {
      return {body: 'export {}', type: 'js'};
    }
    if (!context.path.endsWith('.module.css') || !('wtr-css-module' in context.query)) return;
    const source = await readFile(join(process.cwd(), context.path), 'utf8');
    const names = [...source.matchAll(/\.([A-Za-z_][\w-]*)/g)].map((match) => match[1]);
    const projection = Object.fromEntries([...new Set(names)].map((name) => [name, name]));
    return {
      body: `export default Object.freeze(${JSON.stringify(projection)});`,
      type: 'js',
    };
  },
};

// Accessibility is judged against what the product ships. One production build
// of vite.config.ts, written nowhere, yields the stylesheets the application
// page loads and the class names its CSS Modules received. The product-styles
// page links those stylesheets as one, in the order the application loads them,
// and maps styles/ imports to those class names, so components render exactly
// the selectors the stylesheets hold.
const productStylesPath = '/__product-styles__/';
const productStyledFiles = ['ui/a11y.browser.test.ts'];
let productStyles;

/**
 * The application page links its entry chunk's stylesheets, its imports' first,
 * and loads each dynamically imported chunk's stylesheets with that chunk.
 */
function applicationStylesheets(output) {
  const files = new Map(output.map((file) => [file.fileName, file]));
  const entry = output.find((file) => file.type === 'chunk' && file.isEntry && file.name === 'app');
  if (!entry) throw new Error('Vite did not emit the application entry');
  const stylesheets = new Set();
  const visited = new Set();
  const visit = (chunk) => {
    if (visited.has(chunk.fileName)) return;
    visited.add(chunk.fileName);
    for (const name of chunk.imports) visit(files.get(name));
    for (const name of chunk.viteMetadata.importedCss) stylesheets.add(name);
    for (const name of chunk.dynamicImports) visit(files.get(name));
  };
  visit(entry);
  return [...stylesheets].map((name) => files.get(name).source);
}

async function buildProductStyles() {
  const classNames = new Map();
  const {output} = await build({
    configFile: join(process.cwd(), 'vite.config.ts'),
    logLevel: 'error',
    build: {write: false},
    css: {modules: {getJSON: (file, names) => classNames.set(relative(process.cwd(), file), names)}},
  });
  return {stylesheet: applicationStylesheets(output).join('\n'), classNames};
}

const productStylesPlugin = {
  name: 'product-styles',
  async serve(context) {
    if (!context.path.startsWith(productStylesPath)) return undefined;
    productStyles ??= buildProductStyles();
    const {stylesheet, classNames} = await productStyles;
    const path = context.path.slice(productStylesPath.length);
    if (path === 'style.css') return {body: stylesheet, type: 'css'};
    if (path.endsWith('.module.css')) {
      return {body: `export default Object.freeze(${JSON.stringify(classNames.get(path))});`, type: 'js'};
    }
    // Every plain stylesheet a component imports already ships in style.css.
    return {body: 'export {}', type: 'js'};
  },
};

// The page leaves out the optional <head> tag. WTR injects its own module
// scripts right after a literal <head>, and Firefox ignores an import map that
// follows a module script, so its components would carry class names the
// shipped stylesheet does not hold. Without the tag, WTR injects after <body>,
// behind the import map.
function productStylesPage(testFramework) {
  const importMap = JSON.stringify({imports: {'/styles/': `${productStylesPath}styles/`}});
  return `<!DOCTYPE html>
<html lang="en" data-color-mode="dark">
  <script type="importmap">${importMap}</script>
  <link rel="stylesheet" href="${productStylesPath}style.css">
  <body><script type="module" src="${testFramework}"></script></body>
</html>`;
}

const browserProducts = (process.env.WTR_BROWSERS ?? 'chromium')
  .split(',')
  .map((product) => product.trim())
  .filter(Boolean);

export default {
  port: 7357,
  files: [
    'ui/**/*.browser.test.ts',
    'design-system/**/*.browser.test.ts',
    ...productStyledFiles.map((file) => `!${file}`),
  ],
  groups: [{name: 'product-styles', files: productStyledFiles, testRunnerHtml: productStylesPage}],
  nodeResolve: {exportConditions: ['browser', 'development']},
  plugins: [
    productStylesPlugin,
    cssModulePlugin,
    esbuildPlugin({ts: true, target: 'auto'}),
  ],
  browsers: browserProducts.map((product) => playwrightLauncher({product})),
  testsFinishTimeout: 30000,
  testFramework: {
    config: {
      timeout: 5000,
    },
  },
};
