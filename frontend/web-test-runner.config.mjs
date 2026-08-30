import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {esbuildPlugin} from '@web/dev-server-esbuild';
import {playwrightLauncher} from '@web/test-runner-playwright';

// Vite projects CSS Modules into class-name objects. The test server otherwise
// serves an imported .module.css file as raw CSS, which browsers reject as a JS
// module. Preserve raw stylesheet links while projecting module imports.
const cssModulePlugin = {
  name: 'css-module-projection',
  transformImport({source}) {
    return source.endsWith('.module.css') ? `${source}?wtr-css-module` : undefined;
  },
  async serve(context) {
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

const browserProducts = (process.env.WTR_BROWSERS ?? 'chromium')
  .split(',')
  .map((product) => product.trim())
  .filter(Boolean);

export default {
  files: ['ui/**/*.browser.test.ts', 'design-system/**/*.browser.test.ts'],
  nodeResolve: {exportConditions: ['browser', 'development']},
  plugins: [
    cssModulePlugin,
    esbuildPlugin({ts: true, target: 'auto'}),
  ],
  browsers: browserProducts.map((product) => playwrightLauncher({product})),
  testFramework: {
    config: {
      timeout: 5000,
    },
  },
};
