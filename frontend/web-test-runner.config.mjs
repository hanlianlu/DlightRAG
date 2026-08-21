import {esbuildPlugin} from '@web/dev-server-esbuild';
import {playwrightLauncher} from '@web/test-runner-playwright';

export default {
  files: ['ui/**/*.browser.test.ts'],
  nodeResolve: {exportConditions: ['browser', 'development']},
  plugins: [
    esbuildPlugin({ts: true, target: 'auto'}),
  ],
  browsers: [
    playwrightLauncher({product: 'chromium'}),
  ],
  testFramework: {
    config: {
      timeout: 5000,
    },
  },
};
