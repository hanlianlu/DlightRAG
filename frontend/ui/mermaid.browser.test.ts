// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {expect} from '@esm-bundle/chai';
import {setLanguagePreference} from '../i18n/locale.ts';
import {waitFor} from '../testing/dom.ts';
import {DiagramRenderer, type MermaidApi, type MermaidLoader} from './mermaid.ts';

const SOURCE = 'graph TD;\nA-->B;';

type Config = Parameters<MermaidApi['initialize']>[0];

/** A Mermaid that draws the theme it was initialized with as the SVG's only text. */
function fakeMermaid(diagramType = 'flowchart-v2', hold?: Promise<void>) {
  const configs: Config[] = [];
  const renders: string[] = [];
  const api: MermaidApi = {
    initialize(config) {
      configs.push(config);
    },
    async render(_id, source) {
      renders.push(source);
      const theme = configs.at(-1)?.theme;
      await hold;
      return {
        svg: `<svg xmlns="http://www.w3.org/2000/svg"><text>${theme}</text></svg>`,
        diagramType,
      };
    },
  };
  return {api, configs, renders};
}

function mount(...sources: string[]): HTMLElement {
  const host = document.createElement('div');
  for (const source of sources) {
    const pre = document.createElement('pre');
    pre.className = 'mermaid-source';
    pre.textContent = source;
    host.appendChild(pre);
  }
  document.body.appendChild(host);
  return host;
}

function setColorMode(mode: 'light' | 'dark'): void {
  document.documentElement.dataset.colorMode = mode;
}

/** A macrotask: long enough for a rejected load to be handled. */
async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

const originalColorMode = document.documentElement.getAttribute('data-color-mode');
const originalCreateObjectURL = URL.createObjectURL;
// Every SVG the renderer shows is handed to an <img> as a Blob; the test reads them back.
let shown: Blob[];

async function shownSvgs(): Promise<string[]> {
  return await Promise.all(shown.map(async (blob) => await blob.text()));
}

beforeEach(() => {
  shown = [];
  URL.createObjectURL = (object) => {
    shown.push(object as Blob);
    return originalCreateObjectURL(object);
  };
});

afterEach(async () => {
  URL.createObjectURL = originalCreateObjectURL;
  if (originalColorMode === null) document.documentElement.removeAttribute('data-color-mode');
  else document.documentElement.setAttribute('data-color-mode', originalColorMode);
  document.body.replaceChildren();
  await setLanguagePreference('auto');
});

it('draws a diagram in the page color mode and keeps Mermaid configured to render safely', async () => {
  setColorMode('dark');
  const mermaid = fakeMermaid();
  const host = mount(SOURCE);

  new DiagramRenderer(async () => mermaid.api).render(host);
  await waitFor(() => host.querySelector('figure.mermaid-figure img') !== null);

  expect(host.querySelector('pre.mermaid-source')).to.equal(null);
  expect(mermaid.configs).to.have.length(1);
  expect(mermaid.configs[0].theme).to.equal('dark');
  expect(mermaid.configs[0].securityLevel).to.equal('strict');
  expect(mermaid.configs[0].flowchart?.htmlLabels).to.equal(false);
  expect(await shownSvgs()).to.have.length(1);
  expect(await shownSvgs()).to.satisfy((svgs: string[]) => svgs[0].includes('<text>dark</text>'));
});

it('redraws a mounted diagram from its source when the color mode changes', async () => {
  setColorMode('dark');
  const mermaid = fakeMermaid();
  const host = mount(SOURCE);
  new DiagramRenderer(async () => mermaid.api).render(host);
  await waitFor(() => host.querySelector('figure.mermaid-figure img') !== null);
  const figure = host.querySelector('figure.mermaid-figure')!;
  const firstSrc = figure.querySelector('img')!.src;

  setColorMode('light');
  await waitFor(() => shown.length === 2);

  // The same figure, drawn again in the other theme from the source it kept.
  expect(host.querySelector('figure.mermaid-figure')).to.equal(figure);
  expect(figure.querySelector('img')!.src).to.not.equal(firstSrc);
  expect(mermaid.renders).to.deep.equal([SOURCE, SOURCE]);
  expect(mermaid.configs.map((config) => config.theme)).to.deep.equal(['dark', 'default']);
  const [dark, light] = await shownSvgs();
  expect(dark).to.contain('<text>dark</text>');
  expect(light).to.contain('<text>default</text>');

  // Back to the first mode: the drawing for it is still cached, so Mermaid is not asked again.
  setColorMode('dark');
  await waitFor(() => shown.length === 3);
  expect(mermaid.renders).to.have.length(2);
  expect(await shownSvgs()).to.satisfy((svgs: string[]) => svgs[2].includes('<text>dark</text>'));
});

it('leaves a diagram alone when something other than the color mode changes', async () => {
  setColorMode('dark');
  const mermaid = fakeMermaid();
  const host = mount(SOURCE);
  new DiagramRenderer(async () => mermaid.api).render(host);
  await waitFor(() => host.querySelector('figure.mermaid-figure img') !== null);

  document.documentElement.setAttribute('data-theme', 'dark');
  setColorMode('dark');
  await settle();

  expect(shown).to.have.length(1);
  expect(mermaid.renders).to.have.length(1);
  document.documentElement.setAttribute('data-theme', 'system');
});

it('redraws a diagram whose color mode changed while it was being drawn', async () => {
  setColorMode('dark');
  let release!: () => void;
  const mermaid = fakeMermaid('flowchart-v2', new Promise<void>((resolve) => { release = resolve; }));
  const host = mount(SOURCE);
  new DiagramRenderer(async () => mermaid.api).render(host);
  await waitFor(() => mermaid.renders.length === 1);

  setColorMode('light');
  await settle();
  release();
  await waitFor(() => shown.length === 2);

  expect(host.querySelectorAll('figure.mermaid-figure')).to.have.length(1);
  expect(mermaid.configs.map((config) => config.theme)).to.deep.equal(['dark', 'default']);
  const [dark, light] = await shownSvgs();
  expect(dark).to.contain('<text>dark</text>');
  expect(light).to.contain('<text>default</text>');
});

it('does not redraw a diagram that has left the page', async () => {
  setColorMode('dark');
  const mermaid = fakeMermaid();
  const host = mount(SOURCE);
  new DiagramRenderer(async () => mermaid.api).render(host);
  await waitFor(() => host.querySelector('figure.mermaid-figure img') !== null);

  host.remove();
  setColorMode('light');
  await settle();

  expect(mermaid.renders).to.have.length(1);
  expect(shown).to.have.length(1);
});

it('a Mermaid that fails to load keeps every source, and the next diagram loads it again', async () => {
  const mermaid = fakeMermaid();
  let loads = 0;
  const load: MermaidLoader = async () => {
    loads += 1;
    if (loads === 1) throw new Error('chunk failed to load');
    return mermaid.api;
  };
  const renderer = new DiagramRenderer(load);
  const host = mount(SOURCE, 'graph LR;\nC-->D;');

  renderer.render(host);
  await waitFor(() => loads === 1);
  await settle();

  // One failed attempt for the whole batch, and the readable sources stay.
  expect(loads).to.equal(1);
  expect(host.querySelectorAll('pre.mermaid-source')).to.have.length(2);
  expect(host.querySelector('figure')).to.equal(null);

  renderer.render(host);
  await waitFor(() => host.querySelectorAll('figure.mermaid-figure').length === 2);

  expect(loads).to.equal(2);
  expect(host.querySelector('pre.mermaid-source')).to.equal(null);
});

it('a block Mermaid cannot parse keeps its source while the others are drawn', async () => {
  const mermaid = fakeMermaid();
  const api: MermaidApi = {
    initialize: mermaid.api.initialize,
    async render(id, source) {
      if (source.includes('broken')) throw new Error('parse error');
      return await mermaid.api.render(id, source);
    },
  };
  const host = mount('broken diagram', SOURCE);

  new DiagramRenderer(async () => api).render(host);
  await waitFor(() => host.querySelector('figure.mermaid-figure') !== null);

  expect(host.querySelectorAll('pre.mermaid-source')).to.have.length(1);
  expect(host.querySelector('pre.mermaid-source')!.textContent).to.equal('broken diagram');
});

it('describes a diagram to a reader who cannot see it, in the page language', async () => {
  const labelOf = async (source: string, diagramType: string): Promise<string> => {
    const host = mount(source);
    new DiagramRenderer(async () => fakeMermaid(diagramType).api).render(host);
    await waitFor(() => host.querySelector('figure.mermaid-figure img') !== null);
    return host.querySelector('img')!.alt;
  };

  expect(await labelOf(SOURCE, 'flowchart-v2')).to.equal('Flowchart');
  expect(await labelOf('sequenceDiagram\nA->>B: hi', 'sequence')).to.equal('Sequence diagram');
  expect(await labelOf('unheard-of', 'kanban')).to.equal('Diagram');
  expect(await labelOf('unheard-of', '')).to.equal('Diagram');
  expect(await labelOf('graph TD;\n  accTitle: Release flow\n  A-->B;', 'flowchart-v2'))
    .to.equal('Release flow');

  await setLanguagePreference('zh');
  expect(await labelOf(SOURCE, 'flowchart-v2')).to.equal('流程图');
  expect(await labelOf('unheard-of', 'kanban')).to.equal('图表');
});

it('describes a diagram already on the page again when the page language changes', async () => {
  const host = mount(SOURCE, 'graph TD;\n  accTitle: Release flow\n  A-->B;');
  new DiagramRenderer(async () => fakeMermaid('flowchart-v2').api).render(host);
  await waitFor(() => host.querySelectorAll('figure.mermaid-figure img').length === 2);
  const descriptions = (): string[] => [...host.querySelectorAll('img')].map((img) => img.alt);
  expect(descriptions()).to.deep.equal(['Flowchart', 'Release flow']);

  // A diagram's own title is its words; what kind of diagram it is, is the page's.
  await setLanguagePreference('zh');
  expect(descriptions()).to.deep.equal(['流程图', 'Release flow']);

  await setLanguagePreference('en');
  expect(descriptions()).to.deep.equal(['Flowchart', 'Release flow']);
});
