// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Frontend-owned static contracts previously locked from Python. */

import {readdirSync, readFileSync, existsSync} from 'node:fs';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {test} from 'node:test';
import assert from 'node:assert/strict';

const UI = dirname(fileURLToPath(import.meta.url));
const FRONTEND = join(UI, '..');

function read(relative: string): string {
  return readFileSync(join(FRONTEND, relative), 'utf8');
}

function uiSources(): string[] {
  return readdirSync(UI)
    .filter((name) => name.endsWith('.ts') && !name.endsWith('.test.ts'))
    .map((name) => join(UI, name));
}

test('answer request submits only the unified attachments part', () => {
  const source = read('lib/answer-request.ts');
  assert.match(source, /form\.append\('attachments', file, file\.name\)/);
  assert.equal(source.includes("append('images'"), false);
  assert.equal(source.includes("append('documents'"), false);
});

test('chat bubbles wrap unbroken queries', () => {
  const css = read('styles/chat.module.css');
  const user = css.match(/\.userMessage\s*\{([^}]+)\}/)?.[1] ?? '';
  const wrapper = css.match(/\.userMessageWrapper\s*\{([^}]+)\}/)?.[1] ?? '';
  const ai = css.match(/\.aiMessageContent\s*\{([^}]+)\}/)?.[1] ?? '';
  assert.equal(user.includes('width: fit-content;'), false);
  assert.ok(user.includes('max-width: 100%;'));
  assert.ok(user.includes('min-width: 0;'));
  assert.ok(user.includes('overflow-wrap: anywhere;'));
  assert.ok(user.includes('white-space: pre-wrap;'));
  assert.ok(wrapper.includes('min-width: 0;'));
  assert.ok(ai.includes('overflow-wrap: anywhere;'));
});

test('source downloads use same-origin hrefs', () => {
  const sources = read('ui/inspector-sources.ts');
  assert.ok(sources.includes('safeSameOriginHref(source.downloadUrl)'));
  assert.ok(sources.includes("<a href=${download}"));
});

test('inspector cutover removed legacy panel setup', () => {
  const app = read('ui/app.ts');
  const inspector = read('ui/inspector.ts');
  for (const replaced of ['panel.ts', 'source-panel.ts', 'source_panel_view.ts', 'files-panel.ts']) {
    assert.equal(existsSync(join(UI, replaced)), false);
  }
  for (const legacy of ['setupPanel', 'setupSourcePanel', 'setupFilesPanel', 'panelOpening']) {
    assert.equal((app + inspector).includes(legacy), false);
  }
  assert.match(app, /<dl-inspector/);
  assert.match(inspector, /customElements\.define\('dl-inspector'/);
});

test('file and source actions use the semantic icon registry', () => {
  const files = read('ui/inspector-files.ts');
  const sources = read('ui/inspector-sources.ts');
  assert.ok(files.includes('inspectorFiles.deleteFileAria'));
  assert.ok(files.includes("fileStyles['file-delete-icon']"));
  assert.ok(sources.includes("s['source-action-icon-svg']"));
  assert.equal(files.includes('<svg') || sources.includes('<svg'), false);
});

test('rich content pipeline has one owner and two call sites', () => {
  const pipeline = read('ui/rich-rendering.ts');
  const answer = read('ui/answer-presentation.ts');
  const source = read('ui/inspector-sources.ts');
  for (const stage of ['setSanitizedLlmHtml', 'renderMath', 'renderDiagrams', 'secureExternalLinks']) {
    assert.ok(pipeline.includes(stage));
    assert.equal((answer + source).includes(stage), false);
  }
  for (const entry of ['mountRichHtml', 'typesetRichContent']) {
    assert.ok(answer.includes(entry));
    assert.ok(source.includes(entry));
  }
});

test('split layout keeps behavior out of product state', () => {
  const adapter = read('ui/split-panel.ts');
  const element = read('design-system/elements/split-layout.ts');
  assert.equal(existsSync(join(UI, 'resize.ts')), false);
  assert.ok(adapter.includes('COMPACT_SHELL_MEDIA'));
  assert.ok(adapter.includes('dlightrag-panel-width'));
  assert.ok(element.includes('dl-split-change'));
  assert.ok(element.includes('role="separator"'));
});

test('webawesome and nanoevents stay out of production', () => {
  const pack = read('package.json');
  assert.equal(pack.includes('@awesome.me/webawesome'), false);
  assert.equal(pack.includes('nanoevents'), false);
});

test('shell owns composition without a compatibility layer', () => {
  const app = read('ui/app.ts');
  const main = read('ui/main.ts');
  const production = uiSources()
    .filter((path) => !path.endsWith('.browser.test.ts'))
    .map((path) => readFileSync(path, 'utf8'))
    .join('\n');
  assert.equal(existsSync(join(FRONTEND, 'events')), false);
  assert.ok(app.includes('@dl-chat-memory-operation'));
  assert.equal(app.includes('document.getElementById'), false);
  assert.ok(main.includes("document.readyState === 'loading'"));
  assert.ok(read('ui/workspace-create.ts').includes('this.handles.ingest.set(created.workspace)'));
  assert.ok(production.includes("customElements.define('dl-"));
  assert.equal(production.includes("customElements.define('workspace-scope'"), false);
});

test('frontend sources have no htmx contract', () => {
  for (const path of uiSources()) {
    const source = readFileSync(path, 'utf8').toLowerCase();
    assert.equal(source.includes('htmx'), false, path);
  }
});
