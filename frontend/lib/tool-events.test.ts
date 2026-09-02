// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {prettyToolName, toolDisplay} from './tool-display.ts';
import {applyToolEvent, MAX_TOOL_ROWS, toolStatusText} from './tool-events.ts';

test('known tools map to verbs with i18n ids', () => {
  const display = toolDisplay('load_skill');
  assert.deepEqual(display, {
    verb: 'Loading skill',
    verbId: 'chatFeature.tool.load_skill',
    known: true,
  });
});

test('unknown tools fall back to a prettified raw name without an i18n id', () => {
  assert.deepEqual(toolDisplay('acme_custom_tool'), {
    verb: 'Acme Custom Tool',
    verbId: null,
    known: false,
  });
  assert.equal(prettyToolName('agent-child:abc'), 'Agent Child Abc');
});

test('tool events build, update, and settle one trace row', () => {
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'load_skill', call_id: 'c1'});
  assert.equal(rows.length, 1);
  assert.equal(rows[0].state, 'running');
  assert.equal(rows[0].object, '');

  rows = applyToolEvent(rows, 'tool_progress', {call_id: 'c1', object_label: 'code-review'});
  assert.equal(rows[0].object, 'code-review');

  rows = applyToolEvent(rows, 'tool_end', {call_id: 'c1', duration_ms: 1200});
  assert.equal(rows[0].state, 'done');
  assert.equal(rows[0].durationMs, 1200);
  assert.ok(toolStatusText(rows).includes('code-review'));
});

test('failed outcome marks the row failed', () => {
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'bash', call_id: 'c1'});
  rows = applyToolEvent(rows, 'tool_end', {call_id: 'c1', outcome: 'failed', duration_ms: 30});
  assert.equal(rows[0].state, 'failed');
});

test('the trace keeps at most the newest rows', () => {
  let rows: ReturnType<typeof applyToolEvent> = [];
  for (let i = 0; i < MAX_TOOL_ROWS + 3; i++) {
    rows = applyToolEvent(rows, 'tool_start', {tool_name: 'read', call_id: `c${i}`});
  }
  assert.equal(rows.length, MAX_TOOL_ROWS);
  assert.equal(rows[rows.length - 1].callId, `c${MAX_TOOL_ROWS + 2}`);
});

test('malformed events never corrupt the trace', () => {
  assert.deepEqual(applyToolEvent([], 'tool_start', {}), []);
  assert.deepEqual(applyToolEvent([], 'tool_end', {call_id: 'c1'}), []);
  let rows = applyToolEvent([], 'tool_start', {tool_name: 'read', call_id: 'c1'});
  rows = applyToolEvent(rows, 'tool_progress', {call_id: 'c1', object_label: ''});
  assert.equal(rows[0].object, '');
  assert.deepEqual(applyToolEvent(rows, 'telemetry', {call_id: 'c1'}), rows);
});
