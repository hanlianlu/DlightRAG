// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {test} from 'node:test';
import assert from 'node:assert/strict';

import {parseSkillDirective, skillDirectivePrefix} from './skill_directive.ts';

test('parses a directive with a trailing question', () => {
  assert.deepEqual(parseSkillDirective('/skill:review Check this plan'), {
    skill: 'review',
    query: 'Check this plan',
  });
});

test('parses a directive with no question as an empty query', () => {
  assert.deepEqual(parseSkillDirective('/skill:review'), {skill: 'review', query: ''});
});

test('rejects directive-looking text with invalid skill names', () => {
  assert.equal(parseSkillDirective('/skill:Bad_Name x'), null);
  assert.equal(parseSkillDirective('not /skill:review'), null);
});

test('tolerates surrounding whitespace and multiline questions', () => {
  assert.deepEqual(parseSkillDirective('  /skill:code-review \n  Line one\nLine two  '), {
    skill: 'code-review',
    query: 'Line one\nLine two',
  });
});

test('tracks the directive prefix while the user types it', () => {
  assert.equal(skillDirectivePrefix('/'), '');
  assert.equal(skillDirectivePrefix('/s'), '');
  assert.equal(skillDirectivePrefix('/skill'), '');
  assert.equal(skillDirectivePrefix('/skill:'), '');
  assert.equal(skillDirectivePrefix('/skill:r'), 'r');
  assert.equal(skillDirectivePrefix('/skill:review'), 'review');
});

test('directive prefix tracking rejects off-path or complete drafts', () => {
  assert.equal(skillDirectivePrefix(''), null);
  assert.equal(skillDirectivePrefix('abc'), null);
  assert.equal(skillDirectivePrefix('/x'), null);
  assert.equal(skillDirectivePrefix('/skill:review the plan'), null);
  assert.equal(skillDirectivePrefix('/skill:Bad_Name'), null);
});
