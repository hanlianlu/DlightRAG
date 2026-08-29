// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Chinese catalog. Seeded with the bootstrap strings; completed in Milestone 4. */

import {html, type TemplateResult} from 'lit';

export const templates: Record<string, string | TemplateResult> = {
  'bootstrap.loading': '正在加载 DlightRAG…',
  'bootstrap.error': 'DlightRAG 加载失败。',
  'bootstrap.retry': '重试',
  'settings.language': '语言',
  'settings.language.automatic': '自动',
  'settings.language.english': 'English',
};
