// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** CSRF double-submit token for state-changing web requests. */

const CSRF_COOKIE_NAME = 'dlightrag_web_csrf';

function readCookie(name: string): string {
  const prefix = `${name}=`;
  for (const part of document.cookie.split(';')) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length));
    }
  }
  return '';
}

/** Headers to attach to every state-changing (unsafe-method) web fetch. */
export function csrfHeaders(contentType?: string): Record<string, string> {
  const headers: Record<string, string> = {};
  if (contentType) headers['Content-Type'] = contentType;
  const token = readCookie(CSRF_COOKIE_NAME);
  if (token) headers['X-CSRF-Token'] = token;
  return headers;
}

export {CSRF_COOKIE_NAME};
