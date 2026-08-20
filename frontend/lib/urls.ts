// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const SAFE_DATA_IMAGE_SRC_RE = /^data:image\/(?:avif|bmp|gif|jpeg|jpg|png|webp);base64,[a-z0-9+/=]+$/i;

function parseHttpUrl(value: unknown): URL | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  try {
    const url = new URL(value, window.location.origin);
    return url.protocol === 'http:' || url.protocol === 'https:' ? url : null;
  } catch {
    return null;
  }
}

export function safeSameOriginHref(value: unknown): string {
  const url = parseHttpUrl(value);
  return url?.origin === window.location.origin ? url.href : '';
}

export function safeExternalHttpHref(value: unknown): string {
  return parseHttpUrl(value)?.href ?? '';
}

export function safeImageSrc(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) return '';
  try {
    const url = new URL(value, window.location.origin);
    if (
      (url.protocol === 'http:' || url.protocol === 'https:')
      && url.origin === window.location.origin
    ) return url.href;
    if (url.protocol === 'blob:' && url.origin === window.location.origin) return url.href;
    if (url.protocol === 'data:' && SAFE_DATA_IMAGE_SRC_RE.test(value.trim())) return url.href;
  } catch {
    return '';
  }
  return '';
}
