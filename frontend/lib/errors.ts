// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** True when `error` is a fetch/stream abort raised by `AbortController.abort()`. */
export function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
}
