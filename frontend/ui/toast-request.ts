// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import type {ToastRequestDetail} from './toast.ts';

/** Request a toast by raising `dl-toast-request` toward the app shell, which
 *  owns the single toast region. */
export function requestToast(host: HTMLElement, detail: ToastRequestDetail): void {
  host.dispatchEvent(new CustomEvent<ToastRequestDetail>('dl-toast-request', {
    detail,
    bubbles: true,
    composed: true,
  }));
}
