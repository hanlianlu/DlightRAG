// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What the toast tests share: a region mounted as a host would, and a request that carries an Undo. */

import {defineDesignSystemElements} from '../design-system/index.ts';
import type {DlToastRegion, ToastRequestDetail, ToastTone} from '../ui/toast.ts';
import '../ui/toast.ts';

defineDesignSystemElements();

export function mountToast(tone: ToastTone = 'receipt', parent: HTMLElement = document.body): DlToastRegion {
  const toast = document.createElement('dl-toast-region');
  toast.className = 'toast';
  toast.tone = tone;
  parent.appendChild(toast);
  return toast;
}

export const withUndo = (
  message: string,
  onAction: () => Promise<string | undefined> = async () => undefined,
  focus?: boolean,
): ToastRequestDetail => ({message, action: {actionLabel: 'Undo', onAction, focus}});
