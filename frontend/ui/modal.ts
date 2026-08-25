// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export type FocusRestorer = () => void | Promise<void>;

/** Opens a native modal and resolves after focus restore or lifecycle cancellation. */
export function modalResult(
  dialog: HTMLDialogElement,
  restoreFocus: FocusRestorer,
  signal?: AbortSignal,
): Promise<string> {
  if (signal?.aborted) return Promise.resolve('');
  dialog.returnValue = '';
  dialog.showModal();
  return new Promise((resolve) => {
    let settled = false;
    const cleanup = (): void => {
      dialog.removeEventListener('close', onClose);
      signal?.removeEventListener('abort', onAbort);
    };
    const onClose = (): void => {
      if (settled) return;
      settled = true;
      cleanup();
      const result = dialog.returnValue;
      void (async () => {
        try {
          await restoreFocus();
        } catch {
          // Focus restoration must never leave the owning action unsettled.
        }
        resolve(result);
      })();
    };
    const onAbort = (): void => {
      if (settled) return;
      settled = true;
      cleanup();
      if (dialog.open) dialog.close();
      resolve('');
    };
    dialog.addEventListener('close', onClose);
    signal?.addEventListener('abort', onAbort, {once: true});
  });
}
