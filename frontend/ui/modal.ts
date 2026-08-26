// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export type FocusRestorer = () => void | Promise<void>;

export interface ModalStateDetail {
  open: boolean;
}

/** Publish whether any native modal owned by this Feature is open. */
export function publishModalState(owner: HTMLElement): void {
  owner.dispatchEvent(new CustomEvent<ModalStateDetail>('dl-modal-state-change', {
    detail: {open: Boolean(owner.querySelector('dialog[open]'))},
    bubbles: true,
    composed: true,
  }));
}

/** Open one native modal and publish its owning Feature's aggregate state. */
export function showOwnedModal(owner: HTMLElement, dialog: HTMLDialogElement): void {
  dialog.showModal();
  publishModalState(owner);
}

/** Opens a native modal and resolves after focus restore or lifecycle cancellation. */
export function modalResult(
  owner: HTMLElement,
  dialog: HTMLDialogElement,
  restoreFocus: FocusRestorer,
  signal?: AbortSignal,
): Promise<string> {
  if (signal?.aborted) return Promise.resolve('');
  dialog.returnValue = '';
  showOwnedModal(owner, dialog);
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
      publishModalState(owner);
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
      publishModalState(owner);
      resolve('');
    };
    dialog.addEventListener('close', onClose);
    signal?.addEventListener('abort', onAbort, {once: true});
  });
}

declare global {
  interface HTMLElementEventMap {
    'dl-modal-state-change': CustomEvent<ModalStateDetail>;
  }
}
