// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Single source of truth for the composer's ordered pending attachments (images
// and documents in one collection). Each item keeps a stable id and a preview
// object URL; dl-chat-composer renders and submits the files in this order.
// URL/id factories are injected so ordering and lifecycle can be unit-tested.

import type {AttachmentKind} from '../ui/attachment_policy.ts';

export type PendingAttachmentKind = Extract<AttachmentKind, 'image' | 'document'>;

export interface PendingAttachment {
    id: string;
    file: File;
    kind: PendingAttachmentKind;
    objectUrl: string;
}

export interface AttachmentLease {
    readonly items: readonly PendingAttachment[];
    readonly settled: boolean;
    accept(): void;
    restore(): void;
    discard(): void;
}

interface AttachmentStoreOptions {
    createId?: () => string;
    createObjectUrl?: (file: File) => string;
    revokeObjectUrl?: (url: string) => void;
}

export class AttachmentStore {
    readonly #items: PendingAttachment[] = [];
    readonly #subscribers = new Set<() => void>();
    readonly #createId: () => string;
    readonly #createObjectUrl: (file: File) => string;
    readonly #revokeObjectUrl: (url: string) => void;

    constructor(options: AttachmentStoreOptions = {}) {
        this.#createId = options.createId || (() => crypto.randomUUID());
        this.#createObjectUrl = options.createObjectUrl || ((file) => URL.createObjectURL(file));
        this.#revokeObjectUrl = options.revokeObjectUrl || ((url) => URL.revokeObjectURL(url));
    }

    get size(): number {
        return this.#items.length;
    }

    get imageCount(): number {
        return this.#items.reduce((count, item) => count + (item.kind === 'image' ? 1 : 0), 0);
    }

    list(): readonly PendingAttachment[] {
        return [...this.#items];
    }

    /** Move every attachment into a submission without creating another Blob URL. */
    leaseAll(): AttachmentLease {
        const items = this.#items.splice(0);
        if (items.length > 0) this.#notify();
        let settled = false;
        const settle = (restore: boolean): void => {
            if (settled) return;
            settled = true;
            if (restore) {
                this.#items.unshift(...items);
                if (items.length > 0) this.#notify();
                return;
            }
            for (const item of items) this.#revokeObjectUrl(item.objectUrl);
        };
        return {
            items: [...items],
            get settled() {
                return settled;
            },
            accept: () => settle(false),
            restore: () => settle(true),
            discard: () => settle(false),
        };
    }

    add(file: File, kind: PendingAttachmentKind): PendingAttachment {
        const item: PendingAttachment = {
            id: this.#createId(),
            file,
            kind,
            objectUrl: this.#createObjectUrl(file),
        };
        this.#items.push(item);
        this.#notify();
        return item;
    }

    remove(id: string): void {
        const index = this.#items.findIndex((item) => item.id === id);
        if (index < 0) return;
        const [removed] = this.#items.splice(index, 1);
        this.#revokeObjectUrl(removed.objectUrl);
        this.#notify();
    }

    clear(): void {
        if (this.#items.length === 0) return;
        for (const item of this.#items) this.#revokeObjectUrl(item.objectUrl);
        this.#items.length = 0;
        this.#notify();
    }

    subscribe(handler: () => void): () => void {
        this.#subscribers.add(handler);
        return () => {
            this.#subscribers.delete(handler);
        };
    }

    #notify(): void {
        for (const handler of this.#subscribers) handler();
    }
}

export const attachmentStore = new AttachmentStore();
