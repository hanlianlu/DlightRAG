// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Pure document-attachment admission/classification, DOM- and CSS-free so it
// can be unit-tested in Node. Mirrors ui/image_policy.ts; ui/attachments.ts is
// the browser/composer half that renders and wires these decisions.

export type AttachmentKind = 'image' | 'document' | 'unsupported';

export interface DocumentAdmissionPolicy {
    currentUploadLimit: number;
    maxUploadBytes: number;
    extensions: ReadonlySet<string>;
}

function parseInteger(value: string | undefined, minimum: number): number | null {
    if (!value || !/^\d+$/.test(value)) return null;
    const parsed = Number(value);
    if (!Number.isSafeInteger(parsed) || parsed < minimum) return null;
    return parsed;
}

function parseExtensions(value: string | undefined): ReadonlySet<string> | null {
    if (!value) return null;
    try {
        const parsed: unknown = JSON.parse(value);
        if (!Array.isArray(parsed) || parsed.length === 0) return null;
        const extensions = new Set<string>();
        for (const extension of parsed) {
            if (typeof extension !== 'string' || !/^[a-z0-9]+$/.test(extension)) return null;
            extensions.add(extension);
        }
        return extensions;
    } catch {
        return null;
    }
}

export function classifyAttachmentFile(
    file: File,
    extensions: ReadonlySet<string>,
): AttachmentKind {
    if (file.type.startsWith('image/')) return 'image';
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    return extensions.has(extension) ? 'document' : 'unsupported';
}

export function getDocumentAdmissionPolicy(
    root: Pick<HTMLElement, 'dataset'> | null = document.getElementById('app'),
): DocumentAdmissionPolicy | null {
    if (!root) return null;
    const currentUploadLimit = parseInteger(root.dataset.documentCurrentUploadLimit, 0);
    const maxUploadBytes = parseInteger(root.dataset.documentMaxUploadBytes, 1);
    const extensions = parseExtensions(root.dataset.documentExtensions);
    if (currentUploadLimit === null || maxUploadBytes === null || extensions === null) return null;
    return {currentUploadLimit, maxUploadBytes, extensions};
}

export function acceptsDocumentUpload(
    file: File,
    currentCount: number,
    policy: DocumentAdmissionPolicy,
): boolean {
    return (
        classifyAttachmentFile(file, policy.extensions) === 'document' &&
        currentCount < policy.currentUploadLimit &&
        file.size <= policy.maxUploadBytes
    );
}
