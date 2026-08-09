// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// One unified attachment admission policy for the composer: images and
// documents share a single ordered collection, one count limit, and per-item
// byte limits. DOM- and CSS-free so it can be unit-tested in Node; the browser
// half (ui/attachments.ts) renders and wires these decisions.

export type CapabilityStatus = 'supported' | 'unsupported' | 'unknown';
export type AttachmentKind = 'image' | 'document' | 'unsupported';

export interface AttachmentPolicy {
    countLimit: number;
    imageMaxBytes: number;
    documentMaxBytes: number;
    extensions: ReadonlySet<string>;
    imageCapability: CapabilityStatus;
    imageLimit: number;
}

export interface AttachmentCounts {
    total: number;
    images: number;
}

function parseInteger(value: string | undefined, minimum: number): number | null {
    if (!value || !/^\d+$/.test(value)) return null;
    const parsed = Number(value);
    if (!Number.isSafeInteger(parsed) || parsed < minimum) return null;
    return parsed;
}

function parseCapabilityStatus(value: string | undefined): CapabilityStatus | null {
    if (value === 'supported' || value === 'unsupported' || value === 'unknown') return value;
    return null;
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
    file: {name: string; type: string},
    extensions: ReadonlySet<string>,
): AttachmentKind {
    if (file.type.startsWith('image/')) return 'image';
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    return extensions.has(extension) ? 'document' : 'unsupported';
}

export function getAttachmentPolicy(
    root: Pick<HTMLElement, 'dataset'> | null = document.getElementById('app'),
): AttachmentPolicy | null {
    if (!root) return null;
    const countLimit = parseInteger(root.dataset.attachmentCountLimit, 1);
    const imageMaxBytes = parseInteger(root.dataset.attachmentImageMaxBytes, 1);
    const documentMaxBytes = parseInteger(root.dataset.attachmentDocumentMaxBytes, 1);
    const extensions = parseExtensions(root.dataset.attachmentExtensions);
    const imageCapability = parseCapabilityStatus(root.dataset.attachmentImageCapability);
    const imageLimit = parseInteger(root.dataset.attachmentImageLimit, 0);
    if (
        countLimit === null ||
        imageMaxBytes === null ||
        documentMaxBytes === null ||
        extensions === null ||
        imageCapability === null ||
        imageLimit === null
    ) {
        return null;
    }
    return {countLimit, imageMaxBytes, documentMaxBytes, extensions, imageCapability, imageLimit};
}

export function acceptsAttachmentUpload(
    file: {name: string; type: string; size: number},
    counts: AttachmentCounts,
    policy: AttachmentPolicy,
): boolean {
    const kind = classifyAttachmentFile(file, policy.extensions);
    if (kind === 'unsupported') return false;
    if (counts.total >= policy.countLimit) return false;
    if (kind === 'image') {
        if (policy.imageCapability !== 'supported') return false;
        if (counts.images >= policy.imageLimit) return false;
        return file.size <= policy.imageMaxBytes;
    }
    return file.size <= policy.documentMaxBytes;
}

export function attachmentsEnabled(policy: AttachmentPolicy): boolean {
    if (policy.countLimit <= 0) return false;
    const documentsAllowed = policy.extensions.size > 0;
    const imagesAllowed = policy.imageCapability === 'supported' && policy.imageLimit > 0;
    return documentsAllowed || imagesAllowed;
}
