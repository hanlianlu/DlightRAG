// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Pure document-attachment admission/classification, DOM- and CSS-free so it
// can be unit-tested in Node. Mirrors ui/image_policy.ts; ui/attachments.ts is
// the browser/composer half that renders and wires these decisions.

export type AttachmentKind = 'image' | 'document' | 'unsupported';

export interface DocumentAdmissionPolicy {
    currentUploadLimit: number;
    maxUploadBytes: number;
}

// Mirrors the server-side `_DOCUMENT_SUFFIXES` admission set (without the dot).
const DOCUMENT_EXTENSIONS = new Set([
    'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'md', 'markdown', 'textpack',
    'txt', 'csv', 'json', 'html', 'htm', 'xml', 'yaml', 'yml', 'rtf', 'odt', 'epub',
    'tex', 'log', 'py', 'js', 'ts', 'css', 'scss', 'sql', 'sh', 'conf', 'ini', 'properties',
]);

function parseInteger(value: string | undefined, minimum: number): number | null {
    if (!value || !/^\d+$/.test(value)) return null;
    const parsed = Number(value);
    if (!Number.isSafeInteger(parsed) || parsed < minimum) return null;
    return parsed;
}

export function classifyAttachmentFile(file: File): AttachmentKind {
    if (file.type.startsWith('image/')) return 'image';
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    return DOCUMENT_EXTENSIONS.has(extension) ? 'document' : 'unsupported';
}

export function getDocumentAdmissionPolicy(
    root: Pick<HTMLElement, 'dataset'> | null = document.getElementById('app'),
): DocumentAdmissionPolicy | null {
    if (!root) return null;
    const currentUploadLimit = parseInteger(root.dataset.documentCurrentUploadLimit, 0);
    const maxUploadBytes = parseInteger(root.dataset.documentMaxUploadBytes, 1);
    if (currentUploadLimit === null || maxUploadBytes === null) return null;
    return {currentUploadLimit, maxUploadBytes};
}

export function acceptsDocumentUpload(
    file: File,
    currentCount: number,
    policy: DocumentAdmissionPolicy,
): boolean {
    return (
        classifyAttachmentFile(file) === 'document' &&
        currentCount < policy.currentUploadLimit &&
        file.size <= policy.maxUploadBytes
    );
}
