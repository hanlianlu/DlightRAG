// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export interface ImageAdmissionPolicy {
    maxCurrentImages: number;
    maxUploadBytes: number;
}

interface ImageUploadCandidate {
    type: string;
    size: number;
}

function parseInteger(value: string | undefined, minimum: number): number | null {
    if (!value || !/^\d+$/.test(value)) return null;
    const parsed = Number(value);
    if (!Number.isSafeInteger(parsed) || parsed < minimum) return null;
    return parsed;
}

export function getImageAdmissionPolicy(): ImageAdmissionPolicy | null {
    const root = document.getElementById('app');
    if (!root) return null;
    const maxCurrentImages = parseInteger(root.dataset.maxCurrentImages, 0);
    const maxUploadBytes = parseInteger(root.dataset.maxUploadBytes, 1);
    if (maxCurrentImages === null || maxUploadBytes === null) return null;
    return {maxCurrentImages, maxUploadBytes};
}

export function acceptsImageUpload(
    file: ImageUploadCandidate,
    currentCount: number,
    policy: ImageAdmissionPolicy,
): boolean {
    return (
        currentCount < policy.maxCurrentImages &&
        file.type.startsWith('image/') &&
        file.size <= policy.maxUploadBytes
    );
}
