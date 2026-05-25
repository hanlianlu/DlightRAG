// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export const conversationHistory = [];
export const HISTORY_SEND_CAP = 100;
export const activeWorkspaces = [];

const SESSION_KEY = 'dlightrag.session_id';

export function getSessionId() {
    let sessionId = window.localStorage.getItem(SESSION_KEY);
    if (!sessionId) {
        sessionId = (window.crypto && window.crypto.randomUUID)
            ? window.crypto.randomUUID()
            : `session_${Date.now()}_${Math.random().toString(16).slice(2)}`;
        window.localStorage.setItem(SESSION_KEY, sessionId);
    }
    return sessionId;
}

export function getHistoryWindow() {
    return conversationHistory;
}

export function appendExchange(query, answer, imageIds) {
    let userContent = query;
    if (imageIds && imageIds.length > 0) {
        userContent += `\n[attached images: ${imageIds.join(', ')}]`;
    }
    conversationHistory.push({role: 'user', content: userContent});
    conversationHistory.push({role: 'assistant', content: answer});
    if (conversationHistory.length > HISTORY_SEND_CAP) {
        conversationHistory.splice(0, conversationHistory.length - HISTORY_SEND_CAP);
    }
}

export function setActiveWorkspaces(workspaces) {
    activeWorkspaces.splice(0, activeWorkspaces.length, ...workspaces);
}
