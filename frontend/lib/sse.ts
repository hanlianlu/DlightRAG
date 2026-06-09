// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

function parseData(data) {
    try {
        return JSON.parse(data);
    } catch (_) {
        return data;
    }
}

export function createSSEParser(onEvent) {
    let buffer = '';
    let eventType = '';
    let dataParts = [];

    function dispatch() {
        if (dataParts.length === 0) return;
        onEvent(eventType || 'message', dataParts.join('\n'));
        eventType = '';
        dataParts = [];
    }

    function processLine(rawLine) {
        const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
        if (line === '') {
            dispatch();
        } else if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
        } else if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
        } else if (line.startsWith('data: ')) {
            dataParts.push(line.slice(6));
        } else if (line.startsWith('data:')) {
            dataParts.push(line.slice(5));
        }
    }

    return {
        push(chunk) {
            buffer += chunk;
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            lines.forEach(processLine);
        },
        flush() {
            if (buffer) {
                processLine(buffer);
                buffer = '';
            }
            dispatch();
        },
    };
}

export async function streamSSE(response, onEvent) {
    if (!response.body) {
        throw new Error('Response body is not streamable');
    }
    const parser = createSSEParser(onEvent);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const result = await reader.read();
        if (result.done) break;
        parser.push(decoder.decode(result.value, {stream: true}));
    }
    parser.push(decoder.decode());
    parser.flush();
}

export {parseData};
