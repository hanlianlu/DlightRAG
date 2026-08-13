// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export type SSEEventHandler = (eventType: string, data: string, id: string) => void;

interface SSEParser {
  push(chunk: string): void;
  flush(): void;
}

export function parseData(data: string): unknown {
  try {
    return JSON.parse(data);
  } catch (_) {
    return data;
  }
}

export function createSSEParser(onEvent: SSEEventHandler): SSEParser {
  let buffer = '';
  let eventType = '';
  let eventId = '';
  const dataParts: string[] = [];

  function dispatch(): void {
    if (dataParts.length === 0) return;
    onEvent(eventType || 'message', dataParts.join('\n'), eventId);
    eventType = '';
    eventId = '';
    dataParts.length = 0;
  }

  function processLine(rawLine: string): void {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
    if (line === '') {
      dispatch();
    } else if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('id:')) {
      eventId = line.slice(3).trim();
    } else if (line.startsWith('data: ')) {
      dataParts.push(line.slice(6));
    } else if (line.startsWith('data:')) {
      dataParts.push(line.slice(5));
    }
  }

  return {
    push(chunk: string): void {
      buffer += chunk;
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      lines.forEach(processLine);
    },
    flush(): void {
      if (buffer) {
        processLine(buffer);
        buffer = '';
      }
      dispatch();
    },
  };
}
