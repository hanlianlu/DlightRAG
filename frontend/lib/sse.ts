// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

export type SSEEventHandler = (eventType: string, data: string) => void;

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
  const dataParts: string[] = [];

  function dispatch(): void {
    if (dataParts.length === 0) return;
    onEvent(eventType || 'message', dataParts.join('\n'));
    eventType = '';
    dataParts.length = 0;
  }

  function processLine(rawLine: string): void {
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

export async function streamSSE(
  response: Response,
  onEvent: SSEEventHandler,
): Promise<void> {
  if (!response.body) {
    throw new Error('Response body is not streamable');
  }
  const parser = createSSEParser(onEvent);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      parser.push(decoder.decode(result.value, {stream: true}));
    }
    parser.push(decoder.decode());
    parser.flush();
  } finally {
    // Release the underlying connection on any exit (completion, error, abort).
    reader.cancel().catch(() => {});
  }
}
