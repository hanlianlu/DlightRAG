// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {parseWire} from './wire.ts';

export const videoPlaybackLink = v.pipe(
  v.object({
    url: v.string(), provider: v.string(),
    player_domains: v.array(v.pipe(v.string(), v.regex(/^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/))),
  }),
  v.transform((wire) => ({url: wire.url, provider: wire.provider, playerDomains: wire.player_domains})),
);
export type VideoPlaybackLink = v.InferOutput<typeof videoPlaybackLink>;

function officialPlayer(value: string, link: VideoPlaybackLink): boolean {
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:' || url.username || url.password || url.port) return false;
    // Playback permission comes from the selected link's registry projection,
    // never from origins newly claimed by the resolver response or its HTML.
    if (url.hostname === window.location.hostname) return false;
    return link.playerDomains.some((domain) => url.hostname === domain || url.hostname.endsWith(`.${domain}`));
  } catch { return false; }
}

export async function resolveVideoPlayback(link: VideoPlaybackLink, signal: AbortSignal) {
  const player = v.pipe(
    v.object({
      embed_url: v.pipe(v.string(), v.check((value) => officialPlayer(value, link))),
      aspect_ratio: v.pipe(v.number(), v.minValue(0.25), v.maxValue(4)),
    }),
    v.transform((wire) => ({embedUrl: wire.embed_url, aspectRatio: wire.aspect_ratio})),
  );
  const response = await fetch('/web/api/video-playback', {
    method: 'POST', headers: csrfHeaders('application/json'),
    body: JSON.stringify({url: link.url}), signal: AbortSignal.any([signal, AbortSignal.timeout(8000)]),
  });
  return parseWire(response, player, (_status, message) => new Error(message), 'Video playback unavailable');
}
