# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ephemeral, reader-activated external playback; no Run/history mutation."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from dlightrag.adapters.http.browser.video_playback import VideoPlayer, resolve_video_playback
from dlightrag.engine.answer.client_contracts import ClientContractModel

router = APIRouter()


class VideoPlaybackRequest(ClientContractModel):
    url: str = Field(min_length=1, max_length=2048)


@router.post("/video-playback", response_model=VideoPlayer)
async def video_playback(body: VideoPlaybackRequest, request: Request) -> VideoPlayer:
    if getattr(request.state, "user_context", None) is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    player = await resolve_video_playback(body.url)
    if player is None:
        raise HTTPException(status_code=422, detail="No supported public video player for this URL")
    return player
