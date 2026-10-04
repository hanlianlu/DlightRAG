# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""An uncaught browser error, logged for the operator."""

import logging

from fastapi import APIRouter, status
from pydantic import Field

from dlightrag.engine.answer.client_contracts import ClientContractModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ClientErrorReport(ClientContractModel):
    detail: str = Field(max_length=2000)


@router.post("/client-errors", status_code=status.HTTP_204_NO_CONTENT)
async def client_error(body: ClientErrorReport) -> None:
    # %r keeps client text on one log line.
    logger.warning("Uncaught browser error: %r", body.detail)
