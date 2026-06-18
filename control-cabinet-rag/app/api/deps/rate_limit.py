from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import DefaultDict, Deque

from fastapi import HTTPException, Depends

from app.api.deps.auth import require_api_key

MAX_REQ = 60
WINDOW_SEC = 60

_hits: DefaultDict[str, Deque[float]] = defaultdict(deque)


def rate_limit(api_key: str = Depends(require_api_key)) -> None:
    now = time.time()
    q = _hits[api_key]

    while q and now - q[0] > WINDOW_SEC:
        q.popleft()

    if len(q) >= MAX_REQ:
        raise HTTPException(
            status_code=429,
            detail="Too many requests",
        )

    q.append(now)