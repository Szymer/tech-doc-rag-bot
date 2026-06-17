from fastapi import HTTPException, status


def check_rate_limit(requests_per_minute: int = 60) -> None:
    # Placeholder for Redis or in-memory implementation.
    if requests_per_minute <= 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
