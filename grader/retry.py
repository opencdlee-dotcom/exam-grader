"""
Centralized retry, rate-limit, and circuit-breaker logic for all API calls.

Provides:
- `api_retry` decorator: exponential backoff + jitter for any API function
- `RateLimiter`: token-bucket limiter to proactively space requests
- `CircuitBreaker`: stops calling a failing endpoint after N consecutive failures

Usage:
    from grader.retry import api_retry, rate_limiter

    @api_retry(max_retries=5, base_delay=10)
    def call_api(...):
        return client.messages.create(...)

    # Or use the shared rate limiter before each call:
    rate_limiter.acquire()
    response = client.messages.create(...)
"""

import functools
import logging
import random
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def api_retry(
    max_retries: int = 5,
    base_delay: float = 10.0,
    max_delay: float = 300.0,
    jitter_fraction: float = 0.3,
    retryable_exceptions: tuple = (),
):
    """
    Decorator: exponential backoff with jitter for API calls.

    Args:
        max_retries: Total attempts (including the first).
        base_delay: Starting delay in seconds.
        max_delay: Cap on delay (seconds).
        jitter_fraction: Random jitter as a fraction of the computed delay.
        retryable_exceptions: Extra exception types to retry on (RateLimitError
                              is always retried).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import anthropic  # deferred so module loads even without SDK

            retry_on = (anthropic.RateLimitError,) + retryable_exceptions
            last_exc = None

            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except retry_on as exc:
                    last_exc = exc
                    if attempt >= max_retries - 1:
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * jitter_fraction)
                    wait = delay + jitter

                    exc_name = type(exc).__name__
                    logger.warning("[%s] attempt %d/%d, retrying in %.0fs...", exc_name, attempt + 1, max_retries, wait)
                    time.sleep(wait)

            raise last_exc  # unreachable, but satisfies type checker
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """
    Simple token-bucket rate limiter.

    Ensures no more than `requests_per_minute` calls are made in any
    rolling 60-second window.  Thread-safe.

    Usage:
        limiter = RateLimiter(requests_per_minute=40)
        limiter.acquire()       # blocks until a token is available
        client.messages.create(...)
    """

    def __init__(self, requests_per_minute: int = 40):
        if requests_per_minute <= 0:
            raise ValueError(f"requests_per_minute must be > 0, got {requests_per_minute}")
        self.rpm = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # min seconds between requests
        self._lock = threading.Lock()
        self._last_request: float = 0.0
        self._call_times: list[float] = []

    def acquire(self) -> None:
        """Block until a request token is available."""
        with self._lock:
            now = time.monotonic()

            # Prune calls older than 60s
            cutoff = now - 60.0
            self._call_times = [t for t in self._call_times if t > cutoff]

            if len(self._call_times) >= self.rpm:
                # Must wait until the oldest call expires from the window
                wait_until = self._call_times[0] + 60.0
                sleep_time = max(0, wait_until - now)
                if sleep_time > 0:
                    logger.info("Rate limiter: %d calls in last 60s, waiting %.1fs...", len(self._call_times), sleep_time)
                    time.sleep(sleep_time)

            # Also enforce minimum spacing to avoid micro-bursts
            elapsed = now - self._last_request
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)

            self._last_request = time.monotonic()
            self._call_times.append(self._last_request)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreaker:
    """
    Trips open after `failure_threshold` consecutive failures.
    While open, calls are immediately rejected for `cooldown` seconds,
    then a single probe call is allowed (half-open).

    Usage:
        cb = CircuitBreaker()
        cb.check()              # raises if circuit is open
        try:
            result = call_api()
            cb.record_success()
        except Exception:
            cb.record_failure()
            raise
    """
    failure_threshold: int = 5
    cooldown: float = 120.0  # seconds

    _consecutive_failures: int = field(default=0, init=False)
    _state: str = field(default="closed", init=False)  # closed, open, half_open
    _opened_at: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def check(self) -> None:
        """Raise RuntimeError if the circuit is open."""
        with self._lock:
            if self._state == "closed":
                return
            if self._state == "open":
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.cooldown:
                    self._state = "half_open"
                    logger.info("Circuit breaker: half-open, allowing probe request...")
                    return
                raise RuntimeError(
                    f"Circuit breaker OPEN — API failing. "
                    f"Waiting {self.cooldown - elapsed:.0f}s before retry."
                )
            # half_open: allow one probe
            return

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            if self._state == "half_open":
                logger.info("Circuit breaker: probe succeeded, closing circuit.")
                self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit breaker TRIPPED after %d consecutive failures. Cooling down for %.0fs.",
                    self._consecutive_failures,
                    self.cooldown,
                )
            elif self._state == "half_open":
                self._state = "open"
                self._opened_at = time.monotonic()
                logger.warning("Circuit breaker: probe failed, reopening for %.0fs.", self.cooldown)


# ---------------------------------------------------------------------------
# Shared instances (importable by any module)
# ---------------------------------------------------------------------------

# Default: 40 requests/min for Anthropic API (adjust per your tier)
rate_limiter = RateLimiter(requests_per_minute=40)

# Default: trip after 5 consecutive failures, 2min cooldown
circuit_breaker = CircuitBreaker(failure_threshold=5, cooldown=120.0)
