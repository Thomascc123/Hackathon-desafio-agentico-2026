import time
from collections import defaultdict
from threading import Lock


class RateLimiter:
    def __init__(self, max_per_minute: int = 30):
        self.max_per_minute = max_per_minute
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str = "default") -> bool:
        now = time.time()
        cutoff = now - 60.0

        with self._lock:
            timestamps = self._buckets[key]
            timestamps[:] = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= self.max_per_minute:
                return False

            timestamps.append(now)
            return True

    def remaining(self, key: str = "default") -> int:
        now = time.time()
        cutoff = now - 60.0

        with self._lock:
            timestamps = self._buckets[key]
            timestamps[:] = [t for t in timestamps if t > cutoff]
            return max(0, self.max_per_minute - len(timestamps))
