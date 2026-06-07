from __future__ import annotations

from dataclasses import is_dataclass, asdict
import importlib

import time
import json
import os
import threading
import queue

from collections import defaultdict, Counter, OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

DEFAULT_KV_BASE_PATH = (
    "/var/data/kv_data"
    if os.getenv("RENDER")
    else "kv_data"
)

KV_WARN_SIZE = 5 * 1024 * 1024
KV_MAX_SIZE = 10 * 1024 * 1024

LARGE_STATE_WARN_BYTES = 512 * 1024
LARGE_STATE_WARN_TTL_SECONDS = 30 * 60

MAX_KEYS_PER_STORE = 500_000
MAX_KEYS_WARN_THRESHOLD = 400_000


# --------------------------------------------------
# METRICS
# --------------------------------------------------

class KVMetrics:
    """Lightweight counters for observability. Thread-safe via GIL on CPython."""

    def __init__(self):
        self._counters: Counter = Counter()
        self._lock = threading.Lock()
        self._flush_durations: List[float] = []
        self._max_flush_durations = 1000

    def inc(self, key: str, n: int = 1):
        with self._lock:
            self._counters[key] += n

    def record_flush_duration(self, seconds: float):
        with self._lock:
            self._flush_durations.append(seconds)
            if len(self._flush_durations) > self._max_flush_durations:
                self._flush_durations = self._flush_durations[-self._max_flush_durations:]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            durations = list(self._flush_durations)
            counters = dict(self._counters)

        avg_flush = (sum(durations) / len(durations)) if durations else 0.0
        max_flush = max(durations) if durations else 0.0

        return {
            **counters,
            "flush_avg_ms": round(avg_flush * 1000, 2),
            "flush_max_ms": round(max_flush * 1000, 2),
            "flush_samples": len(durations),
        }

    def reset(self):
        with self._lock:
            self._counters.clear()
            self._flush_durations.clear()


# --------------------------------------------------
# WAL
# --------------------------------------------------

class WAL:
    """
    Minimal append-only Write-Ahead Log.

    Before any store file is overwritten, the pending snapshot is appended
    to the WAL.  On startup, any unacknowledged WAL entry is replayed so
    that a crash between os.replace calls cannot lose data.

    Format (one JSON object per line):
        {"store": "<name>", "ts": <epoch_float>, "snapshot": "<json_str>"}
    """

    def __init__(self, base_path: str):
        self._path = os.path.join(base_path, "_wal.jsonl")
        self._lock = threading.Lock()

    def append(self, store: str, snapshot: str):
        entry = json.dumps({
            "store": store,
            "ts": time.time(),
            "snapshot": snapshot,
        }, separators=(",", ":"))
        with self._lock:
            with open(self._path, "a") as f:
                f.write(entry + "\n")
                f.flush()
                os.fsync(f.fileno())

    def acknowledge(self, store: str):
        """Remove all WAL entries for a store after successful persist."""
        with self._lock:
            if not os.path.exists(self._path):
                return
            try:
                with open(self._path, "r") as f:
                    lines = f.readlines()
                remaining = [
                    l for l in lines
                    if not self._line_matches_store(l, store)
                ]
                with open(self._path, "w") as f:
                    f.writelines(remaining)
            except Exception as e:
                print(f"[WAL ACK ERROR] store={store} error={e}")

    def recover(self) -> Dict[str, str]:
        """
        Returns a dict of store -> latest snapshot string for any stores
        that have unacknowledged WAL entries.
        """
        if not os.path.exists(self._path):
            return {}
        recovered: Dict[str, str] = {}
        try:
            with self._lock:
                with open(self._path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            store = entry.get("store")
                            snapshot = entry.get("snapshot")
                            if store and snapshot:
                                recovered[store] = snapshot
                        except Exception:
                            continue
        except Exception as e:
            print(f"[WAL RECOVER ERROR] {e}")
        return recovered

    @staticmethod
    def _line_matches_store(line: str, store: str) -> bool:
        try:
            return json.loads(line).get("store") == store
        except Exception:
            return False


# --------------------------------------------------
# BACKGROUND FLUSH WORKER
# --------------------------------------------------

class FlushWorker(threading.Thread):
    """
    Dedicated background thread that performs all disk I/O.
    The hot request path enqueues work items; this thread drains the queue.
    Each item is a callable.  Shutdown is signalled via a sentinel None.
    """

    FLUSH_TIMEOUT_SECONDS = 30

    def __init__(self):
        super().__init__(daemon=True, name="kv-flush-worker")
        self._queue: queue.Queue = queue.Queue()
        self._metrics = KVMetrics()

    @property
    def metrics(self) -> KVMetrics:
        return self._metrics

    def submit(self, fn: Callable):
        self._queue.put_nowait(fn)

    def drain(self, timeout: float = 5.0):
        """Block until the queue is empty or timeout expires."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.empty():
                return
            time.sleep(0.01)

    def run(self):
        while True:
            try:
                fn = self._queue.get(timeout=self.FLUSH_TIMEOUT_SECONDS)
            except queue.Empty:
                continue

            if fn is None:
                break

            t0 = time.monotonic()
            try:
                fn()
            except Exception as e:
                print(f"[FLUSH WORKER ERROR] {e}")
            finally:
                elapsed = time.monotonic() - t0
                self._metrics.record_flush_duration(elapsed)
                self._queue.task_done()

    def shutdown(self):
        self._queue.put(None)
        self.join(timeout=10)


# --------------------------------------------------
# KV
# --------------------------------------------------

class KV:
    """
    Thread-safe, multi-tenant in-memory key-value store with:
    - Disk persistence (batched, background)
    - Write-ahead log for crash safety
    - TTL expiry with monotonic-aware delta checks
    - Per-store key-count cap with LRU eviction
    - Lazy cache loading with coordinated single-flight
    - Metrics and observability
    - Fully injectable base_path for testability
    """

    # Keys in this set are RAM-only and must never be persisted to disk.
    # catalog: and promo: writes via set/set_with_ttl are intentionally
    # blocked — use get_catalog_lazy / get_promo_lazy instead.
    ALLOWED_USER_FIELDS = {
        "language",
        "state",
    }

    _OP_STATE_KEYS = [
        "intent_ctx",

        "history_items_scope",
        "history_item",
        "history_days",

        "budget_match_text",
        "budget_candidates",
        "budget_phase_totals",
        "budget_currency",

        "last_result_text",
        "item_price_results",

        "advice_results",
        "advice_seen",

        "promo_lines",

        "variants_options",
        "variants_root_type",
        "variants_mode",

        "advisory_active",
        "advisory_page",
        "advisory_scope",
        "advisory_context",
        "advisory_suggested",

        "pending_action",
    ]


    def __init__(self, base_path: Optional[str] = None):

        self._base_path = base_path or DEFAULT_KV_BASE_PATH

        self._stores: Dict[str, Dict[str, Any]] = {}
        self._expiries: Dict[str, Dict[str, float]] = {}

        # Per-store LRU key order (oldest first).
        self._store_key_order: Dict[str, OrderedDict[str, None]] = {}

        self._thread_local = threading.local()

        # --------------------------------------------------
        # LOCK ORDERING POLICY
        #
        # ALWAYS:
        #   global_lock -> store_lock
        #
        # NEVER REVERSE.
        #
        # Disk IO must NEVER occur while holding global_lock.
        # --------------------------------------------------

        self._global_lock = threading.RLock()

        self._store_locks: Dict[str, threading.RLock] = {}
        self._active_ref_locks: Dict[str, threading.Lock] = {}
        self._store_loading: Dict[str, threading.Event] = {}

        # batching
        self._dirty: Dict[str, bool] = {}
        self._op_count: Dict[str, int] = {}
        self._last_flush: Dict[str, float] = {}
        self._flush_pending: Dict[str, bool] = {}

        # lifecycle tracking
        self._last_access: Dict[str, float] = {}
        self._loaded_stores: set = set()

        # active request protection
        self._active_refs: Dict[str, int] = defaultdict(int)

        # request ownership
        self._request_depth: Dict[Tuple, int] = defaultdict(int)

        # lightweight store residency
        self._resident_store_order: List[str] = []
        self._resident_store_set: set = set()

        # eviction
        self.STORE_IDLE_EVICT_SECONDS = 6 * 60 * 60
        self.STORE_EVICT_SCAN_INTERVAL = 10 * 60
        self._last_evict_scan = 0.0

        self.DIRTY_STORE_FORCE_FLUSH_SECONDS = 15 * 60

        self.MAX_HOT_STORES = 200
        self.SOFT_MAX_RESIDENT_STORES = 240

        self._BATCH_SIZE = 2000
        self._FLUSH_INTERVAL = 60

        # lifecycle
        self.ROTATION_SECONDS = 45 * 24 * 60 * 60
        self.CSW_TTL_SECONDS = 26 * 60 * 60
        self.USER_TTL_SECONDS = 3 * 24 * 60 * 60
        self.SHOPPING_TTL_SECONDS = 960 * 60 * 60

        self._large_state_warned: Dict[str, float] = {}

        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

        # Subsystems
        self._wal = WAL(self._base_path)
        self._flush_worker = FlushWorker()
        self._flush_worker.start()
        self._metrics = KVMetrics()

        # Replay any WAL entries from a previous crash before opening stores.
        self._replay_wal()

        self._initialize_empty_store("global")

        self.MAX_TTL_SECONDS = max(
            self.CSW_TTL_SECONDS,
            self.USER_TTL_SECONDS,
            self.SHOPPING_TTL_SECONDS,
        )

        if self.ROTATION_SECONDS <= self.MAX_TTL_SECONDS:
            self.ROTATION_SECONDS = self.MAX_TTL_SECONDS + (24 * 60 * 60)

    # --------------------------------------------------
    # PUBLIC METRICS
    # --------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        base = self._metrics.snapshot()
        worker = self._flush_worker.metrics.snapshot()
        with self._global_lock:
            base["loaded_stores"] = len(self._loaded_stores)
            base["resident_stores"] = len(self._resident_store_set)
            base["total_active_refs"] = sum(self._active_refs.values())
        return {**base, "flush_worker": worker}

    # --------------------------------------------------
    # STORE CONTEXT
    # --------------------------------------------------

    def _get_current_store(self) -> str:
        return getattr(self._thread_local, "store", "global")

    def _set_current_store(self, store: str):
        self._thread_local.store = store or "global"

    def _get_store_lock(self, store: Optional[str] = None) -> threading.RLock:
        store = store or self._get_current_store()
        with self._global_lock:
            lock = self._store_locks.get(store)
            if lock is None:
                lock = threading.RLock()
                self._store_locks[store] = lock
            return lock

    def _get_active_ref_lock(self, store: str) -> threading.Lock:
        with self._global_lock:
            lock = self._active_ref_locks.get(store)
            if lock is None:
                lock = threading.Lock()
                self._active_ref_locks[store] = lock
            return lock

    def _touch_store(self, store: str):
        now = time.time()
        with self._global_lock:
            self._last_access[store] = now
            if store in self._resident_store_set:
                if (
                    self._resident_store_order
                    and self._resident_store_order[-1] == store
                ):
                    return
                try:
                    self._resident_store_order.remove(store)
                except ValueError:
                    pass
            else:
                self._resident_store_set.add(store)
            self._resident_store_order.append(store)

    def _mark_store_active(self) -> str:
        store = self._get_current_store()
        self._touch_store(store)
        return store

    def _initialize_empty_store(self, store: str):
        with self._global_lock:
            if store not in self._stores:
                self._stores[store] = {}
            if store not in self._expiries:
                self._expiries[store] = {}
            if store not in self._store_key_order:
                self._store_key_order[store] = OrderedDict()
            self._dirty.setdefault(store, False)
            self._op_count.setdefault(store, 0)
            self._last_flush.setdefault(store, time.time())
            self._flush_pending.setdefault(store, False)
            self._loaded_stores.add(store)
        self._touch_store(store)

    def _enter_store_ref(self, store: str) -> bool:
        refs = getattr(self._thread_local, "active_refs", None)
        if refs is None:
            refs = set()
            self._thread_local.active_refs = refs

        depth_key = (threading.get_ident(), store)

        if store in refs:
            self._request_depth[depth_key] += 1
            return False

        with self._get_active_ref_lock(store):
            self._active_refs[store] += 1

        refs.add(store)
        self._request_depth[depth_key] = 1
        return True

    def _exit_store_ref(self, store: str):
        refs = getattr(self._thread_local, "active_refs", None)
        if not refs or store not in refs:
            return

        depth_key = (threading.get_ident(), store)
        current_depth = self._request_depth.get(depth_key, 0)

        if current_depth > 1:
            self._request_depth[depth_key] = current_depth - 1
            return

        # Clean up stale depth entry to prevent unbounded growth
        # under high concurrency with many short-lived threads.
        self._request_depth.pop(depth_key, None)

        try:
            refs.remove(store)
        except KeyError:
            pass

        with self._get_active_ref_lock(store):
            current = self._active_refs.get(store, 0)
            if current <= 1:
                self._active_refs.pop(store, None)
            else:
                self._active_refs[store] = current - 1

    @contextmanager
    def store_ref(self, store: Optional[str] = None):
        """
        Context manager that guarantees _exit_store_ref is called even if
        the caller raises.  Prefer this over manual enter/exit pairs.
        """
        s = store or self._get_current_store()
        owned = self._enter_store_ref(s)
        try:
            yield s
        finally:
            if owned:
                self._exit_store_ref(s)

    def clear_context(self):
        refs = getattr(self._thread_local, "active_refs", None)
        if refs:
            for store in list(refs):
                depth_key = (threading.get_ident(), store)
                while True:
                    try:
                        depth = self._request_depth.get(depth_key, 0)
                        if depth <= 0:
                            break
                        self._exit_store_ref(store)
                    except Exception:
                        try:
                            self._request_depth.pop(depth_key, None)
                            refs.discard(store)
                            with self._get_active_ref_lock(store):
                                current = self._active_refs.get(store, 0)
                                if current <= 1:
                                    self._active_refs.pop(store, None)
                                else:
                                    self._active_refs[store] = current - 1
                        except Exception:
                            pass
                        break

        self._thread_local.active_refs = set()
        self._thread_local.store = "global"


    def _is_store_loader_active(self, store: str) -> bool:
        return store in self._store_loading

    def _ensure_store_loaded(self, store: str):
        store = store or "global"

        should_load = False
        wait_event = None

        with self._global_lock:
            if store in self._loaded_stores:
                self._touch_store(store)
                return

            existing = self._store_loading.get(store)
            if existing:
                wait_event = existing
            else:
                wait_event = threading.Event()
                self._store_loading[store] = wait_event
                should_load = True

        if not should_load:
            completed = wait_event.wait(timeout=15)
            with self._global_lock:
                if store in self._loaded_stores:
                    self._touch_store(store)
                    return
                current = self._store_loading.get(store)
                if current is wait_event and not completed:
                    replacement = threading.Event()
                    self._store_loading[store] = replacement
                    wait_event = replacement
                    should_load = True

            if not should_load:
                completed = wait_event.wait(timeout=15)
                with self._global_lock:
                    if store in self._loaded_stores:
                        self._touch_store(store)
                        return
                    if not completed:
                        raise RuntimeError(f"Store load timeout for store={store}")

        tmp_store: Dict[str, Any] = {}
        tmp_expiry: Dict[str, float] = {}
        load_successful = False

        try:
            self._load_store(store, store_bucket=tmp_store, expiry_bucket=tmp_expiry)
            load_successful = True
        except Exception as e:
            print("[KV LOAD ERROR]", e)
            load_successful = False
        finally:
            with self._global_lock:
                if load_successful:
                    self._stores[store] = tmp_store
                    self._expiries[store] = tmp_expiry
                    self._store_key_order[store] = OrderedDict((k, None) for k in tmp_store.keys())
                    self._dirty.setdefault(store, False)
                    self._op_count.setdefault(store, 0)
                    self._last_flush.setdefault(store, time.time())
                    self._flush_pending.setdefault(store, False)
                    self._loaded_stores.add(store)

                event = self._store_loading.pop(store, None)
                if event:
                    event.set()

        if not load_successful:
            raise RuntimeError(f"Failed to load store={store}")

        self._touch_store(store)

    def _get_store_bucket(self) -> Dict[str, Any]:
        store = self._get_current_store()
        self._ensure_store_loaded(store)
        return self._stores[store]

    def _get_expiry_bucket(self) -> Dict[str, float]:
        store = self._get_current_store()
        self._ensure_store_loaded(store)
        return self._expiries[store]

    def set_context(self, store: str):
        store = store or "global"
        self._ensure_store_loaded(store)
        self._set_current_store(store)
        self._touch_store(store)

        now = time.time()
        if (now - self._last_evict_scan) >= self.STORE_EVICT_SCAN_INTERVAL:
            self._last_evict_scan = now
            self._evict_inactive_stores()

    # --------------------------------------------------
    # KEY HELPERS
    # --------------------------------------------------

    def _ns(self, key: str) -> str:
        return f"{self._get_current_store()}::{key}"

    # --------------------------------------------------
    # FILE PATHS
    # --------------------------------------------------

    def _file_path(self, store: Optional[str] = None) -> str:
        store = store or self._get_current_store()
        return os.path.join(self._base_path, f"{store}.json")

    def _backup_file_path(self, store: Optional[str] = None) -> str:
        store = store or self._get_current_store()
        return os.path.join(self._base_path, f"{store}.json.bak")

    def _file_size(self, store: Optional[str] = None) -> int:
        path = self._file_path(store)
        if not os.path.exists(path):
            return 0
        try:
            return os.path.getsize(path)
        except Exception:
            return 0

    def _monitor_size(self, store: Optional[str] = None) -> int:
        store = store or self._get_current_store()
        size = self._file_size(store)
        if size >= KV_WARN_SIZE:
            print(f"[KV WARNING] file size={size/1024/1024:.2f} MB store={store}")
            self._metrics.inc("warn.file_size_exceeded")
        return size

    # --------------------------------------------------
    # STATE SIZE WARNINGS
    # --------------------------------------------------

    def _estimate_state_size(self, value) -> int:
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            if isinstance(value, list):
                return len(value) * 72 + min(len(value), 100) * 32
            if isinstance(value, dict):
                return len(value) * 160 + min(len(value), 100) * 48
            if isinstance(value, tuple):
                return len(value) * 72
            return 256
        except Exception:
            return 0

    def _warn_large_state(self, key: str, value):
        try:
            now = time.time()
            cache_key = key.split(":")[-1]
            last_warn = self._large_state_warned.get(cache_key)
            if last_warn and (now - last_warn) < LARGE_STATE_WARN_TTL_SECONDS:
                return
            approx = self._estimate_state_size(value)
            if approx < LARGE_STATE_WARN_BYTES:
                return
            self._large_state_warned[cache_key] = now
            print(
                f"[KV LARGE STATE] key={key} "
                f"estimated_mb={approx/1024/1024:.2f} (approx)"
            )
            self._metrics.inc("warn.large_state")
        except Exception:
            pass

    # --------------------------------------------------
    # PER-STORE LRU KEY CAP
    # --------------------------------------------------

    def _track_key_access(self, store: str, namespaced_key: str):
        """Update LRU order for a key within a store. Must be called under store lock."""
        order = self._store_key_order.get(store)
        if order is None:
            return

        if namespaced_key in order:
            order.move_to_end(namespaced_key)
        else:
            order[namespaced_key] = None

    def _enforce_key_cap(self, store: str):
        """
        If the store exceeds MAX_KEYS_PER_STORE, evict the oldest (LRU) keys
        until we're back under the limit.  Must be called under store lock.
        """
        store_bucket = self._stores.get(store, {})
        order = self._store_key_order.get(store)

        key_count = len(store_bucket)

        if key_count >= MAX_KEYS_WARN_THRESHOLD:
            print(
                f"[KV KEY CAP WARNING] store={store} "
                f"keys={key_count} cap={MAX_KEYS_PER_STORE}"
            )
            self._metrics.inc("warn.key_cap_threshold")

        while len(store_bucket) > MAX_KEYS_PER_STORE and order:
            oldest_key, _ = order.popitem(last=False)
            store_bucket.pop(oldest_key, None)
            self._expiries.get(store, {}).pop(oldest_key, None)
            self._metrics.inc("evict.key_lru")
    # --------------------------------------------------
    # WAL REPLAY
    # --------------------------------------------------

    def _replay_wal(self):
        recovered = self._wal.recover()
        if not recovered:
            return

        for store, snapshot_str in recovered.items():
            try:
                data = json.loads(snapshot_str)
                tmp_store: Dict[str, Any] = {}
                tmp_expiry: Dict[str, float] = {}
                for k, v in data.get("store", {}).items():
                    tmp_store[k] = self._deserialize(v)

                for k, v in data.get("expiry", {}).items():
                    tmp_expiry[k] = v

                # Write the recovered snapshot to the store file.
                self._persist_snapshot(store, snapshot_str)
                self._wal.acknowledge(store)
                print(f"[KV WAL REPLAY] store={store} keys={len(tmp_store)}")
                self._metrics.inc("wal.replayed")
            except Exception as e:
                print(f"[KV WAL REPLAY ERROR] store={store} error={e}")
                self._metrics.inc("wal.replay_error")

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------

    def _safe_load_json(self, path: str) -> Any:
        with open(path, "r") as f:
            return json.load(f)

    def _recover_corrupted_store(self, store: str, path: str) -> Dict:
        backup_path = self._backup_file_path(store)
        quarantine_path = f"{path}.corrupt.{int(time.time())}"
        try:
            if os.path.exists(path):
                os.replace(path, quarantine_path)
        except Exception:
            pass

        if os.path.exists(backup_path):
            try:
                return self._safe_load_json(backup_path)
            except Exception:
                pass

        return {"store": {}, "expiry": {}}

    def _load_store(
        self,
        store: str,
        store_bucket: Optional[Dict[str, Any]] = None,
        expiry_bucket: Optional[Dict[str, float]] = None,
    ):
        path = self._file_path(store)
        store_bucket = store_bucket if store_bucket is not None else {}
        expiry_bucket = expiry_bucket if expiry_bucket is not None else {}

        if not os.path.exists(path):
            return

        try:
            data = self._safe_load_json(path)
        except Exception:
            print(f"[KV CORRUPT] store={store}")
            self._metrics.inc("error.corrupt_store")
            data = self._recover_corrupted_store(store, path)

        for k, v in data.get("store", {}).items():
            store_bucket[k] = self._deserialize(v)

        for k, v in data.get("expiry", {}).items():
            expiry_bucket[k] = v

    # --------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------

    def _serialize(self, value) -> Any:
        if is_dataclass(value):
            return {
                "__type__": value.__class__.__name__,
                "__module__": value.__class__.__module__,
                "data": asdict(value),
            }
        # Preserve set type so it round-trips correctly.
        if isinstance(value, set):
            return {
                "__type__": "__set__",
                "data": [self._serialize(v) for v in value],
            }
        if isinstance(value, list):
            return [self._serialize(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        return value

    def _deserialize(self, value) -> Any:
        if isinstance(value, dict):
            if value.get("__type__") == "__set__":
                return set(self._deserialize(v) for v in value.get("data", []))
            if "__type__" in value and "__module__" in value:
                try:
                    module = importlib.import_module(value["__module__"])
                    cls = getattr(module, value["__type__"])
                    return cls(**value["data"])
                except Exception:
                    return value.get("data", value)
            return {k: self._deserialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._deserialize(v) for v in value]
        return value

    # --------------------------------------------------
    # TTL HELPERS
    # --------------------------------------------------

    def _is_expired(self, expiry_ts: float) -> bool:
        """
        Use wall-clock time for expiry storage (so it survives restarts)
        but guard against backward clock skew by never treating a key as
        expired if the apparent age exceeds twice the max TTL.
        """
        now = time.time()
        if now > expiry_ts:
            # Sanity check: if the key appears to have expired more than
            # 2x the maximum possible TTL ago, the clock likely jumped
            # backward at some point — treat it as still valid.
            age = now - expiry_ts
            if age > 2 * self.MAX_TTL_SECONDS:
                return False
            return True
        return False

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    def _cleanup_expired(self, store: Optional[str] = None):
        store = store or self._get_current_store()
        lock = self._get_store_lock(store)
        with lock:
            store_bucket = self._stores.get(store, {})
            expiry_bucket = self._expiries.get(store, {})
            expired = [k for k, v in expiry_bucket.items() if self._is_expired(v)]
            for k in expired:
                store_bucket.pop(k, None)
                expiry_bucket.pop(k, None)
                self._store_key_order.get(store, {}).pop(k, None)

            if expired:
                self._metrics.inc("cleanup.expired_keys", len(expired))

    # --------------------------------------------------
    # PERSIST
    # --------------------------------------------------

    def _build_persist_snapshot(self, store: str) -> Tuple[str, bool, int]:
        lock = self._get_store_lock(store)
        with lock:
            store_bucket = dict(self._stores.get(store, {}))
            expiry_bucket = dict(self._expiries.get(store, {}))
            dirty_before = self._dirty.get(store, False)
            op_before = self._op_count.get(store, 0)

        data: Dict[str, Any] = {"store": {}, "expiry": {}}

        for k, v in store_bucket.items():
            data["store"][k] = self._serialize(v)

        for k, v in expiry_bucket.items():
            data["expiry"][k] = v


        return (
            json.dumps(data, separators=(",", ":")),
            dirty_before,
            op_before,
        )

    def _persist_snapshot(self, store: str, snapshot: str):
        """
        Atomically write snapshot to disk using a temp file + os.replace.
        The backup is written before the live file is replaced so that a
        crash mid-replace leaves either the old file or the new file intact.
        """
        file_path = self._file_path(store)
        backup_path = self._backup_file_path(store)
        tmp = f"{file_path}.{os.getpid()}.tmp"

        with open(tmp, "w") as f:
            f.write(snapshot)
            f.flush()
            os.fsync(f.fileno())

        if os.path.exists(file_path):
            try:
                os.replace(file_path, backup_path)
            except Exception:
                pass

        os.replace(tmp, file_path)

    def _persist(self, store: Optional[str] = None):
        store = store or self._get_current_store()

        # Guard: if the store was evicted before we got here, skip.
        with self._global_lock:
            if store not in self._loaded_stores:
                return

        t0 = time.monotonic()
        try:
            snapshot, dirty_before, op_before = self._build_persist_snapshot(store)

            # Write to WAL before touching the live file.
            self._wal.append(store, snapshot)

            self._persist_snapshot(store, snapshot)

            # Acknowledge WAL entry — live file is now safe.
            self._wal.acknowledge(store)

            lock = self._get_store_lock(store)
            with lock:
                current_dirty = self._dirty.get(store, False)
                current_op = self._op_count.get(store, 0)
                if current_dirty == dirty_before and current_op == op_before:
                    self._dirty[store] = False
                    self._op_count[store] = 0
                    self._last_flush[store] = time.time()

            elapsed = time.monotonic() - t0
            self._metrics.record_flush_duration(elapsed)
            self._metrics.inc("persist.success")

        except Exception as e:
            print("KV PERSIST ERROR:", e)
            self._metrics.inc("persist.error")

    def flush_all(self, timeout: float = 30.0):
        """
        Flush all dirty stores.  Each store has `timeout` seconds; the total
        wall time is bounded by timeout * number_of_dirty_stores.
        """
        with self._global_lock:
            stores = [s for s, dirty in self._dirty.items() if dirty]

        for store in stores:
            try:
                done = threading.Event()

                def _flush(s=store, ev=done):
                    try:
                        self._cleanup_expired(s)
                        self._persist(s)
                    finally:
                        ev.set()

                self._flush_worker.submit(_flush)

                if not done.wait(timeout=timeout):
                    print(f"[KV FLUSH ALL TIMEOUT] store={store}")
                    self._metrics.inc("flush_all.timeout")

            except Exception as e:
                print("[KV FLUSH ALL ERROR]", store, e)
                self._metrics.inc("flush_all.error")

    # --------------------------------------------------
    # ROTATION
    # --------------------------------------------------

    def _file_age_exceeded(self, store: Optional[str] = None) -> bool:
        path = self._file_path(store)
        if not os.path.exists(path):
            return False
        try:
            return (time.time() - os.path.getmtime(path)) >= self.ROTATION_SECONDS
        except Exception:
            return False

    def _rotate_if_needed(self, store: Optional[str] = None):
        store = store or self._get_current_store()
        lock = self._get_store_lock(store)

        should_rotate = False
        with lock:
            file_path = self._file_path(store)
            if not os.path.exists(file_path):
                return
            size = self._monitor_size(store)
            should_rotate = self._file_age_exceeded(store) or size >= KV_MAX_SIZE

        if not should_rotate:
            return

        backup_path = self._backup_file_path(store)
        file_path = self._file_path(store)

        try:
            snapshot, _, _ = self._build_persist_snapshot(store)

            # Write new content to temp first — single atomic replace at end.
            tmp = f"{file_path}.{os.getpid()}.rotate.tmp"
            with open(tmp, "w") as f:
                f.write(snapshot)
                f.flush()
                os.fsync(f.fileno())

            if os.path.exists(file_path):
                os.replace(file_path, backup_path)

            os.replace(tmp, file_path)

            if os.path.exists(backup_path):
                os.remove(backup_path)

            self._metrics.inc("rotation.success")

        except Exception as e:
            print("[KV ROTATION ERROR]", e)
            self._metrics.inc("rotation.error")
            if os.path.exists(backup_path) and not os.path.exists(file_path):
                os.replace(backup_path, file_path)

    # --------------------------------------------------
    # FLUSH CONTROLLER
    # --------------------------------------------------

    def _mark_dirty(self, store: str):
        self._dirty[store] = True
        self._op_count[store] += 1

    def _maybe_flush(self, store: Optional[str] = None):
        """
        Enqueue a background flush if the batch size or flush interval
        has been reached.  Never performs disk I/O on the calling thread.
        """
        store = store or self._get_current_store()

        if not self._dirty.get(store):
            return

        now = time.time()
        should_flush = (
            self._op_count.get(store, 0) >= self._BATCH_SIZE
            or (now - self._last_flush.get(store, now)) >= self._FLUSH_INTERVAL
        )

        if not should_flush:
            return

        with self._global_lock:
            if self._flush_pending.get(store):
                return
            self._flush_pending[store] = True

        def _work(s=store):
            try:
                self._cleanup_expired(s)
                self._persist(s)
                self._monitor_size(s)
                self._rotate_if_needed(s)
            except Exception as e:
                print("[KV MAYBE FLUSH ERROR]", e)
            finally:
                with self._global_lock:
                    self._flush_pending[s] = False

        self._flush_worker.submit(_work)

    # --------------------------------------------------
    # CORE METHODS
    # --------------------------------------------------

    def get(self, key: str, touch: bool = False):
        """
        Retrieve a value.

        Parameters
        ----------
        touch : bool
            If True, refresh the TTL on read (useful for session-style keys
            to prevent expiry during active use).
        """
        store = self._mark_store_active()

        with self.store_ref(store):
            lock = self._get_store_lock(store)
            with lock:
                namespaced_key = self._ns(key)
                store_bucket = self._get_store_bucket()
                expiry_bucket = self._get_expiry_bucket()

                expiry = expiry_bucket.get(namespaced_key)
                if expiry is not None and self._is_expired(expiry):
                    store_bucket.pop(namespaced_key, None)
                    expiry_bucket.pop(namespaced_key, None)
                    self._store_key_order.get(store, {}).pop(namespaced_key, None)
                    self._dirty[store] = True
                    self._maybe_flush(store)
                    self._metrics.inc("get.expired")
                    return None

                value = store_bucket.get(namespaced_key)

                if value is None:
                    self._metrics.inc("get.miss")
                    return None

                self._metrics.inc("get.hit")
                self._track_key_access(store, namespaced_key)

                if touch and expiry is not None:
                    # Renew TTL by extending by the original remaining delta.
                    # We don't know the original TTL, so we extend by the
                    # remaining time (keeping absolute expiry the same) —
                    # or use a sensible default.  Here we simply extend by
                    # USER_TTL_SECONDS as a safe default for session keys.
                    expiry_bucket[namespaced_key] = time.time() + self.USER_TTL_SECONDS
                    self._mark_dirty(store)

                return value

    def set(self, key: str, value):

        store = self._mark_store_active()

        with self.store_ref(store):
            self._warn_large_state(key, value)
            lock = self._get_store_lock(store)
            with lock:
                namespaced_key = self._ns(key)
                store_bucket = self._get_store_bucket()

                if namespaced_key.endswith(":nav_state") and isinstance(value, dict):
                    if value.get("node") == "MENU":
                        try:
                            phone = namespaced_key.split("user:")[1].split(":")[0]
                            self._clear_op_state_unlocked(phone, store)
                        except Exception:
                            pass

                store_bucket[namespaced_key] = value
                self._track_key_access(store, namespaced_key)
                self._enforce_key_cap(store)
                self._mark_dirty(store)
                self._maybe_flush(store)
                self._metrics.inc("set")

    def delete(self, key: str):
        store = self._mark_store_active()

        with self.store_ref(store):
            lock = self._get_store_lock(store)
            with lock:
                namespaced_key = self._ns(key)
                store_bucket = self._get_store_bucket()
                expiry_bucket = self._get_expiry_bucket()
                store_bucket.pop(namespaced_key, None)
                expiry_bucket.pop(namespaced_key, None)
                self._store_key_order.get(store, {}).pop(namespaced_key, None)
                self._mark_dirty(store)
                self._maybe_flush(store)
                self._metrics.inc("delete")

    def _delete_unlocked(self, key: str, store: str):
        """
        Delete a key while already holding the store lock.
        Used internally to avoid re-entrant lock acquisition in code paths
        such as clear_op_state called from within set/set_with_ttl.
        """
        namespaced_key = self._ns(key)
        self._stores.get(store, {}).pop(namespaced_key, None)
        self._expiries.get(store, {}).pop(namespaced_key, None)
        self._store_key_order.get(store, {}).pop(namespaced_key, None)
        self._mark_dirty(store)

    def set_with_ttl(self, key: str, value, ttl_seconds: int):

        store = self._mark_store_active()

        with self.store_ref(store):
            self._warn_large_state(key, value)
            lock = self._get_store_lock(store)
            with lock:
                namespaced_key = self._ns(key)
                store_bucket = self._get_store_bucket()
                expiry_bucket = self._get_expiry_bucket()

                if namespaced_key.endswith(":nav_state") and isinstance(value, dict):
                    old = store_bucket.get(namespaced_key)
                    old_node = old.get("node") if isinstance(old, dict) else None
                    new_node = value.get("node")
                    if old_node and old_node != "MENU" and new_node == "MENU":
                        try:
                            phone = namespaced_key.split("user:")[1].split(":")[0]
                            self._clear_op_state_unlocked(phone, store)
                        except Exception:
                            pass

                store_bucket[namespaced_key] = value
                expiry_bucket[namespaced_key] = time.time() + ttl_seconds
                self._track_key_access(store, namespaced_key)
                self._enforce_key_cap(store)
                self._mark_dirty(store)
                self._maybe_flush(store)
                self._metrics.inc("set_with_ttl")

    # --------------------------------------------------
    # CSW
    # --------------------------------------------------

    def _csw_key(self, phone: str) -> str:
        return f"user:{phone}:last_inbound_ts"

    def set_last_inbound(self, phone: str):
        self.set_with_ttl(self._csw_key(phone), time.time(), self.CSW_TTL_SECONDS)

    def get_last_inbound(self, phone: str) -> Optional[float]:
        value = self.get(self._csw_key(phone), touch=True)
        return float(value) if isinstance(value, (int, float)) else None

    def is_within_csw(self, phone: str) -> bool:
        return self.get_last_inbound(phone) is not None

    # --------------------------------------------------
    # USER DATA
    # --------------------------------------------------

    def _user_key(self, phone: str, name: str) -> str:
        return f"user:{phone}:{name}"

    def set_user(self, phone: str, name: str, value):
        if name not in self.ALLOWED_USER_FIELDS:
            return
        self.set_with_ttl(self._user_key(phone, name), value, self.USER_TTL_SECONDS)

    def get_user(self, phone: str, name: str):
        if name not in self.ALLOWED_USER_FIELDS:
            return None
        return self.get(self._user_key(phone, name), touch=True)

    def delete_user(self, phone: str, name: str):
        if name not in self.ALLOWED_USER_FIELDS:
            return
        self.delete(self._user_key(phone, name))

    # --------------------------------------------------
    # SHOPPING LIST
    # --------------------------------------------------

    def _shopping_key(self, phone: str) -> str:
        return f"user:{phone}:shopping_list"

    def _get_shopping_list_locked(self, store_bucket, expiry_bucket, key) -> List:
        namespaced_key = self._ns(key)
        expiry = expiry_bucket.get(namespaced_key)
        if expiry is not None and self._is_expired(expiry):
            store_bucket.pop(namespaced_key, None)
            expiry_bucket.pop(namespaced_key, None)
            self._store_key_order.get(self._get_current_store(), {}).pop(namespaced_key, None)
            return []
        return store_bucket.get(namespaced_key) or []

    def add_to_shopping_list(self, phone: str, item: Dict[str, Any]):
        store = self._mark_store_active()
        with self.store_ref(store):
            key = self._shopping_key(phone)
            lock = self._get_store_lock(store)
            with lock:
                store_bucket = self._get_store_bucket()
                expiry_bucket = self._get_expiry_bucket()
                items = self._get_shopping_list_locked(store_bucket, expiry_bucket, key)
                sku = item.get("sku_fingerprint")
                qty = int(item.get("quantity", 1))

                for existing in items:
                    if existing.get("sku_fingerprint") == sku:
                        existing["quantity"] += qty
                        namespaced_key = self._ns(key)
                        store_bucket[namespaced_key] = items
                        expiry_bucket[namespaced_key] = time.time() + self.SHOPPING_TTL_SECONDS
                        self._track_key_access(store, namespaced_key)
                        self._mark_dirty(store)
                        self._maybe_flush(store)
                        return

                items.append({
                    "item_name": item.get("item_name"),
                    "sku_fingerprint": sku,
                    "price": item.get("price"),
                    "currency": item.get("currency"),
                    "quantity": qty,
                    "size": item.get("size"),
                    "weight": item.get("weight"),
                    "volume": item.get("volume"),
                    "added_at": int(time.time()),
                })

                namespaced_key = self._ns(key)
                store_bucket[namespaced_key] = items
                expiry_bucket[namespaced_key] = time.time() + self.SHOPPING_TTL_SECONDS
                self._track_key_access(store, namespaced_key)
                self._mark_dirty(store)
                self._maybe_flush(store)

    def decrement_shopping_item(self, phone: str, index: int, qty: int):
        store = self._mark_store_active()
        with self.store_ref(store):
            key = self._shopping_key(phone)
            lock = self._get_store_lock(store)
            with lock:
                store_bucket = self._get_store_bucket()
                expiry_bucket = self._get_expiry_bucket()
                items = self._get_shopping_list_locked(store_bucket, expiry_bucket, key)
                if index < 0 or index >= len(items):
                    return None
                item = items[index]
                new_qty = item.get("quantity", 1) - qty
                if new_qty > 0:
                    item["quantity"] = new_qty
                else:
                    items.pop(index)

                namespaced_key = self._ns(key)
                store_bucket[namespaced_key] = items
                expiry_bucket[namespaced_key] = time.time() + self.SHOPPING_TTL_SECONDS
                self._track_key_access(store, namespaced_key)
                self._mark_dirty(store)
                self._maybe_flush(store)
                return item

    def remove_shopping_item(self, phone: str, index: int):
        store = self._mark_store_active()
        with self.store_ref(store):
            key = self._shopping_key(phone)
            lock = self._get_store_lock(store)
            with lock:
                store_bucket = self._get_store_bucket()
                expiry_bucket = self._get_expiry_bucket()
                items = self._get_shopping_list_locked(store_bucket, expiry_bucket, key)
                if index < 0 or index >= len(items):
                    return None
                removed = items.pop(index)

                namespaced_key = self._ns(key)
                store_bucket[namespaced_key] = items
                expiry_bucket[namespaced_key] = time.time() + self.SHOPPING_TTL_SECONDS
                self._track_key_access(store, namespaced_key)
                self._mark_dirty(store)
                self._maybe_flush(store)
                return removed

    def get_shopping_list(self, phone: str) -> List[Dict[str, Any]]:
        return self.get(self._shopping_key(phone)) or []

    def clear_shopping_list(self, phone: str):
        self.delete(self._shopping_key(phone))

    # --------------------------------------------------
    # OP STATE CLEANUP
    # --------------------------------------------------

    def _clear_op_state_unlocked(self, phone: str, store: str):
        """
        Remove all op-state keys for a phone number while already holding
        the store lock. Safe to call from within set / set_with_ttl.
        """
        for k in self._OP_STATE_KEYS:
            self._delete_unlocked(f"user:{phone}:{k}", store)

    def clear_op_state(self, phone: str):
        """Public API — acquires the store lock itself."""
        for k in self._OP_STATE_KEYS:
            self.delete(f"user:{phone}:{k}")

    # --------------------------------------------------
    # STORE EVICTION
    # --------------------------------------------------

    def _evict_inactive_stores(self):
        now = time.time()
        current_store = self._get_current_store()
        flush_candidates = []
        removable = []

        # Snapshot under global_lock so active_refs reads are consistent.
        with self._global_lock:
            snapshot = list(self._last_access.items())
            resident_snapshot = list(self._resident_store_order)
            active_refs_snapshot = dict(self._active_refs)

        for store, last_used in snapshot:
            if store == current_store:
                continue
            if active_refs_snapshot.get(store, 0) > 0:
                continue
            if self._is_store_loader_active(store):
                continue
            if (now - last_used) < self.STORE_IDLE_EVICT_SECONDS:
                if (
                    self._dirty.get(store)
                    and (now - self._last_flush.get(store, now))
                    >= self.DIRTY_STORE_FORCE_FLUSH_SECONDS
                ):
                    flush_candidates.append(store)
                continue
            removable.append(store)

        resident_overflow = len(resident_snapshot) - self.SOFT_MAX_RESIDENT_STORES
        if resident_overflow > 0:
            overflow_candidates = resident_snapshot[:resident_overflow]
            for store in overflow_candidates:
                if store == current_store:
                    continue
                if active_refs_snapshot.get(store, 0) > 0:
                    continue
                if self._is_store_loader_active(store):
                    continue
                if store not in removable:
                    removable.append(store)

        for store in flush_candidates:
            try:
                self._persist(store)
            except Exception:
                pass

        with self._global_lock:
            for store in removable:
                # Re-check under global_lock — store may have become active
                # between the snapshot above and now.
                if self._active_refs.get(store, 0) > 0:
                    continue
                if self._is_store_loader_active(store):
                    continue

                self._stores.pop(store, None)
                self._expiries.pop(store, None)
                self._store_key_order.pop(store, None)
                self._dirty.pop(store, None)
                self._op_count.pop(store, None)
                self._last_flush.pop(store, None)
                self._flush_pending.pop(store, None)
                self._last_access.pop(store, None)
                self._store_loading.pop(store, None)
                self._active_refs.pop(store, None)
                self._loaded_stores.discard(store)
                self._store_locks.pop(store, None)
                self._active_ref_locks.pop(store, None)
                self._resident_store_set.discard(store)
                try:
                    self._resident_store_order.remove(store)
                except Exception:
                    pass

                print(f"[KV EVICT] store={store}")
                self._metrics.inc("evict.store")

    # --------------------------------------------------
    # LIFECYCLE
    # --------------------------------------------------

    def shutdown(self, flush_timeout: float = 10.0):
        """Gracefully flush all dirty stores and stop the background worker."""
        self.flush_all(timeout=flush_timeout)
        self._flush_worker.shutdown()


# --------------------------------------------------
# SINGLETON
# --------------------------------------------------

def _make_kv() -> KV:
    return KV()


_kv = _make_kv()


def get_kv() -> KV:
    return _kv


kv = _kv
