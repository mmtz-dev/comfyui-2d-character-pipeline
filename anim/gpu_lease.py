"""GPU gatekeeper client: book GPU memory before using it.

Single file, standard library only, so it can be copied into any Docker image
(scripts/sync-client.sh does that). The canonical copy lives in the
gpu-gatekeeper repo under client/; edit it there, not in the copies.

    from gpu_lease import gpu_lease

    with gpu_lease("flux-gen") as lease:   # waits in the queue if the GPU is busy
        load_model()
        lease.mark_loaded()                 # model is on the GPU now
        generate()
    # lease released here, also on exceptions

Explicit form:

    lease = acquire("lora-train", timeout=600)
    try:
        ...
        lease.mark_loaded()
    finally:
        lease.release()

Gatekeeper URL: env GPU_GATEKEEPER_URL, default http://host.docker.internal:7886.
If the gatekeeper can't be reached at all, the job runs anyway ("fail-open")
after a short wait, with a loud warning.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Callable, Iterator

__version__ = "1.1.0"

__all__ = [
    "gpu_lease",
    "acquire",
    "Lease",
    "GpuBusyError",
    "GpuLeaseError",
    "GpuLeaseCancelled",
    "DEFAULT_URL",
    "__version__",
]

log = logging.getLogger("gpu_lease")

DEFAULT_URL = "http://host.docker.internal:7886"
UNREACHABLE_WAIT_S = 5.0  # how long to keep trying an unreachable gatekeeper before running unguarded
POLL_WAIT_S = 25.0  # long-poll length per GET while queued (server max is 60)
CANCEL_POLL_S = 2.0  # shorter polls when a `cancel` check is given, so a cancel is seen quickly
DEFAULT_RENEW_S = 30.0
GIB = 1024**3


class GpuLeaseError(RuntimeError):
    """The gatekeeper refused the request (unknown profile, can never fit, bad input)."""

    def __init__(self, message: str, status: int | None = None, detail: object = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class GpuBusyError(TimeoutError):
    """The lease was still queued when `timeout` ran out: the GPU is busy."""


class GpuLeaseCancelled(RuntimeError):
    """`cancel()` turned true while the lease was still queued; the request was withdrawn."""


class _Unreachable(Exception):
    """Connection-level failure talking to the gatekeeper."""


class _HttpError(Exception):
    def __init__(self, status: int, detail: object) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail

    @property
    def reason(self) -> str | None:
        d = self.detail
        if isinstance(d, dict) and isinstance(d.get("detail"), dict):
            return d["detail"].get("reason")
        return None


def _call(base: str, method: str, path: str, body: dict | None = None, timeout: float = 10.0) -> dict | None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read() or b"null")
        except ValueError:
            detail = None
        raise _HttpError(e.code, detail) from None
    except (urllib.error.URLError, OSError, TimeoutError) as e:  # refused, DNS, timeout, reset
        raise _Unreachable(str(getattr(e, "reason", e))) from None


def _default_holder() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class Lease:
    """A booking of GPU memory. Renewed in the background while held.

    `guarded` is False when the job runs without a lease (gatekeeper unreachable).
    """

    def __init__(
        self,
        url: str,
        info: dict | None,
        renew_s: float | None = None,
        cap_torch: bool = True,
    ) -> None:
        self.url = url
        self.info = info or {}
        self.id: str | None = self.info.get("id")
        self.gb: float | None = self.info.get("gb")
        self.guarded = info is not None
        self.expired = False
        self.released = False
        self._renew_s = renew_s or float(self.info.get("renew_every_s") or DEFAULT_RENEW_S)
        self._cap_torch = cap_torch
        self._torch_capped = False
        self._loaded_pending = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:
        state = "unguarded" if not self.guarded else ("released" if self.released else "held")
        return f"<Lease {self.id} {self.gb} GB {state}>"

    # -- lifecycle --------------------------------------------------------

    def _start(self) -> None:
        if not self.guarded:
            return
        if self._cap_torch:
            self._torch_capped = _set_torch_fraction(self.gb)
        self._thread = threading.Thread(target=self._renew_loop, name=f"gpu-lease-{self.id}", daemon=True)
        self._thread.start()

    def _renew_loop(self) -> None:
        while not self._stop.wait(self._renew_s):
            self._renew()
            if self.expired:
                return

    def _renew(self) -> bool:
        """One renewal; sends a pending `loaded` flag. Returns True if it reached the gatekeeper."""
        with self._lock:
            if self.released or self.expired:
                return False
            body = {"loaded": True} if self._loaded_pending else {}
            try:
                self.info = _call(self.url, "POST", f"/leases/{self.id}/renew", body) or self.info
                if body:
                    self._loaded_pending = False
                return True
            except _HttpError as e:
                if e.status == 404:
                    self.expired = True
                    log.error(
                        "!!! GPU lease %s is GONE (%s) — the gatekeeper no longer counts this job's "
                        "memory; other jobs may now be started on top of it. Renewals stopped. !!!",
                        self.id,
                        e.reason or "not found",
                    )
                else:
                    log.warning("GPU lease %s renew failed: %s", self.id, e)
            except _Unreachable as e:
                log.warning("GPU lease %s renew failed, gatekeeper unreachable (%s); will retry", self.id, e)
            return False

    def mark_loaded(self) -> None:
        """Tell the gatekeeper this job's model is on the GPU now. Sent immediately;
        if that fails it is retried with the next background renewal."""
        if not self.guarded:
            return
        with self._lock:
            self._loaded_pending = True
        self._renew()

    def release(self) -> None:
        """Give the memory back. Safe to call more than once."""
        with self._lock:
            if self.released:
                return
            self.released = True
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        if self._torch_capped:
            _set_torch_fraction(None)
            self._torch_capped = False
        if not self.guarded or self.expired:
            return
        try:
            _call(self.url, "DELETE", f"/leases/{self.id}")
        except _HttpError as e:
            if e.status != 404:
                log.warning("GPU lease %s release failed: %s", self.id, e)
        except _Unreachable as e:
            log.warning("GPU lease %s release failed, gatekeeper unreachable (%s); it will expire", self.id, e)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _set_torch_fraction(lease_gb: float | None) -> bool:
    """Cap this process's PyTorch allocator to the lease size (None resets to 1.0).
    Returns True if a cap was applied. Torch is optional; any failure is ignored."""
    try:
        if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
            return False
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return False
        dev = torch.cuda.current_device()
        if lease_gb is None:
            torch.cuda.set_per_process_memory_fraction(1.0, dev)
            return False
        total = torch.cuda.get_device_properties(dev).total_memory
        fraction = min(1.0, lease_gb * GIB / total)
        torch.cuda.set_per_process_memory_fraction(fraction, dev)
        log.info("PyTorch memory capped to %.2f of the GPU (%.1f GB lease)", fraction, lease_gb)
        return True
    except Exception as e:  # noqa: BLE001 - the cap is a safety net, never a failure
        log.warning("could not set PyTorch memory cap: %s", e)
        return False


def _unguarded(url: str, why: str) -> Lease:
    log.warning("gatekeeper unreachable — running unguarded (%s: %s)", url, why)
    return Lease(url, None)


def acquire(
    profile: str | None = None,
    *,
    gb: float | None = None,
    holder: str | None = None,
    service: str | None = None,
    timeout: float | None = None,
    url: str | None = None,
    cap_torch: bool = True,
    unreachable_wait: float = UNREACHABLE_WAIT_S,
    renew_s: float | None = None,
    cancel: Callable[[], bool] | None = None,
) -> Lease:
    """Request a lease and block until it is granted.

    profile   job profile name (its size is used unless `gb` is given)
    gb        explicit size in GB, overrides the profile size
    holder    label shown in the gatekeeper status (default "<hostname>:<pid>")
    service   allowlist name of the service running this job, if any
    timeout   max seconds to wait in the queue; None waits forever.
              On timeout the queued request is withdrawn and GpuBusyError is raised.
    url       gatekeeper URL (default: env GPU_GATEKEEPER_URL, else DEFAULT_URL)
    cap_torch cap PyTorch's allocator to the lease size if torch+CUDA are present
    cancel    optional check, called while queued; when it returns True the queued
              request is withdrawn and GpuLeaseCancelled is raised (e.g. the user
              cancelled the job while it waited for the GPU)

    Raises GpuLeaseError if the gatekeeper refuses the request (unknown profile,
    too large to ever fit, missing profile/gb). If the gatekeeper is unreachable
    for `unreachable_wait` seconds, returns an unguarded Lease (fail-open).
    """
    url = url or os.environ.get("GPU_GATEKEEPER_URL") or DEFAULT_URL
    body = {"profile": profile, "gb": gb, "holder": holder or _default_holder(), "service": service}
    body = {k: v for k, v in body.items() if v is not None}
    deadline = None if timeout is None else time.monotonic() + timeout

    def post() -> dict | None:
        """POST /leases, retrying connection failures for `unreachable_wait` s. None = unreachable."""
        give_up = time.monotonic() + unreachable_wait
        while True:
            try:
                return _call(url, "POST", "/leases", body)
            except _HttpError as e:
                raise GpuLeaseError(f"gatekeeper refused lease request: {e}", e.status, e.detail) from None
            except _Unreachable as e:
                if time.monotonic() >= give_up:
                    post.last_error = str(e)  # type: ignore[attr-defined]
                    return None
                time.sleep(min(1.0, max(0.05, give_up - time.monotonic())))

    info = post()
    if info is None:
        return _unguarded(url, getattr(post, "last_error", "?"))

    unreachable_since: float | None = None
    announced = False
    while info.get("state") != "granted":
        if not announced:
            log.info("GPU busy: lease %s queued (position %s), waiting", info.get("id"), info.get("position"))
            announced = True
        wait = POLL_WAIT_S
        if cancel is not None:
            if cancel():
                try:
                    _call(url, "DELETE", f"/leases/{info['id']}")
                except (_HttpError, _Unreachable):
                    pass
                raise GpuLeaseCancelled(f"GPU lease {info.get('id')} withdrawn: cancelled while queued")
            wait = CANCEL_POLL_S
        if deadline is not None:
            left = deadline - time.monotonic()
            if left <= 0:
                try:
                    _call(url, "DELETE", f"/leases/{info['id']}")
                except (_HttpError, _Unreachable):
                    pass
                raise GpuBusyError(
                    f"GPU busy: lease for {profile or f'{gb} GB'} still queued after {timeout:g}s "
                    f"(position {info.get('position')})"
                )
            wait = min(wait, left)
        try:
            info = _call(url, "GET", f"/leases/{info['id']}?wait={wait:.1f}", timeout=wait + 10)
            unreachable_since = None
        except _HttpError as e:
            if e.status == 404:  # lost while queued (e.g. gatekeeper restarted): ask again
                log.warning("queued GPU lease %s vanished (%s); requesting again", info.get("id"), e.reason)
                info = post()
                if info is None:
                    return _unguarded(url, "lost while queued")
            else:
                raise GpuLeaseError(f"gatekeeper error while queued: {e}", e.status, e.detail) from None
        except _Unreachable as e:
            now = time.monotonic()
            unreachable_since = unreachable_since or now
            if now - unreachable_since >= unreachable_wait:
                return _unguarded(url, str(e))
            time.sleep(1.0)

    lease = Lease(url, info, renew_s=renew_s, cap_torch=cap_torch)
    log.info("GPU lease %s granted: %s GB", lease.id, lease.gb)
    lease._start()
    return lease


@contextmanager
def gpu_lease(profile: str | None = None, **kwargs) -> Iterator[Lease]:
    """`with gpu_lease("flux-gen") as lease: ...` — see `acquire` for arguments.
    The lease is released when the block exits, including on exceptions."""
    lease = acquire(profile, **kwargs)
    try:
        yield lease
    finally:
        lease.release()
