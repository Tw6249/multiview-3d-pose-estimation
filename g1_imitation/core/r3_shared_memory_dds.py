"""Module R3: Lock-Free Seqlock Shared Memory Bridge & Unitree G1 (`unitree_hg`) Low-Level DDS Adapter.

Specification Reference: `自建硬件部署与前期执行手册.md` — 阶段五 4.5.1 & 4.5.2
Input (Writer / Process 1 - Vision + IK + R4 SafetyGovernor):
  - `SafeReferenceFrame` (`q_safe(36,)`, `dq_safe(35,)`, `safety_state_code`, `t_host_s`)
Output (Reader / Process 2 - 50 Hz SONIC + 500 Hz DDS Loop):
  - `SharedReferenceSnapshot`:
    - `seq`: `int` (even sequence number)
    - `timestamp_monotonic_us`: `int`
    - `safety_state_code`: `int`
    - `q_safe`: `np.ndarray` `(36,)` `float64`
    - `dq_safe`: `np.ndarray` `(35,)` `float64`
    - `age_ms`: `float` (age relative to reader's monotonic clock)
    - `watchdog_expired`: `bool` (`True` if `age_ms > watchdog_timeout_ms`, default `35.0 ms`)
Failure Behavior:
  - Tear-free Seqlock: writer increments `seq` to odd before writing and even after writing; reader retries if `seq`
    is odd or changed during read.
  - Watchdog: if `age_ms > 35.0 ms` (e.g., camera cable unplugged or Process 1 stalled), `watchdog_expired = True`
    and `UnitreeG1DdsBridge.compute_motor_commands` immediately damps/freezes the command.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import struct
import time
import zlib
import numpy as np

from .s6_mujoco_closed_loop import COMMON_DEFAULT_Q_29

# Memory layout:
#   offset 0:   seq_head (uint64, 8B)
#   offset 8:   timestamp_monotonic_us (int64, 8B)
#   offset 16:  safety_state_code (int32, 4B) + pad (4B)
#   offset 24:  q_safe (36 * float64 = 288B)
#   offset 312: dq_safe (35 * float64 = 280B)
#   offset 592: seq_tail (uint64, 8B)
# Total = 600 bytes
_HEADER_STRUCT = struct.Struct("<Qqii")
_Q_OFFSET = 24
_DQ_OFFSET = 24 + 36 * 8
_TAIL_OFFSET = _DQ_OFFSET + 35 * 8
SHM_BYTE_SIZE = _TAIL_OFFSET + 8


@dataclass(frozen=True)
class SharedReferenceSnapshot:
    seq: int
    timestamp_monotonic_us: int
    safety_state_code: int
    q_safe: np.ndarray
    dq_safe: np.ndarray
    age_ms: float
    watchdog_expired: bool


class SeqlockReferenceBuffer:
    """Lock-free Seqlock shared memory buffer between Process 1 (Vision/IK) and Process 2 (SONIC/DDS)."""

    def __init__(self, name: str | None = None, *, create: bool = True, watchdog_timeout_ms: float = 35.0) -> None:
        self.watchdog_timeout_ms = float(watchdog_timeout_ms)
        self._owns_shm = bool(create)
        if create:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=SHM_BYTE_SIZE)
            assert self.shm.buf is not None
            self.shm.buf[:SHM_BYTE_SIZE] = b"\x00" * SHM_BYTE_SIZE
        else:
            if not name:
                raise ValueError("SharedMemory name is required when create=False")
            self.shm = shared_memory.SharedMemory(name=name, create=False, size=SHM_BYTE_SIZE)
        self.name = self.shm.name
        self._seq = 0

    def write(
        self,
        q_safe: np.ndarray,
        dq_safe: np.ndarray,
        *,
        safety_state_code: int = 2,
        t_monotonic_s: float | None = None,
    ) -> int:
        q = np.ascontiguousarray(q_safe, dtype=np.float64)
        dq = np.ascontiguousarray(dq_safe, dtype=np.float64)
        if q.shape != (36,) or dq.shape != (35,):
            raise ValueError(f"Expected q(36,) and dq(35,), got {q.shape} and {dq.shape}")
        t_us = int(round((time.perf_counter() if t_monotonic_s is None else float(t_monotonic_s)) * 1e6))
        buf = self.shm.buf
        assert buf is not None
        odd_seq = self._seq + 1
        even_seq = self._seq + 2
        struct.pack_into("<Q", buf, 0, odd_seq)
        struct.pack_into("<Q", buf, _TAIL_OFFSET, odd_seq)
        struct.pack_into("<qii", buf, 8, t_us, int(safety_state_code), 0)
        buf[_Q_OFFSET:_DQ_OFFSET] = q.tobytes()
        buf[_DQ_OFFSET:_TAIL_OFFSET] = dq.tobytes()
        struct.pack_into("<Q", buf, _TAIL_OFFSET, even_seq)
        struct.pack_into("<Q", buf, 0, even_seq)
        self._seq = even_seq
        return even_seq

    def read(self, *, now_monotonic_s: float | None = None, max_retries: int = 16) -> SharedReferenceSnapshot:
        buf = self.shm.buf
        assert buf is not None
        for _ in range(max_retries):
            seq1, t_us, state_code, _ = _HEADER_STRUCT.unpack_from(buf, 0)
            if seq1 == 0 or (seq1 & 1) == 1:
                continue
            q = np.frombuffer(buf, dtype=np.float64, count=36, offset=_Q_OFFSET).copy()
            dq = np.frombuffer(buf, dtype=np.float64, count=35, offset=_DQ_OFFSET).copy()
            (seq2,) = struct.unpack_from("<Q", buf, _TAIL_OFFSET)
            if seq1 == seq2 and (seq2 & 1) == 0:
                now_us = int(round((time.perf_counter() if now_monotonic_s is None else float(now_monotonic_s)) * 1e6))
                age_ms = float(now_us - t_us) * 1e-3
                expired = bool(age_ms > self.watchdog_timeout_ms or age_ms < -1.0)
                return SharedReferenceSnapshot(
                    seq=int(seq1),
                    timestamp_monotonic_us=int(t_us),
                    safety_state_code=int(state_code),
                    q_safe=q,
                    dq_safe=dq,
                    age_ms=age_ms,
                    watchdog_expired=expired,
                )
        raise RuntimeError("Seqlock read exceeded max_retries")

    def close(self) -> None:
        self.shm.close()
        if self._owns_shm:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class G1LowCmdPacket:
    mode_pr: int
    mode_machine: int
    q_target_29: np.ndarray
    dq_target_29: np.ndarray
    kp_29: np.ndarray
    kd_29: np.ndarray
    tau_ff_29: np.ndarray
    crc32: int
    damping_mode: bool


def compute_unitree_crc32(payload_bytes: bytes) -> int:
    """Compute standard 32-bit CRC checksum over packed command payload."""
    return int(zlib.crc32(payload_bytes) & 0xFFFFFFFF)


class UnitreeG1DdsBridge:
    """Stage-1/2/3 safe command builder for Unitree G1 EDU 29-DoF (`unitree_hg` SDK ordering)."""

    def __init__(
        self,
        kp_29: np.ndarray,
        kd_29: np.ndarray,
        *,
        damping_kd_29: np.ndarray | None = None,
        stage1_zero_gain: bool = False,
    ) -> None:
        self.kp_29 = np.asarray(kp_29, dtype=np.float64).copy()
        self.kd_29 = np.asarray(kd_29, dtype=np.float64).copy()
        if self.kp_29.shape != (29,) or self.kd_29.shape != (29,):
            raise ValueError("kp_29 and kd_29 must have shape (29,)")
        self.damping_kd_29 = (
            np.full(29, 3.0, dtype=np.float64)
            if damping_kd_29 is None
            else np.asarray(damping_kd_29, dtype=np.float64).copy()
        )
        self.stage1_zero_gain = bool(stage1_zero_gain)

    def build_low_cmd(
        self,
        q_des_29: np.ndarray,
        snapshot: SharedReferenceSnapshot,
        *,
        q_measured_29: np.ndarray | None = None,
        mode_pr: int = 0,
        mode_machine: int = 5,
    ) -> G1LowCmdPacket:
        q_des = np.asarray(q_des_29, dtype=np.float64)
        if q_des.shape != (29,) or not np.all(np.isfinite(q_des)):
            raise ValueError("q_des_29 must be finite of shape (29,)")
        if self.stage1_zero_gain:
            kp = np.zeros(29, dtype=np.float64)
            kd = np.zeros(29, dtype=np.float64)
            q_cmd = q_des.copy()
            damping = True
        elif snapshot.watchdog_expired:
            # Stage-2/3 watchdog trip (< 35 ms): zero Kp, pure damping Kd around current measured or default pose
            kp = np.zeros(29, dtype=np.float64)
            kd = self.damping_kd_29.copy()
            q_cmd = COMMON_DEFAULT_Q_29.copy() if q_measured_29 is None else np.asarray(q_measured_29, dtype=np.float64).copy()
            damping = True
        else:
            kp = self.kp_29.copy()
            kd = self.kd_29.copy()
            q_cmd = q_des.copy()
            damping = False

        dq_cmd = np.zeros(29, dtype=np.float64)
        tau_ff = np.zeros(29, dtype=np.float64)
        payload = struct.pack("<BB", int(mode_pr), int(mode_machine)) + q_cmd.astype(np.float32).tobytes() + kp.astype(np.float32).tobytes() + kd.astype(np.float32).tobytes()
        crc = compute_unitree_crc32(payload)
        return G1LowCmdPacket(
            mode_pr=int(mode_pr),
            mode_machine=int(mode_machine),
            q_target_29=q_cmd,
            dq_target_29=dq_cmd,
            kp_29=kp,
            kd_29=kd,
            tau_ff_29=tau_ff,
            crc32=crc,
            damping_mode=damping,
        )
