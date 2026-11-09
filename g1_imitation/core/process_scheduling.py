"""Opt the current process out of Windows execution-speed power throttling (EcoQoS).

Observed 2026-11-09 on the i9-13900KF host: a background console process slows about 5x after roughly 3 s
(a pure Python loop 13 ms -> 65 ms; keypoint IK p50 1.3 ms -> 11.9 ms). Any latency figure measured in a
throttled process is invalid. Input: none. Output: True if the OS accepted the request, False on other
platforms or if the call is refused (the caller must then report its timings as unverified).
"""
from __future__ import annotations

import sys

_PROCESS_POWER_THROTTLING = 4
_THROTTLING_EXECUTION_SPEED = 0x1


def disable_power_throttling() -> bool:
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetProcessInformation.restype = wintypes.BOOL
    state = _State(1, _THROTTLING_EXECUTION_SPEED, 0)
    return bool(kernel.SetProcessInformation(kernel.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
                                             ctypes.byref(state), ctypes.sizeof(state)))
