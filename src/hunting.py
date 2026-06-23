"""
hunting.py
==========
Insider Threat Detection & User Risk Scoring Platform
Threat Hunting Module — Rule-Based Behavioral Sequence Detection

Author  : Detection Engineering Team
Version : 2.0.0
Dataset : processed_logs.csv (7,400 events | 30-day window | ~5,031 users)

Overview
--------
This module performs timeline-based threat hunting on preprocessed corporate
activity logs.  It reconstructs per-user event sequences and detects suspicious
behavioral chains that are invisible when events are evaluated in isolation.

Design Philosophy
-----------------
- Detect *sequences*, not single events.
- Thresholds are top-level constants — easy to tune without touching logic.
- Every detection function is independent and idempotent.
- Defensive: handles missing columns, NaN values, and malformed timestamps.
- Output is always a uniform list[dict] → aggregated into a single DataFrame.

Supported Hunts
---------------
  Hunt 1  Brute Force Chain                          (High)
  Hunt 2  USB Followed By File Access                (Medium)
  Hunt 3  Potential Data Theft Chain                 (Critical)
  Hunt 4  Suspicious Activity After Remote Login     (High)
  Hunt 5  Off-Hours Login Followed By Sensitive Action (Medium)
  Hunt 6  Rapid IP Switching                         (High)
  Hunt 7  Multi-Anomaly Stacking                     (Critical)
  Hunt 8  High-Velocity File Access                  (Medium)
  Hunt 9  New-IP Remote Login Followed By Exfiltration (Critical)

Dataset Notes (calibrated from actual data)
-------------------------------------------
- action values : login | remote login | usb insert | file access |
                  exfiltration | network traffic
- login_attempts: number of failed attempts encoded in a single login row
                  (0 never occurs; 1-10 range)
- file_size     : unit is MB (range 0-50 MB; 0 means non-file event)
- timestamp     : events span ~30 days; median inter-event gap per user ~7 days
- Time windows for sequence hunts use DAYS because the dataset is spread
  across a 30-day observation window. Constants document the unit.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("threat_hunter")


# ===========================================================================
# ── CONFIGURABLE DETECTION THRESHOLDS ──────────────────────────────────────
# ===========================================================================

# Hunt 1 — Brute Force Chain
BRUTE_FORCE_MIN_ATTEMPTS: int = 3
"""Minimum failed login attempts (login_attempts column) to flag as brute force."""

# Hunt 2 — USB Followed By File Access
USB_FILE_WINDOW_DAYS: float = 3.0
"""Max days between USB insert and subsequent file access to trigger the alert."""

# Hunt 3 — Large File Access -> Exfiltration
LARGE_FILE_SIZE_MB: float = 30.0
"""File size threshold in MB above which a file access is considered 'large'."""
EXFIL_WINDOW_DAYS: float = 7.0
"""Max days between large file access and subsequent exfiltration event."""

# Hunt 4 — Remote Login -> Suspicious Activity
REMOTE_LOGIN_WINDOW_DAYS: float = 7.0
"""Max days between remote login and a subsequent suspicious action."""
SUSPICIOUS_ACTIONS_AFTER_REMOTE: set[str] = {"usb insert", "exfiltration"}
"""Action strings that trigger Hunt 4 when they follow a remote login."""

# Hunt 5 — Off-Hours Login -> Sensitive Action
OFF_HOURS_START: int = 22     # 10 PM  (inclusive)
OFF_HOURS_END: int = 6        # 6 AM   (exclusive, so 00:00-05:59 is off-hours)
OFF_HOURS_FOLLOW_WINDOW_DAYS: float = 3.0
"""Max days between an off-hours login and a sensitive follow-up action."""
OFF_HOURS_SENSITIVE_ACTIONS: set[str] = {"file access", "exfiltration", "usb insert"}
"""Actions considered sensitive when they follow an off-hours login."""

# Hunt 6 — Rapid IP Switching
IP_SWITCH_WINDOW_DAYS: float = 3.0
"""Max days between two login events from different IPs to flag as rapid switching."""

# Hunt 7 — Multi-Anomaly Stacking
ANOMALY_STACK_MIN_TYPES: int = 2
"""Minimum number of distinct anomaly_type values per user to trigger stacking alert."""

# Hunt 8 — High-Velocity File Access
FILE_ACCESS_SIGMA_MULTIPLIER: float = 2.0
"""Flag users whose file-access count exceeds mean + N*std across all users."""
FILE_ACCESS_ABSOLUTE_MIN: int = 3
"""Hard floor: never flag users with fewer than this many file access events,
   regardless of the sigma calculation."""

# Hunt 9 — New-IP Remote Login -> Exfiltration
NEW_IP_EXFIL_WINDOW_DAYS: float = 7.0
"""Max days between a first-seen-IP remote login and subsequent exfiltration."""


# ===========================================================================
# ── INTERNAL HELPERS ────────────────────────────────────────────────────────
# ===========================================================================

_REQUIRED_COLS = {"user_id", "timestamp", "action"}


def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the input DataFrame and return a clean, sorted working copy.

    Steps
    -----
    1. Assert required columns are present.
    2. Parse 'timestamp' to datetime (errors -> NaT, dropped later).
    3. Drop rows with NaT timestamps or null user_id / action.
    4. Sort by [user_id, timestamp] to reconstruct timelines.
    5. Normalise 'action' to lowercase-stripped strings.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame from the caller.

    Returns
    -------
    pd.DataFrame
        Cleaned, sorted copy of df.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    work = df.copy()

    # Normalise action strings for robust matching
    work["action"] = work["action"].astype(str).str.strip().str.lower()

    # Parse timestamps defensively
    if not pd.api.types.is_datetime64_any_dtype(work["timestamp"]):
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")

    # Drop unusable rows
    before = len(work)
    work = work.dropna(subset=["timestamp", "user_id", "action"])
    dropped = before - len(work)
    if dropped:
        logger.warning("Dropped %d rows with null timestamp / user_id / action.", dropped)

    # Sort to reconstruct per-user timeline
    work = work.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return work


def _safe_float(value) -> float:
    """Return float(value) or 0.0 on any conversion error."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value) -> int:
    """Return int(value) or 0 on any conversion error."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_off_hours(ts: pd.Timestamp, start: int = OFF_HOURS_START, end: int = OFF_HOURS_END) -> bool:
    """
    Return True if the timestamp falls within the off-hours window.

    Off-hours spans two calendar ranges:
        [start, 23] and [0, end-1]
    Example with start=22, end=6:
        22:00 -> 23:59  and  00:00 -> 05:59
    """
    h = ts.hour
    return h >= start or h < end


def _build_finding(
    user_id,
    detection_time: pd.Timestamp,
    pattern: str,
    severity: str,
    details: str,
) -> dict:
    """
    Construct a standardised finding dictionary.

    All detection functions return a list of these dicts, which are later
    aggregated by run_all_hunts() into a single DataFrame.

    Parameters
    ----------
    user_id        : Identifier of the flagged user.
    detection_time : Timestamp of the triggering event.
    pattern        : Human-readable pattern name.
    severity       : One of Critical | High | Medium | Low.
    details        : Free-text description of what was detected.

    Returns
    -------
    dict
        Standardised finding record.
    """
    return {
        "user_id": user_id,
        "detection_time": detection_time.strftime("%Y-%m-%d %H:%M:%S"),
        "pattern": pattern,
        "severity": severity,
        "details": details,
    }


# ===========================================================================
# ── HUNT 1 — BRUTE FORCE CHAIN ──────────────────────────────────────────────
# ===========================================================================

def detect_bruteforce_chain(
    df: pd.DataFrame,
    min_attempts: int = BRUTE_FORCE_MIN_ATTEMPTS,
) -> list[dict]:
    """
    Hunt 1 — Brute Force Chain (Severity: High)

    Detection Logic
    ---------------
    Each 'login' row carries a login_attempts counter encoding the number of
    failed attempts before the final login event. A high counter is the
    behavioural equivalent of observing:

        Failed -> Failed -> Failed -> Success

    We flag every 'login' row where login_attempts >= min_attempts.

    Parameters
    ----------
    df           : Prepared DataFrame (output of _validate_and_prepare).
    min_attempts : Minimum failed attempts threshold (default: 3).

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []

    if "login_attempts" not in df.columns:
        logger.warning("Hunt 1 skipped — 'login_attempts' column not found.")
        return findings

    login_rows = df[df["action"] == "login"].copy()
    if login_rows.empty:
        return findings

    login_rows["login_attempts"] = login_rows["login_attempts"].apply(_safe_int)
    bf_rows = login_rows[login_rows["login_attempts"] >= min_attempts]

    logger.info(
        "Hunt 1 — Brute Force Chain: found %d event(s) with >= %d failed attempts.",
        len(bf_rows), min_attempts,
    )

    for _, row in bf_rows.iterrows():
        attempts = _safe_int(row["login_attempts"])
        findings.append(_build_finding(
            user_id=row["user_id"],
            detection_time=row["timestamp"],
            pattern="Brute Force Chain",
            severity="High",
            details=(
                f"{attempts} failed login attempt(s) recorded before successful "
                f"authentication (threshold: {min_attempts}). "
                f"Source IP: {row.get('ip_address', 'N/A')}."
            ),
        ))

    return findings


# ===========================================================================
# ── HUNT 2 — USB INSERT -> FILE ACCESS ──────────────────────────────────────
# ===========================================================================

def detect_usb_file_chain(
    df: pd.DataFrame,
    window_days: float = USB_FILE_WINDOW_DAYS,
) -> list[dict]:
    """
    Hunt 2 — USB Followed By File Access (Severity: Medium)

    Detection Logic
    ---------------
    For each user, whenever a 'usb insert' event is found, look ahead for
    the next 'file access' within window_days.

        usb insert
            |  (within window_days)
        file access

    Potential interpretation: user plugged in removable media then accessed
    files — consistent with data staging for theft.

    Parameters
    ----------
    df          : Prepared DataFrame.
    window_days : Max allowed gap between USB insert and file access (days).

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []
    window_td = pd.Timedelta(days=window_days)

    for user_id, user_df in df.groupby("user_id"):
        user_events = user_df.reset_index(drop=True)
        usb_events = user_events[user_events["action"] == "usb insert"]
        if usb_events.empty:
            continue

        for usb_idx, usb_row in usb_events.iterrows():
            usb_time = usb_row["timestamp"]
            subsequent = user_events[
                (user_events.index > usb_idx)
                & (user_events["action"] == "file access")
                & (user_events["timestamp"] - usb_time <= window_td)
                & (user_events["timestamp"] > usb_time)
            ]
            if not subsequent.empty:
                fa_row = subsequent.iloc[0]
                delta_hours = (fa_row["timestamp"] - usb_time).total_seconds() / 3600
                file_sz = _safe_float(fa_row.get("file_size", 0))
                findings.append(_build_finding(
                    user_id=user_id,
                    detection_time=fa_row["timestamp"],
                    pattern="USB Followed By File Access",
                    severity="Medium",
                    details=(
                        f"USB insert at {usb_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"followed by file access {delta_hours:.1f}h later "
                        f"(file_size: {file_sz:.2f} MB). "
                        f"Window threshold: {window_days} day(s)."
                    ),
                ))

    logger.info("Hunt 2 — USB Followed By File Access: found %d finding(s).", len(findings))
    return findings


# ===========================================================================
# ── HUNT 3 — LARGE FILE ACCESS -> EXFILTRATION ──────────────────────────────
# ===========================================================================

def detect_exfiltration_chain(
    df: pd.DataFrame,
    file_size_threshold_mb: float = LARGE_FILE_SIZE_MB,
    window_days: float = EXFIL_WINDOW_DAYS,
) -> list[dict]:
    """
    Hunt 3 — Potential Data Theft Chain (Severity: Critical)

    Detection Logic
    ---------------
    For each user, find 'file access' events where file_size exceeds the
    threshold, then look ahead for an 'exfiltration' event within window_days.

        file access  (size > threshold_mb)
            |  (within window_days)
        exfiltration

    Parameters
    ----------
    df                     : Prepared DataFrame.
    file_size_threshold_mb : Minimum file size in MB to flag as 'large'.
    window_days            : Max gap between large file access and exfiltration.

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []

    if "file_size" not in df.columns:
        logger.warning("Hunt 3 skipped — 'file_size' column not found.")
        return findings

    window_td = pd.Timedelta(days=window_days)

    for user_id, user_df in df.groupby("user_id"):
        user_events = user_df.reset_index(drop=True).copy()
        user_events["file_size"] = user_events["file_size"].apply(_safe_float)

        large_file_events = user_events[
            (user_events["action"] == "file access")
            & (user_events["file_size"] >= file_size_threshold_mb)
        ]
        if large_file_events.empty:
            continue

        for fa_idx, fa_row in large_file_events.iterrows():
            fa_time = fa_row["timestamp"]
            exfil_events = user_events[
                (user_events.index > fa_idx)
                & (user_events["action"] == "exfiltration")
                & (user_events["timestamp"] - fa_time <= window_td)
                & (user_events["timestamp"] > fa_time)
            ]
            if not exfil_events.empty:
                exfil_row = exfil_events.iloc[0]
                delta_hours = (exfil_row["timestamp"] - fa_time).total_seconds() / 3600
                findings.append(_build_finding(
                    user_id=user_id,
                    detection_time=exfil_row["timestamp"],
                    pattern="Potential Data Theft Chain",
                    severity="Critical",
                    details=(
                        f"Large file access ({fa_row['file_size']:.2f} MB, "
                        f"threshold: {file_size_threshold_mb} MB) at "
                        f"{fa_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"followed by exfiltration {delta_hours:.1f}h later. "
                        f"Exfil IP: {exfil_row.get('ip_address', 'N/A')}."
                    ),
                ))

    logger.info("Hunt 3 — Potential Data Theft Chain: found %d finding(s).", len(findings))
    return findings


# ===========================================================================
# ── HUNT 4 — REMOTE LOGIN -> SUSPICIOUS ACTIVITY ────────────────────────────
# ===========================================================================

def detect_remote_login_chain(
    df: pd.DataFrame,
    window_days: float = REMOTE_LOGIN_WINDOW_DAYS,
    suspicious_actions: Optional[set[str]] = None,
) -> list[dict]:
    """
    Hunt 4 — Suspicious Activity After Remote Login (Severity: High)

    Detection Logic
    ---------------
    For each user, find every 'remote login' event, then look ahead for any
    action in suspicious_actions within window_days.

        remote login  ->  usb insert        (data theft via removable media)
        remote login  ->  exfiltration      (direct data exfiltration)

    Parameters
    ----------
    df                 : Prepared DataFrame.
    window_days        : Max gap between remote login and suspicious event.
    suspicious_actions : Set of action strings considered suspicious.

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []

    if suspicious_actions is None:
        suspicious_actions = SUSPICIOUS_ACTIONS_AFTER_REMOTE

    suspicious_actions_lower = {a.lower() for a in suspicious_actions}
    window_td = pd.Timedelta(days=window_days)

    for user_id, user_df in df.groupby("user_id"):
        user_events = user_df.reset_index(drop=True)
        remote_logins = user_events[user_events["action"] == "remote login"]
        if remote_logins.empty:
            continue

        for rl_idx, rl_row in remote_logins.iterrows():
            rl_time = rl_row["timestamp"]
            after = user_events[
                (user_events.index > rl_idx)
                & (user_events["action"].isin(suspicious_actions_lower))
                & (user_events["timestamp"] - rl_time <= window_td)
                & (user_events["timestamp"] > rl_time)
            ]
            if not after.empty:
                susp_row = after.iloc[0]
                delta_hours = (susp_row["timestamp"] - rl_time).total_seconds() / 3600
                findings.append(_build_finding(
                    user_id=user_id,
                    detection_time=susp_row["timestamp"],
                    pattern="Suspicious Activity After Remote Login",
                    severity="High",
                    details=(
                        f"Remote login from {rl_row.get('ip_address', 'N/A')} "
                        f"at {rl_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"followed by '{susp_row['action']}' "
                        f"{delta_hours:.1f}h later "
                        f"(within {window_days}-day window). "
                        f"Suspicious event IP: {susp_row.get('ip_address', 'N/A')}."
                    ),
                ))

    logger.info(
        "Hunt 4 — Suspicious Activity After Remote Login: found %d finding(s).",
        len(findings),
    )
    return findings


# ===========================================================================
# ── HUNT 5 — OFF-HOURS LOGIN -> SENSITIVE ACTION ────────────────────────────
# ===========================================================================

def detect_off_hours_login_chain(
    df: pd.DataFrame,
    off_hours_start: int = OFF_HOURS_START,
    off_hours_end: int = OFF_HOURS_END,
    window_days: float = OFF_HOURS_FOLLOW_WINDOW_DAYS,
    sensitive_actions: Optional[set[str]] = None,
) -> list[dict]:
    """
    Hunt 5 — Off-Hours Login Followed By Sensitive Action (Severity: Medium)

    Detection Logic
    ---------------
    Employees logging in during off-hours (default: 22:00-05:59) and then
    performing a sensitive action (file access, exfiltration, USB insert)
    within window_days is a classic insider threat indicator.

        login / remote login  (hour in [22..23] or [00..05])
            |  (within window_days)
        file access | exfiltration | usb insert

    Dataset calibration: 660 users had off-hours logins; 36 of those were
    followed by a sensitive action within 3 days.

    Parameters
    ----------
    df                : Prepared DataFrame.
    off_hours_start   : Hour that begins the off-hours window (inclusive, 24h).
    off_hours_end     : Hour that ends the off-hours window (exclusive, 24h).
    window_days       : Max days between the off-hours login and follow-up action.
    sensitive_actions : Set of action strings considered sensitive.

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []

    if sensitive_actions is None:
        sensitive_actions = OFF_HOURS_SENSITIVE_ACTIONS

    sensitive_lower = {a.lower() for a in sensitive_actions}
    window_td = pd.Timedelta(days=window_days)

    for user_id, user_df in df.groupby("user_id"):
        user_events = user_df.reset_index(drop=True)

        # Identify all login events that fall within off-hours
        login_mask = user_events["action"].isin(["login", "remote login"])
        login_events = user_events[login_mask]
        if login_events.empty:
            continue

        for l_idx, l_row in login_events.iterrows():
            # Check if this login is within the off-hours window
            if not _is_off_hours(l_row["timestamp"], off_hours_start, off_hours_end):
                continue

            login_time = l_row["timestamp"]
            login_hour = login_time.hour

            # Look ahead for the next sensitive action within window_days
            after = user_events[
                (user_events.index > l_idx)
                & (user_events["action"].isin(sensitive_lower))
                & (user_events["timestamp"] - login_time <= window_td)
                & (user_events["timestamp"] > login_time)
            ]

            if not after.empty:
                sens_row = after.iloc[0]
                delta_hours = (sens_row["timestamp"] - login_time).total_seconds() / 3600
                findings.append(_build_finding(
                    user_id=user_id,
                    detection_time=sens_row["timestamp"],
                    pattern="Off-Hours Login Followed By Sensitive Action",
                    severity="Medium",
                    details=(
                        f"Login at {login_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"(hour: {login_hour:02d}:00, off-hours window: "
                        f"{off_hours_start:02d}:00-{off_hours_end:02d}:00) "
                        f"followed by '{sens_row['action']}' "
                        f"{delta_hours:.1f}h later. "
                        f"Login IP: {l_row.get('ip_address', 'N/A')}."
                    ),
                ))

    logger.info(
        "Hunt 5 — Off-Hours Login Followed By Sensitive Action: found %d finding(s).",
        len(findings),
    )
    return findings


# ===========================================================================
# ── HUNT 6 — RAPID IP SWITCHING ─────────────────────────────────────────────
# ===========================================================================

def detect_ip_switching(
    df: pd.DataFrame,
    window_days: float = IP_SWITCH_WINDOW_DAYS,
) -> list[dict]:
    """
    Hunt 6 — Rapid IP Switching (Severity: High)

    Detection Logic
    ---------------
    A user authenticating from two geographically or logistically distinct
    IP addresses within a short window may indicate:
      - Credential theft (attacker + legitimate user both active)
      - VPN anomaly or proxy abuse
      - Account sharing

    For each user we scan consecutive login events. If two logins from
    different IPs occur within window_days, we raise a finding.

        login  (IP: A)
            |  (within window_days)
        login  (IP: B, B != A)

    Dataset calibration: 207 users had 2+ different login IPs; 48 of those
    switched IPs within 3 days.

    Parameters
    ----------
    df          : Prepared DataFrame.
    window_days : Max gap (days) between two different-IP logins to alert.

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []

    if "ip_address" not in df.columns:
        logger.warning("Hunt 6 skipped — 'ip_address' column not found.")
        return findings

    window_td = pd.Timedelta(days=window_days)
    login_actions = {"login", "remote login"}

    for user_id, user_df in df.groupby("user_id"):
        user_events = user_df.reset_index(drop=True)

        # Keep only login events that have a valid IP
        login_events = user_events[
            user_events["action"].isin(login_actions)
            & user_events["ip_address"].notna()
        ].reset_index(drop=True)

        if len(login_events) < 2:
            continue

        # Scan consecutive login pairs for IP change within the window
        for i in range(len(login_events) - 1):
            row_a = login_events.iloc[i]
            row_b = login_events.iloc[i + 1]

            ip_a = str(row_a["ip_address"]).strip()
            ip_b = str(row_b["ip_address"]).strip()

            if ip_a == ip_b:
                continue  # Same IP — not suspicious

            time_a = row_a["timestamp"]
            time_b = row_b["timestamp"]

            if (time_b - time_a) <= window_td:
                delta_hours = (time_b - time_a).total_seconds() / 3600
                findings.append(_build_finding(
                    user_id=user_id,
                    detection_time=time_b,
                    pattern="Rapid IP Switching",
                    severity="High",
                    details=(
                        f"Login from IP {ip_a} at "
                        f"{time_a.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"followed by login from different IP {ip_b} "
                        f"only {delta_hours:.1f}h later "
                        f"(within {window_days}-day window). "
                        f"Possible credential theft or account sharing."
                    ),
                ))

    logger.info("Hunt 6 — Rapid IP Switching: found %d finding(s).", len(findings))
    return findings


# ===========================================================================
# ── HUNT 7 — MULTI-ANOMALY STACKING ─────────────────────────────────────────
# ===========================================================================

def detect_anomaly_stacking(
    df: pd.DataFrame,
    min_anomaly_types: int = ANOMALY_STACK_MIN_TYPES,
) -> list[dict]:
    """
    Hunt 7 — Multi-Anomaly Stacking (Severity: Critical)

    Detection Logic
    ---------------
    A user whose logs contain multiple distinct anomaly_type labels is
    exhibiting a composite threat profile — the combination of different
    attack-class signals is far more suspicious than any single one.

    Examples of dangerous stacks detected in this dataset:
        Brute_Force + Data_Exfil         (14 users)
        DDoS_Attempt + Data_Exfil        (12 users)
        Brute_Force + USB_Access         (9 users)

    We raise one finding per user who accumulates >= min_anomaly_types
    distinct anomaly types. The detection_time is the timestamp of the
    LAST anomaly event — the point at which stacking becomes confirmed.

    Dataset calibration: 58 users had 2+ distinct anomaly types;
    2 users had 3 distinct types (the maximum in this dataset).

    Parameters
    ----------
    df                 : Prepared DataFrame.
    min_anomaly_types  : Minimum distinct anomaly types to flag (default: 2).

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []

    if "anomaly_type" not in df.columns:
        logger.warning("Hunt 7 skipped — 'anomaly_type' column not found.")
        return findings

    # Work only on labeled rows
    labeled = df[df["anomaly_type"].notna()].copy()
    if labeled.empty:
        return findings

    for user_id, user_df in labeled.groupby("user_id"):
        distinct_types = user_df["anomaly_type"].unique().tolist()

        if len(distinct_types) >= min_anomaly_types:
            # Detection time = last anomaly event for this user
            last_event = user_df.sort_values("timestamp").iloc[-1]
            first_event = user_df.sort_values("timestamp").iloc[0]

            span_days = (
                last_event["timestamp"] - first_event["timestamp"]
            ).total_seconds() / 86400

            findings.append(_build_finding(
                user_id=user_id,
                detection_time=last_event["timestamp"],
                pattern="Multi-Anomaly Stacking",
                severity="Critical",
                details=(
                    f"User triggered {len(distinct_types)} distinct anomaly "
                    f"type(s): [{', '.join(sorted(distinct_types))}] "
                    f"over {span_days:.1f} days "
                    f"(threshold: {min_anomaly_types} types). "
                    f"Composite threat profile — high priority investigation."
                ),
            ))

    logger.info("Hunt 7 — Multi-Anomaly Stacking: found %d finding(s).", len(findings))
    return findings


# ===========================================================================
# ── HUNT 8 — HIGH-VELOCITY FILE ACCESS ──────────────────────────────────────
# ===========================================================================

def detect_high_velocity_file_access(
    df: pd.DataFrame,
    sigma_multiplier: float = FILE_ACCESS_SIGMA_MULTIPLIER,
    absolute_min: int = FILE_ACCESS_ABSOLUTE_MIN,
) -> list[dict]:
    """
    Hunt 8 — High-Velocity File Access (Severity: Medium)

    Detection Logic
    ---------------
    Abnormally high file access frequency relative to the user population
    baseline is a classic data enumeration / staging indicator.

    Method:
        1. Compute per-user file access count.
        2. Calculate population mean and standard deviation.
        3. Flag users whose count > mean + sigma_multiplier * std
           AND count >= absolute_min (hard floor to avoid flagging
           users who just happen to be 2-sigma above a near-zero baseline).

    Dataset calibration:
        Mean file accesses per user: 1.18
        Std:  0.43
        2-sigma threshold: ~2.04  ->  absolute_min of 3 is the effective floor
        Users flagged: 46 (all with 3-4 file access events)

    Parameters
    ----------
    df               : Prepared DataFrame.
    sigma_multiplier : Number of standard deviations above mean to flag.
    absolute_min     : Hard minimum count regardless of sigma calculation.

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []

    # Count file access events per user
    fa_df = df[df["action"] == "file access"].copy()
    if fa_df.empty:
        logger.warning("Hunt 8: no 'file access' events found.")
        return findings

    fa_counts = fa_df.groupby("user_id").size()

    # Population statistics
    pop_mean = fa_counts.mean()
    pop_std = fa_counts.std()
    sigma_threshold = pop_mean + sigma_multiplier * pop_std
    effective_threshold = max(sigma_threshold, float(absolute_min))

    logger.info(
        "Hunt 8 — File access baseline: mean=%.2f, std=%.2f, "
        "sigma threshold=%.2f, effective threshold=%.2f",
        pop_mean, pop_std, sigma_threshold, effective_threshold,
    )

    outlier_users = fa_counts[fa_counts >= effective_threshold].index

    for user_id in outlier_users:
        user_fa = fa_df[fa_df["user_id"] == user_id].sort_values("timestamp")
        count = len(user_fa)
        first_fa = user_fa.iloc[0]["timestamp"]
        last_fa = user_fa.iloc[-1]["timestamp"]
        span_hours = (last_fa - first_fa).total_seconds() / 3600

        # Total MB accessed
        total_mb = user_fa["file_size"].apply(_safe_float).sum() if "file_size" in user_fa.columns else 0.0

        findings.append(_build_finding(
            user_id=user_id,
            detection_time=last_fa,
            pattern="High-Velocity File Access",
            severity="Medium",
            details=(
                f"{count} file access event(s) recorded "
                f"(population threshold: {effective_threshold:.1f}, "
                f"mean: {pop_mean:.1f}). "
                f"Span: {span_hours:.1f}h, "
                f"total data accessed: {total_mb:.2f} MB. "
                f"Possible data enumeration or staging."
            ),
        ))

    logger.info("Hunt 8 — High-Velocity File Access: found %d finding(s).", len(findings))
    return findings


# ===========================================================================
# ── HUNT 9 — NEW-IP REMOTE LOGIN -> EXFILTRATION ────────────────────────────
# ===========================================================================

def detect_new_ip_exfiltration(
    df: pd.DataFrame,
    window_days: float = NEW_IP_EXFIL_WINDOW_DAYS,
) -> list[dict]:
    """
    Hunt 9 — New-IP Remote Login Followed By Exfiltration (Severity: Critical)

    Detection Logic
    ---------------
    This is a stronger variant of Hunt 4. Rather than flagging all remote
    logins followed by suspicious activity, we specifically flag cases where:
      1. The remote login originates from an IP address NEVER seen before
         for that user (first-time IP).
      2. An exfiltration event follows within window_days.

    This filters out legitimate remote workers (who regularly use the same
    VPN / home IP) and focuses on genuinely anomalous access patterns:
    a new entry point immediately followed by data theft.

        remote login  (IP never seen before for this user)
            |  (within window_days)
        exfiltration

    Dataset calibration: 28 users had both new-IP remote login and exfil;
    all 30 new-IP occurrences were followed by exfil within 30 days.
    14 of those occurred within 7 days.

    Parameters
    ----------
    df          : Prepared DataFrame.
    window_days : Max days between new-IP remote login and exfiltration.

    Returns
    -------
    list[dict]
    """
    findings: list[dict] = []

    if "ip_address" not in df.columns:
        logger.warning("Hunt 9 skipped — 'ip_address' column not found.")
        return findings

    window_td = pd.Timedelta(days=window_days)
    login_actions = {"login", "remote login"}

    for user_id, user_df in df.groupby("user_id"):
        user_events = user_df.reset_index(drop=True)
        seen_ips: set[str] = set()

        for i, row in user_events.iterrows():
            ip = str(row.get("ip_address", "")).strip()

            if row["action"] in login_actions:
                if ip and ip not in seen_ips:
                    # First-ever sighting of this IP for this user
                    login_time = row["timestamp"]

                    # Look ahead: is there an exfiltration within the window?
                    after_exfil = user_events[
                        (user_events.index > i)
                        & (user_events["action"] == "exfiltration")
                        & (user_events["timestamp"] - login_time <= window_td)
                        & (user_events["timestamp"] > login_time)
                    ]

                    if not after_exfil.empty:
                        exfil_row = after_exfil.iloc[0]
                        delta_hours = (
                            exfil_row["timestamp"] - login_time
                        ).total_seconds() / 3600

                        findings.append(_build_finding(
                            user_id=user_id,
                            detection_time=exfil_row["timestamp"],
                            pattern="New-IP Remote Login Followed By Exfiltration",
                            severity="Critical",
                            details=(
                                f"First-time login from IP {ip} "
                                f"at {login_time.strftime('%Y-%m-%d %H:%M:%S')} "
                                f"(action: {row['action']}) "
                                f"followed by exfiltration {delta_hours:.1f}h later "
                                f"(within {window_days}-day window). "
                                f"Exfil IP: {exfil_row.get('ip_address', 'N/A')}. "
                                f"New entry point — high confidence indicator."
                            ),
                        ))

                # Always mark this IP as seen after processing
                if ip:
                    seen_ips.add(ip)
            else:
                # Non-login event — still track the IP as seen
                if ip:
                    seen_ips.add(ip)

    logger.info(
        "Hunt 9 — New-IP Remote Login Followed By Exfiltration: found %d finding(s).",
        len(findings),
    )
    return findings


# ===========================================================================
# ── ORCHESTRATOR — run_all_hunts ────────────────────────────────────────────
# ===========================================================================

def run_all_hunts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute all nine threat hunts and return a consolidated findings DataFrame.

    Pipeline
    --------
    1. Validate and prepare the input DataFrame.
    2. Run all hunt functions sequentially.
    3. Aggregate all findings into a single list.
    4. Drop exact duplicates (user_id + detection_time + pattern).
    5. Sort by detection_time ascending.
    6. Log a severity breakdown summary.

    Parameters
    ----------
    df : pd.DataFrame
        Raw activity log DataFrame. Must contain at least:
        user_id, timestamp, action.

    Returns
    -------
    pd.DataFrame
        Columns: user_id | detection_time | pattern | severity | details
        Returns an empty DataFrame (with those columns) if no findings.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_csv("processed_logs.csv")
    >>> results = run_all_hunts(df)
    >>> print(results["pattern"].value_counts())
    """
    OUTPUT_COLUMNS = ["user_id", "detection_time", "pattern", "severity", "details"]

    logger.info("=" * 60)
    logger.info("Threat Hunting Engine v2.0 — starting run_all_hunts()")
    logger.info("Input shape: %s rows x %s cols", *df.shape)
    logger.info("=" * 60)

    # ── Step 1: Validate & prepare ─────────────────────────────────────────
    try:
        clean_df = _validate_and_prepare(df)
    except ValueError as exc:
        logger.error("Input validation failed: %s", exc)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    logger.info(
        "Prepared dataset: %d rows, %d unique users.",
        len(clean_df), clean_df["user_id"].nunique(),
    )

    # ── Step 2: Run each hunt ──────────────────────────────────────────────
    all_findings: list[dict] = []

    hunt_runners = [
        # Original four hunts
        detect_bruteforce_chain,           # Hunt 1
        detect_usb_file_chain,             # Hunt 2
        detect_exfiltration_chain,         # Hunt 3
        detect_remote_login_chain,         # Hunt 4
        # New hunts
        detect_off_hours_login_chain,      # Hunt 5
        detect_ip_switching,               # Hunt 6
        detect_anomaly_stacking,           # Hunt 7
        detect_high_velocity_file_access,  # Hunt 8
        detect_new_ip_exfiltration,        # Hunt 9
    ]

    for hunt_fn in hunt_runners:
        try:
            results = hunt_fn(clean_df)
            all_findings.extend(results)
        except Exception as exc:  # noqa: BLE001
            logger.error("Hunt '%s' raised an exception: %s", hunt_fn.__name__, exc)

    # ── Step 3-5: Aggregate, deduplicate, sort ─────────────────────────────
    if not all_findings:
        logger.warning("No findings generated by any hunt.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    results_df = pd.DataFrame(all_findings, columns=OUTPUT_COLUMNS)

    before_dedup = len(results_df)
    results_df = results_df.drop_duplicates(
        subset=["user_id", "detection_time", "pattern"]
    ).reset_index(drop=True)
    after_dedup = len(results_df)

    if before_dedup > after_dedup:
        logger.info("Removed %d duplicate finding(s).", before_dedup - after_dedup)

    results_df = results_df.sort_values("detection_time").reset_index(drop=True)

    # ── Step 6: Summary log ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Threat Hunting Complete — %d total finding(s).", len(results_df))
    for sev in ["Critical", "High", "Medium", "Low"]:
        count = (results_df["severity"] == sev).sum()
        if count:
            logger.info("  %-10s : %d", sev, count)
    logger.info("  Unique users flagged: %d", results_df["user_id"].nunique())
    logger.info("=" * 60)

    return results_df


# ===========================================================================
# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
# ===========================================================================

def save_results(results_df: pd.DataFrame, path: str = "hunting_results.csv") -> None:
    """
    Persist the findings DataFrame to CSV for downstream consumers
    (Risk Engine, Alert Engine, Dashboard).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_all_hunts().
    path       : Output file path. Default: 'hunting_results.csv'.
    """
    if results_df.empty:
        logger.warning("Nothing to save — results DataFrame is empty.")
        return

    results_df.to_csv(path, index=False)
    logger.info("Results saved -> %s  (%d rows)", path, len(results_df))


# ===========================================================================
# ── EXAMPLE USAGE (run as script) ───────────────────────────────────────────
# ===========================================================================

if __name__ == "__main__":
    import os

    DATA_PATH = "processed_logs.csv"

    if not os.path.exists(DATA_PATH):
        logger.error("Dataset not found at '%s'. Update DATA_PATH.", DATA_PATH)
        raise SystemExit(1)

    logger.info("Loading dataset from '%s' ...", DATA_PATH)
    raw_df = pd.read_csv(DATA_PATH)

    findings_df = run_all_hunts(raw_df)

    if findings_df.empty:
        print("\nNo findings generated.")
    else:
        print("\n" + "=" * 70)
        print(f"  THREAT HUNTING RESULTS  —  {len(findings_df)} finding(s)")
        print("=" * 70)

        for sev in ["Critical", "High", "Medium"]:
            subset = findings_df[findings_df["severity"] == sev]
            if subset.empty:
                continue
            print(f"\n{'─'*70}")
            print(f"  [{sev.upper()}]  {len(subset)} finding(s)")
            print(f"{'─'*70}")
            for _, row in subset.head(5).iterrows():   # cap display to 5 per severity
                print(f"  User    : {row['user_id']}")
                print(f"  Time    : {row['detection_time']}")
                print(f"  Pattern : {row['pattern']}")
                print(f"  Details : {row['details']}")
                print()
            if len(subset) > 5:
                print(f"  ... and {len(subset)-5} more {sev} finding(s).")

        save_results(findings_df, "hunting_results.csv")

        print("\n" + "=" * 70)
        print("  SUMMARY BY PATTERN")
        print("=" * 70)
        summary = (
            findings_df.groupby(["pattern", "severity"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print(summary.to_string(index=False))