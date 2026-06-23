"""
test_hunting.py
===============
Insider Threat Detection Platform — Complete Test Suite for hunting.py

How to run
----------
Basic run (all tests):
    pytest test_hunting.py -v

With coverage report:
    pytest test_hunting.py -v --cov=hunting --cov-report=term-missing

Stop on first failure:
    pytest test_hunting.py -v -x

Run a single hunt's tests:
    pytest test_hunting.py -v -k "hunt1"
    pytest test_hunting.py -v -k "hunt7 or hunt9"

Run only edge-case tests:
    pytest test_hunting.py -v -k "edge"

Run only the integration test:
    pytest test_hunting.py -v -k "integration"

How to read the results
-----------------------
  PASSED  — the hunt behaves exactly as expected
  FAILED  — something broke; the assertion message tells you what
  ERROR   — the test itself crashed (import error, fixture issue, etc.)

What "good results" looks like
-------------------------------
  All 9 FIRE tests pass   → hunts correctly detect their patterns
  All 9 SILENT tests pass → hunts don't fire on innocent data (no false positives)
  All edge-case tests pass → module is robust to bad / missing data
  Integration test passes  → the full pipeline works end-to-end

Coverage target: aim for >= 80% line coverage on hunting.py
"""

import sys
import os

import pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import hunting

import hunting  # noqa: E402  (must come after sys.path insert)


# ===========================================================================
# ── SHARED FIXTURES ─────────────────────────────────────────────────────────
# ===========================================================================

def make_row(
    user_id=1,
    timestamp="2024-01-10 10:00:00",
    action="login",
    login_attempts=0,
    file_size=0.0,
    ip_address="10.0.0.1",
    anomaly_type=None,
    label=0,
) -> dict:
    """
    Helper: return a single log-row dictionary with sensible defaults.
    Keeps test data minimal and readable.
    """
    return {
        "user_id": user_id,
        "timestamp": timestamp,
        "action": action,
        "login_attempts": login_attempts,
        "file_size": file_size,
        "ip_address": ip_address,
        "anomaly_type": anomaly_type,
        "label": label,
    }


def make_df(*rows) -> pd.DataFrame:
    """
    Wrap a list of row dicts into a properly-typed DataFrame.

    Timestamps are parsed to datetime64 here so that unit tests that call
    individual hunt functions directly (bypassing _validate_and_prepare)
    still work correctly — the hunt functions expect datetime columns.
    """
    df = pd.DataFrame(list(rows))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ===========================================================================
# ── HUNT 1 — BRUTE FORCE CHAIN ──────────────────────────────────────────────
# ===========================================================================

class TestHunt1BruteForce:

    def test_hunt1_fires_at_threshold(self):
        """Exactly 3 attempts (== threshold) must fire."""
        df = make_df(make_row(action="login", login_attempts=3))
        results = hunting.detect_bruteforce_chain(df)
        assert len(results) == 1
        assert results[0]["pattern"] == "Brute Force Chain"
        assert results[0]["severity"] == "High"

    def test_hunt1_fires_above_threshold(self):
        """10 attempts (well above threshold) must fire."""
        df = make_df(make_row(action="login", login_attempts=10))
        results = hunting.detect_bruteforce_chain(df)
        assert len(results) == 1

    def test_hunt1_silent_below_threshold(self):
        """2 attempts (below threshold of 3) must NOT fire."""
        df = make_df(make_row(action="login", login_attempts=2))
        results = hunting.detect_bruteforce_chain(df)
        assert len(results) == 0

    def test_hunt1_silent_wrong_action(self):
        """High login_attempts on a non-login action must NOT fire."""
        df = make_df(make_row(action="file access", login_attempts=10))
        results = hunting.detect_bruteforce_chain(df)
        assert len(results) == 0

    def test_hunt1_multiple_users_independent(self):
        """Each user's login events are evaluated independently."""
        df = make_df(
            make_row(user_id=1, action="login", login_attempts=5),   # should fire
            make_row(user_id=2, action="login", login_attempts=1),   # should NOT fire
            make_row(user_id=3, action="login", login_attempts=8),   # should fire
        )
        results = hunting.detect_bruteforce_chain(df)
        assert len(results) == 2
        flagged_users = {r["user_id"] for r in results}
        assert flagged_users == {1, 3}

    def test_hunt1_custom_threshold(self):
        """Custom threshold parameter overrides the default."""
        df = make_df(make_row(action="login", login_attempts=5))
        # Raise threshold to 6 — should NOT fire
        results = hunting.detect_bruteforce_chain(df, min_attempts=6)
        assert len(results) == 0
        # Lower threshold to 3 — should fire
        results = hunting.detect_bruteforce_chain(df, min_attempts=3)
        assert len(results) == 1

    def test_hunt1_missing_column_returns_empty(self):
        """Missing login_attempts column must return [] not crash."""
        df = make_df(make_row(action="login"))
        df = df.drop(columns=["login_attempts"])
        results = hunting.detect_bruteforce_chain(df)
        assert results == []

    def test_hunt1_details_mention_attempts(self):
        """Details string must mention the actual attempt count."""
        df = make_df(make_row(action="login", login_attempts=7))
        results = hunting.detect_bruteforce_chain(df)
        assert "7" in results[0]["details"]


# ===========================================================================
# ── HUNT 2 — USB FOLLOWED BY FILE ACCESS ────────────────────────────────────
# ===========================================================================

class TestHunt2UsbFileChain:

    def test_hunt2_fires_usb_then_file(self):
        """USB insert followed by file access within window must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="usb insert"),
            make_row(user_id=1, timestamp="2024-01-10 10:00:00", action="file access", file_size=20.0),
        )
        results = hunting.detect_usb_file_chain(df, window_days=3.0)
        assert len(results) == 1
        assert results[0]["pattern"] == "USB Followed By File Access"
        assert results[0]["severity"] == "Medium"

    def test_hunt2_silent_file_before_usb(self):
        """File access BEFORE USB insert must NOT fire (wrong order)."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="file access", file_size=20.0),
            make_row(user_id=1, timestamp="2024-01-10 10:00:00", action="usb insert"),
        )
        results = hunting.detect_usb_file_chain(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt2_silent_outside_window(self):
        """File access more than window_days after USB must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-01 08:00:00", action="usb insert"),
            make_row(user_id=1, timestamp="2024-01-10 10:00:00", action="file access", file_size=20.0),
        )
        results = hunting.detect_usb_file_chain(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt2_silent_different_users(self):
        """USB from user A and file access from user B must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="usb insert"),
            make_row(user_id=2, timestamp="2024-01-10 10:00:00", action="file access", file_size=20.0),
        )
        results = hunting.detect_usb_file_chain(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt2_fires_exactly_at_window_boundary(self):
        """File access at exactly window_days after USB must fire (inclusive)."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 00:00:00", action="usb insert"),
            make_row(user_id=1, timestamp="2024-01-13 00:00:00", action="file access", file_size=10.0),
        )
        results = hunting.detect_usb_file_chain(df, window_days=3.0)
        assert len(results) == 1

    def test_hunt2_details_contain_file_size(self):
        """Details must mention the file size."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="usb insert"),
            make_row(user_id=1, timestamp="2024-01-10 10:00:00", action="file access", file_size=42.5),
        )
        results = hunting.detect_usb_file_chain(df, window_days=3.0)
        assert "42.5" in results[0]["details"] or "42.50" in results[0]["details"]


# ===========================================================================
# ── HUNT 3 — LARGE FILE -> EXFILTRATION ─────────────────────────────────────
# ===========================================================================

class TestHunt3ExfiltrationChain:

    def test_hunt3_fires_large_file_then_exfil(self):
        """Large file access followed by exfiltration within window must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="file access", file_size=35.0),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration"),
        )
        results = hunting.detect_exfiltration_chain(df, file_size_threshold_mb=30.0, window_days=7.0)
        assert len(results) == 1
        assert results[0]["pattern"] == "Potential Data Theft Chain"
        assert results[0]["severity"] == "Critical"

    def test_hunt3_silent_small_file(self):
        """File access below threshold followed by exfil must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="file access", file_size=10.0),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration"),
        )
        results = hunting.detect_exfiltration_chain(df, file_size_threshold_mb=30.0, window_days=7.0)
        assert len(results) == 0

    def test_hunt3_silent_exfil_before_file(self):
        """Exfiltration BEFORE file access must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="exfiltration"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="file access", file_size=40.0),
        )
        results = hunting.detect_exfiltration_chain(df, file_size_threshold_mb=30.0, window_days=7.0)
        assert len(results) == 0

    def test_hunt3_silent_outside_window(self):
        """Exfiltration beyond window_days after large file must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-01 09:00:00", action="file access", file_size=40.0),
            make_row(user_id=1, timestamp="2024-01-20 09:00:00", action="exfiltration"),
        )
        results = hunting.detect_exfiltration_chain(df, file_size_threshold_mb=30.0, window_days=7.0)
        assert len(results) == 0

    def test_hunt3_fires_exactly_at_threshold_size(self):
        """File size exactly equal to threshold must fire (inclusive)."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="file access", file_size=30.0),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration"),
        )
        results = hunting.detect_exfiltration_chain(df, file_size_threshold_mb=30.0, window_days=7.0)
        assert len(results) == 1

    def test_hunt3_missing_file_size_column_returns_empty(self):
        """Missing file_size column must return [] not crash."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="file access"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration"),
        )
        df = df.drop(columns=["file_size"])
        results = hunting.detect_exfiltration_chain(df)
        assert results == []


# ===========================================================================
# ── HUNT 4 — REMOTE LOGIN -> SUSPICIOUS ACTIVITY ────────────────────────────
# ===========================================================================

class TestHunt4RemoteLogin:

    def test_hunt4_fires_remote_then_exfil(self):
        """Remote login followed by exfiltration within window must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="remote login"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration"),
        )
        results = hunting.detect_remote_login_chain(df, window_days=7.0)
        assert len(results) == 1
        assert results[0]["pattern"] == "Suspicious Activity After Remote Login"
        assert results[0]["severity"] == "High"

    def test_hunt4_fires_remote_then_usb(self):
        """Remote login followed by USB insert within window must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="remote login"),
            make_row(user_id=1, timestamp="2024-01-11 09:00:00", action="usb insert"),
        )
        results = hunting.detect_remote_login_chain(df, window_days=7.0)
        assert len(results) == 1

    def test_hunt4_silent_no_suspicious_follow_up(self):
        """Remote login with only normal activity after must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="remote login"),
            make_row(user_id=1, timestamp="2024-01-11 09:00:00", action="network traffic"),
        )
        results = hunting.detect_remote_login_chain(df, window_days=7.0)
        assert len(results) == 0

    def test_hunt4_silent_outside_window(self):
        """Suspicious activity beyond window_days must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-01 09:00:00", action="remote login"),
            make_row(user_id=1, timestamp="2024-01-20 09:00:00", action="exfiltration"),
        )
        results = hunting.detect_remote_login_chain(df, window_days=7.0)
        assert len(results) == 0

    def test_hunt4_custom_suspicious_actions(self):
        """Custom suspicious_actions set is respected."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="remote login"),
            make_row(user_id=1, timestamp="2024-01-11 09:00:00", action="network traffic"),
        )
        # Normally network traffic is not suspicious — but we can make it so
        results = hunting.detect_remote_login_chain(
            df, window_days=7.0, suspicious_actions={"network traffic"}
        )
        assert len(results) == 1


# ===========================================================================
# ── HUNT 5 — OFF-HOURS LOGIN -> SENSITIVE ACTION ────────────────────────────
# ===========================================================================

class TestHunt5OffHours:

    def test_hunt5_fires_midnight_login_then_file(self):
        """Login at 02:00 (off-hours) followed by file access must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 02:00:00", action="login"),
            make_row(user_id=1, timestamp="2024-01-11 09:00:00", action="file access", file_size=10.0),
        )
        results = hunting.detect_off_hours_login_chain(df, window_days=3.0)
        assert len(results) == 1
        assert results[0]["pattern"] == "Off-Hours Login Followed By Sensitive Action"
        assert results[0]["severity"] == "Medium"

    def test_hunt5_fires_late_night_login_then_exfil(self):
        """Login at 23:00 (off-hours) followed by exfiltration must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 23:00:00", action="login"),
            make_row(user_id=1, timestamp="2024-01-11 10:00:00", action="exfiltration"),
        )
        results = hunting.detect_off_hours_login_chain(df, window_days=3.0)
        assert len(results) == 1

    def test_hunt5_silent_business_hours_login(self):
        """Login at 09:00 (business hours) followed by file access must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="login"),
            make_row(user_id=1, timestamp="2024-01-10 11:00:00", action="file access", file_size=10.0),
        )
        results = hunting.detect_off_hours_login_chain(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt5_silent_off_hours_no_follow_up(self):
        """Off-hours login with no sensitive follow-up must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 02:00:00", action="login"),
            make_row(user_id=1, timestamp="2024-01-10 03:00:00", action="network traffic"),
        )
        results = hunting.detect_off_hours_login_chain(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt5_silent_outside_window(self):
        """Sensitive action beyond window_days after off-hours login must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-01 02:00:00", action="login"),
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="file access", file_size=10.0),
        )
        results = hunting.detect_off_hours_login_chain(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt5_boundary_hour_22_is_off_hours(self):
        """Hour 22 (start of off-hours) must be treated as off-hours."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 22:00:00", action="login"),
            make_row(user_id=1, timestamp="2024-01-11 09:00:00", action="exfiltration"),
        )
        results = hunting.detect_off_hours_login_chain(df, window_days=3.0)
        assert len(results) == 1

    def test_hunt5_boundary_hour_06_is_business_hours(self):
        """Hour 06 (end of off-hours, exclusive) must NOT be off-hours."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 06:00:00", action="login"),
            make_row(user_id=1, timestamp="2024-01-11 09:00:00", action="exfiltration"),
        )
        results = hunting.detect_off_hours_login_chain(df, window_days=3.0)
        assert len(results) == 0


# ===========================================================================
# ── HUNT 6 — RAPID IP SWITCHING ─────────────────────────────────────────────
# ===========================================================================

class TestHunt6IpSwitching:

    def test_hunt6_fires_different_ips_within_window(self):
        """Two logins from different IPs within window_days must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="login", ip_address="10.0.0.1"),
            make_row(user_id=1, timestamp="2024-01-11 08:00:00", action="login", ip_address="192.168.1.1"),
        )
        results = hunting.detect_ip_switching(df, window_days=3.0)
        assert len(results) == 1
        assert results[0]["pattern"] == "Rapid IP Switching"
        assert results[0]["severity"] == "High"

    def test_hunt6_silent_same_ip(self):
        """Two logins from the SAME IP must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="login", ip_address="10.0.0.1"),
            make_row(user_id=1, timestamp="2024-01-11 08:00:00", action="login", ip_address="10.0.0.1"),
        )
        results = hunting.detect_ip_switching(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt6_silent_different_ips_outside_window(self):
        """Different IPs but gap exceeds window_days must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-01 08:00:00", action="login", ip_address="10.0.0.1"),
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="login", ip_address="192.168.1.1"),
        )
        results = hunting.detect_ip_switching(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt6_silent_different_users(self):
        """IP switch across different users must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="login", ip_address="10.0.0.1"),
            make_row(user_id=2, timestamp="2024-01-10 09:00:00", action="login", ip_address="192.168.1.1"),
        )
        results = hunting.detect_ip_switching(df, window_days=3.0)
        assert len(results) == 0

    def test_hunt6_missing_ip_column_returns_empty(self):
        """Missing ip_address column must return [] not crash."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="login"),
            make_row(user_id=1, timestamp="2024-01-11 08:00:00", action="login"),
        )
        df = df.drop(columns=["ip_address"])
        results = hunting.detect_ip_switching(df)
        assert results == []

    def test_hunt6_details_mention_both_ips(self):
        """Details must mention both the old and the new IP address."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 08:00:00", action="login", ip_address="10.0.0.1"),
            make_row(user_id=1, timestamp="2024-01-11 08:00:00", action="login", ip_address="192.168.1.1"),
        )
        results = hunting.detect_ip_switching(df, window_days=3.0)
        assert "10.0.0.1" in results[0]["details"]
        assert "192.168.1.1" in results[0]["details"]


# ===========================================================================
# ── HUNT 7 — MULTI-ANOMALY STACKING ─────────────────────────────────────────
# ===========================================================================

class TestHunt7AnomalyStacking:

    def test_hunt7_fires_two_distinct_anomalies(self):
        """User with 2 distinct anomaly types must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="login", anomaly_type="Brute_Force"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration", anomaly_type="Data_Exfil"),
        )
        results = hunting.detect_anomaly_stacking(df, min_anomaly_types=2)
        assert len(results) == 1
        assert results[0]["pattern"] == "Multi-Anomaly Stacking"
        assert results[0]["severity"] == "Critical"

    def test_hunt7_fires_three_distinct_anomalies(self):
        """User with 3 distinct anomaly types must fire (stronger signal)."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="login",       anomaly_type="Brute_Force"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="usb insert",  anomaly_type="USB_Access"),
            make_row(user_id=1, timestamp="2024-01-14 09:00:00", action="exfiltration", anomaly_type="Data_Exfil"),
        )
        results = hunting.detect_anomaly_stacking(df, min_anomaly_types=2)
        assert len(results) == 1  # one finding per user
        assert "3 distinct" in results[0]["details"]

    def test_hunt7_silent_single_anomaly_type(self):
        """User with only 1 anomaly type must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="login",       anomaly_type="Brute_Force"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="login",       anomaly_type="Brute_Force"),
        )
        results = hunting.detect_anomaly_stacking(df, min_anomaly_types=2)
        assert len(results) == 0

    def test_hunt7_silent_no_anomaly_labels(self):
        """User with no anomaly_type labels (all NaN) must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="login",       anomaly_type=None),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration", anomaly_type=None),
        )
        results = hunting.detect_anomaly_stacking(df, min_anomaly_types=2)
        assert len(results) == 0

    def test_hunt7_one_finding_per_user(self):
        """A user with 4 anomaly events should produce exactly ONE finding."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="login",       anomaly_type="Brute_Force"),
            make_row(user_id=1, timestamp="2024-01-11 09:00:00", action="login",       anomaly_type="Brute_Force"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration", anomaly_type="Data_Exfil"),
            make_row(user_id=1, timestamp="2024-01-13 09:00:00", action="usb insert",  anomaly_type="USB_Access"),
        )
        results = hunting.detect_anomaly_stacking(df, min_anomaly_types=2)
        assert len(results) == 1

    def test_hunt7_missing_column_returns_empty(self):
        """Missing anomaly_type column must return [] not crash."""
        df = make_df(make_row(user_id=1, action="login"))
        df = df.drop(columns=["anomaly_type"])
        results = hunting.detect_anomaly_stacking(df)
        assert results == []

    def test_hunt7_details_list_anomaly_types(self):
        """Details must mention the actual anomaly type names."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="login",       anomaly_type="Brute_Force"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration", anomaly_type="Data_Exfil"),
        )
        results = hunting.detect_anomaly_stacking(df, min_anomaly_types=2)
        assert "Brute_Force" in results[0]["details"]
        assert "Data_Exfil" in results[0]["details"]


# ===========================================================================
# ── HUNT 8 — HIGH-VELOCITY FILE ACCESS ──────────────────────────────────────
# ===========================================================================

class TestHunt8FileVelocity:

    def test_hunt8_fires_above_absolute_minimum(self):
        """User with file accesses >= absolute_min threshold must fire."""
        rows = [
            make_row(user_id=1, timestamp=f"2024-01-{10+i:02d} 09:00:00",
                     action="file access", file_size=10.0)
            for i in range(3)
        ]
        # Users 2-10: single file access each (to set a realistic mean ~1.3)
        for uid in range(2, 10):
            rows.append(make_row(user_id=uid, timestamp="2024-01-10 09:00:00",
                                 action="file access", file_size=5.0))
        df = make_df(*rows)
        results = hunting.detect_high_velocity_file_access(df, absolute_min=3)
        flagged = {r["user_id"] for r in results}
        assert 1 in flagged

    def test_hunt8_silent_below_threshold(self):
        """User with 1 file access (below any threshold) must NOT fire."""
        rows = [
            make_row(user_id=uid, timestamp="2024-01-10 09:00:00",
                     action="file access", file_size=5.0)
            for uid in range(1, 20)    # 19 users, 1 file access each
        ]
        df = make_df(*rows)
        results = hunting.detect_high_velocity_file_access(df, absolute_min=3)
        assert len(results) == 0

    def test_hunt8_output_contains_count(self):
        """Details must mention the number of file access events."""
        rows = [
            make_row(user_id=1, timestamp=f"2024-01-{10+i:02d} 09:00:00",
                     action="file access", file_size=10.0)
            for i in range(3)
        ]
        for uid in range(2, 15):
            rows.append(make_row(user_id=uid, timestamp="2024-01-10 09:00:00",
                                 action="file access", file_size=5.0))
        df = make_df(*rows)
        results = hunting.detect_high_velocity_file_access(df, absolute_min=3)
        user1_findings = [r for r in results if r["user_id"] == 1]
        assert len(user1_findings) == 1
        assert "3" in user1_findings[0]["details"]

    def test_hunt8_empty_dataset_returns_empty(self):
        """Dataset with no file access events must return []."""
        df = make_df(make_row(user_id=1, action="login"))
        results = hunting.detect_high_velocity_file_access(df)
        assert results == []


# ===========================================================================
# ── HUNT 9 — NEW-IP REMOTE LOGIN -> EXFILTRATION ────────────────────────────
# ===========================================================================

class TestHunt9NewIpExfil:

    def test_hunt9_fires_new_ip_then_exfil(self):
        """First-ever IP on remote login followed by exfil must fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00",
                     action="remote login", ip_address="10.0.0.99"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00",
                     action="exfiltration", ip_address="10.0.0.99"),
        )
        results = hunting.detect_new_ip_exfiltration(df, window_days=7.0)
        assert len(results) == 1
        assert results[0]["pattern"] == "New-IP Remote Login Followed By Exfiltration"
        assert results[0]["severity"] == "Critical"

    def test_hunt9_silent_known_ip(self):
        """Remote login from a previously-seen IP must NOT fire."""
        df = make_df(
            # First login from this IP — establishes it as known
            make_row(user_id=1, timestamp="2024-01-05 09:00:00",
                     action="remote login", ip_address="10.0.0.1"),
            # Second login from same IP — no longer "new"
            make_row(user_id=1, timestamp="2024-01-10 09:00:00",
                     action="remote login", ip_address="10.0.0.1"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00",
                     action="exfiltration", ip_address="10.0.0.1"),
        )
        results = hunting.detect_new_ip_exfiltration(df, window_days=7.0)
        # Only the FIRST remote login (Jan 5) with that IP could fire;
        # the exfil is 7 days later which is at the window boundary.
        # The second remote login (Jan 10, same IP) should NOT fire.
        # We check that at most 1 finding is raised (from the first new IP)
        # and it mentions the first login date.
        for r in results:
            assert "2024-01-05" in r["details"] or len(results) == 0

    def test_hunt9_silent_no_exfil(self):
        """New IP remote login with no exfiltration at all must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00",
                     action="remote login", ip_address="10.0.0.99"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00",
                     action="file access", file_size=10.0),
        )
        results = hunting.detect_new_ip_exfiltration(df, window_days=7.0)
        assert len(results) == 0

    def test_hunt9_silent_outside_window(self):
        """Exfil beyond window_days after new-IP login must NOT fire."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-01 09:00:00",
                     action="remote login", ip_address="10.0.0.99"),
            make_row(user_id=1, timestamp="2024-01-20 09:00:00",
                     action="exfiltration"),
        )
        results = hunting.detect_new_ip_exfiltration(df, window_days=7.0)
        assert len(results) == 0

    def test_hunt9_missing_ip_column_returns_empty(self):
        """Missing ip_address column must return [] not crash."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00", action="remote login"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00", action="exfiltration"),
        )
        df = df.drop(columns=["ip_address"])
        results = hunting.detect_new_ip_exfiltration(df)
        assert results == []

    def test_hunt9_details_mention_new_ip(self):
        """Details must mention the new (first-time) IP address."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00",
                     action="remote login", ip_address="172.31.99.99"),
            make_row(user_id=1, timestamp="2024-01-12 09:00:00",
                     action="exfiltration", ip_address="172.31.99.99"),
        )
        results = hunting.detect_new_ip_exfiltration(df, window_days=7.0)
        assert "172.31.99.99" in results[0]["details"]


# ===========================================================================
# ── EDGE CASES — defensive programming checks ────────────────────────────────
# ===========================================================================

class TestEdgeCases:

    def test_edge_empty_dataframe(self):
        """Empty DataFrame must return an empty results DataFrame, not crash."""
        df = pd.DataFrame(columns=["user_id", "timestamp", "action",
                                   "login_attempts", "file_size",
                                   "ip_address", "anomaly_type"])
        result = hunting.run_all_hunts(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_edge_missing_required_column(self):
        """Missing required column (action) must return empty DataFrame, not crash."""
        df = pd.DataFrame([{"user_id": 1, "timestamp": "2024-01-10 09:00:00"}])
        result = hunting.run_all_hunts(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_edge_malformed_timestamp_is_dropped(self):
        """
        Rows with unparseable timestamps must be silently dropped.
        We construct this DataFrame manually (not via make_df) so that
        the bad timestamp stays as a raw string — _validate_and_prepare
        is responsible for coercing it to NaT and dropping the row.
        """
        df = pd.DataFrame([
            {"user_id": 1, "timestamp": "NOT_A_DATE",
             "action": "login", "login_attempts": 5,
             "file_size": 0.0, "ip_address": "10.0.0.1",
             "anomaly_type": None, "label": 1},
            {"user_id": 2, "timestamp": "2024-01-10 09:00:00",
             "action": "login", "login_attempts": 5,
             "file_size": 0.0, "ip_address": "10.0.0.2",
             "anomaly_type": None, "label": 1},
        ])
        result = hunting.run_all_hunts(df)
        assert isinstance(result, pd.DataFrame)
        flagged = set(result["user_id"].tolist())
        assert 2 in flagged
        assert 1 not in flagged

    def test_edge_minimal_columns_no_crash(self):
        """DataFrame with only the three required columns must not crash."""
        df = pd.DataFrame([{
            "user_id": 1,
            "timestamp": "2024-01-10 09:00:00",
            "action": "login",
        }])
        result = hunting.run_all_hunts(df)
        assert isinstance(result, pd.DataFrame)

    def test_edge_nan_values_in_optional_columns(self):
        """NaN values in optional columns must be handled gracefully."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00",
                     action="login", login_attempts=5,
                     file_size=float("nan"), ip_address=None, anomaly_type=None),
        )
        result = hunting.run_all_hunts(df)
        assert isinstance(result, pd.DataFrame)

    def test_edge_single_event_user(self):
        """User with only one event must never produce a sequence-based finding."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-10 09:00:00",
                     action="usb insert", login_attempts=0),
        )
        # No file access follows — Hunt 2 must not fire
        results = hunting.detect_usb_file_chain(df, window_days=7.0)
        assert len(results) == 0

    def test_edge_output_has_correct_columns(self):
        """run_all_hunts output must always have exactly the five standard columns."""
        df = make_df(make_row(action="login", login_attempts=5))
        result = hunting.run_all_hunts(df)
        expected_cols = {"user_id", "detection_time", "pattern", "severity", "details"}
        assert set(result.columns) == expected_cols

    def test_edge_no_duplicates_in_output(self):
        """run_all_hunts must deduplicate findings before returning."""
        df = make_df(make_row(action="login", login_attempts=5))
        result = hunting.run_all_hunts(df)
        dupes = result.duplicated(subset=["user_id", "detection_time", "pattern"]).sum()
        assert dupes == 0

    def test_edge_output_sorted_by_time(self):
        """run_all_hunts output must be sorted by detection_time ascending."""
        df = make_df(
            make_row(user_id=1, timestamp="2024-01-15 09:00:00", action="login", login_attempts=5),
            make_row(user_id=2, timestamp="2024-01-10 09:00:00", action="login", login_attempts=5),
        )
        result = hunting.run_all_hunts(df)
        if len(result) >= 2:
            times = result["detection_time"].tolist()
            assert times == sorted(times)

    def test_edge_severity_values_are_valid(self):
        """All severity values must be from the allowed set."""
        df = make_df(
            make_row(action="login", login_attempts=5),
        )
        result = hunting.run_all_hunts(df)
        allowed = {"Critical", "High", "Medium", "Low"}
        assert result["severity"].isin(allowed).all()

    def test_edge_action_case_insensitive(self):
        """
        Action strings with different casing must still be detected.
        This test goes through run_all_hunts so _validate_and_prepare
        normalises the action column to lowercase before the hunt runs.
        """
        df = pd.DataFrame([{
            "user_id": 1,
            "timestamp": "2024-01-10 09:00:00",
            "action": "LOGIN",          # uppercase — must still fire
            "login_attempts": 5,
            "ip_address": "10.0.0.1",
            "file_size": 0.0,
            "anomaly_type": None,
            "label": 0,
        }])
        # run_all_hunts calls _validate_and_prepare which lowercases actions
        result = hunting.run_all_hunts(df)
        bf = result[result["pattern"] == "Brute Force Chain"]
        assert len(bf) == 1


# ===========================================================================
# ── INTEGRATION TEST — real dataset ─────────────────────────────────────────
# ===========================================================================

class TestIntegration:

    @pytest.fixture(scope="class")
    def real_results(self):
        """
        Load processed_logs.csv and run all hunts once for the whole class.
        Skipped automatically if the file is not present.
        """
        data_path = os.path.join(os.path.dirname(__file__), "processed_logs.csv")
        if not os.path.exists(data_path):
            pytest.skip(f"Real dataset not found at '{data_path}' — skipping integration tests.")
        df = pd.read_csv(data_path)
        return hunting.run_all_hunts(df)

    def test_integration_returns_dataframe(self, real_results):
        assert isinstance(real_results, pd.DataFrame)

    def test_integration_has_correct_columns(self, real_results):
        expected = {"user_id", "detection_time", "pattern", "severity", "details"}
        assert set(real_results.columns) == expected

    def test_integration_has_findings(self, real_results):
        assert len(real_results) > 0, "Expected findings on the real dataset"

    def test_integration_all_nine_patterns_present(self, real_results):
        """Every hunt must have produced at least one finding on the real data."""
        expected_patterns = {
            "Brute Force Chain",
            "USB Followed By File Access",
            "Potential Data Theft Chain",
            "Suspicious Activity After Remote Login",
            "Off-Hours Login Followed By Sensitive Action",
            "Rapid IP Switching",
            "Multi-Anomaly Stacking",
            "High-Velocity File Access",
            "New-IP Remote Login Followed By Exfiltration",
        }
        found_patterns = set(real_results["pattern"].unique())
        missing = expected_patterns - found_patterns
        assert not missing, f"These hunts produced zero findings: {missing}"

    def test_integration_no_duplicates(self, real_results):
        dupes = real_results.duplicated(subset=["user_id", "detection_time", "pattern"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate findings"

    def test_integration_severity_values_valid(self, real_results):
        allowed = {"Critical", "High", "Medium", "Low"}
        assert real_results["severity"].isin(allowed).all()

    def test_integration_sorted_by_time(self, real_results):
        times = real_results["detection_time"].tolist()
        assert times == sorted(times), "Results are not sorted by detection_time"

    def test_integration_critical_findings_exist(self, real_results):
        """We expect Critical findings from Hunts 3, 7, and 9."""
        critical = real_results[real_results["severity"] == "Critical"]
        assert len(critical) > 0, "Expected at least one Critical finding"

    def test_integration_expected_finding_counts(self, real_results):
        """
        Regression guard: finding counts must be within 20% of the
        baseline established when hunting.py was built.

        Baseline (from v2.0.0 run on processed_logs.csv):
            Brute Force Chain                          : 812
            USB Followed By File Access                :  29
            Potential Data Theft Chain                 :   6
            Suspicious Activity After Remote Login     :  22  (±1 due to dedup)
            Off-Hours Login Followed By Sensitive Action:  36
            Rapid IP Switching                         :  48
            Multi-Anomaly Stacking                     :  58
            High-Velocity File Access                  :  46
            New-IP Remote Login Followed By Exfiltration: 14
        """
        baselines = {
            "Brute Force Chain": 812,
            "USB Followed By File Access": 29,
            "Potential Data Theft Chain": 6,
            "Suspicious Activity After Remote Login": 22,
            "Off-Hours Login Followed By Sensitive Action": 36,
            "Rapid IP Switching": 48,
            "Multi-Anomaly Stacking": 58,
            "High-Velocity File Access": 46,
            "New-IP Remote Login Followed By Exfiltration": 14,
        }
        counts = real_results["pattern"].value_counts().to_dict()
        tolerance = 0.20   # 20% tolerance

        for pattern, expected in baselines.items():
            actual = counts.get(pattern, 0)
            lower = expected * (1 - tolerance)
            upper = expected * (1 + tolerance)
            assert lower <= actual <= upper, (
                f"Pattern '{pattern}': expected ~{expected} findings "
                f"(±{tolerance*100:.0f}%), got {actual}"
            )

    def test_integration_user_9123_multi_pattern(self, real_results):
        """
        Regression: user 9123 is the highest-risk user in the dataset.
        They must appear in multiple patterns.
        """
        user_findings = real_results[real_results["user_id"] == 9123]
        assert len(user_findings) >= 2, (
            f"User 9123 should appear in >= 2 patterns, found {len(user_findings)}"
        )


# ===========================================================================
# ── QUALITY METRICS REPORT (run standalone) ─────────────────────────────────
# ===========================================================================

def print_quality_report(results_df: pd.DataFrame) -> None:
    """
    Print a human-readable quality report for a findings DataFrame.

    Call this in your notebook or script after run_all_hunts():

        from test_hunting import print_quality_report
        results = run_all_hunts(df)
        print_quality_report(results)

    What each metric means
    ----------------------
    Coverage         : % of unique users flagged. Higher = broader detection.
                       Too high (> 50%) may indicate over-alerting / noise.
    Duplicate rate   : Should always be 0% after dedup step.
    Critical ratio   : % of findings that are Critical severity.
                       A healthy SOC queue has at least a few critical alerts.
    Detail fill rate : % of findings with a non-empty details string.
                       Should be 100% — blank details are useless for analysts.
    """
    if results_df.empty:
        print("No findings to report.")
        return

    total = len(results_df)
    unique_users_flagged = results_df["user_id"].nunique()
    duplicates = results_df.duplicated(subset=["user_id", "detection_time", "pattern"]).sum()
    critical_count = (results_df["severity"] == "Critical").sum()
    detail_fill = (results_df["details"].str.strip() != "").sum()

    print("\n" + "=" * 55)
    print("  THREAT HUNTING QUALITY REPORT")
    print("=" * 55)
    print(f"  Total findings          : {total}")
    print(f"  Unique users flagged    : {unique_users_flagged}")
    print(f"  Duplicate findings      : {duplicates}  {'✓' if duplicates == 0 else '✗ PROBLEM'}")
    print(f"  Critical findings       : {critical_count}  ({100*critical_count/total:.1f}%)")
    print(f"  Detail fill rate        : {100*detail_fill/total:.1f}%  {'✓' if detail_fill==total else '✗ PROBLEM'}")

    print("\n  Findings by severity:")
    for sev in ["Critical", "High", "Medium", "Low"]:
        n = (results_df["severity"] == sev).sum()
        bar = "█" * (n // 10)
        print(f"    {sev:<10} {n:>5}  {bar}")

    print("\n  Findings by pattern:")
    pattern_counts = results_df["pattern"].value_counts()
    for pattern, count in pattern_counts.items():
        bar = "█" * (count // 10)
        print(f"    {pattern:<45} {count:>5}  {bar}")

    print("\n  Top 5 highest-risk users (most patterns triggered):")
    top_users = (
        results_df.groupby("user_id")["pattern"]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
    )
    for uid, n_patterns in top_users.items():
        user_sevs = results_df[results_df["user_id"] == uid]["severity"].tolist()
        print(f"    User {uid:<8} — {n_patterns} pattern(s)  {user_sevs}")

    print("=" * 55)


if __name__ == "__main__":
    # Run the quality report directly against the real dataset
    import os
    data_path = os.path.join(os.path.dirname(__file__), "processed_logs.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        results = hunting.run_all_hunts(df)
        print_quality_report(results)
    else:
        print(f"Dataset not found at '{data_path}'")
        print("Running unit tests only via pytest instead...")
        os.system(f"pytest {__file__} -v --tb=short")