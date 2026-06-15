# Insider Threat Detection & User Risk Scoring Platform

## Overview

This project analyzes corporate activity logs to identify potentially risky insider behavior using rule-based cybersecurity analytics. Instead of relying on black-box machine learning models, the platform uses deterministic risk scoring, threat-hunting logic, and MITRE ATT&CK mappings to generate explainable security alerts.

The system includes an interactive Streamlit dashboard that allows analysts to:

* Search and investigate users
* View risk scores
* Explore attack timelines
* Review security alerts
* Map suspicious activity to MITRE ATT&CK techniques

---

## System Architecture

```text
[Raw Logs]
      │
      ▼
┌─────────────────────┐
│ Preprocessing Engine│
└──────────┬──────────┘
           ▼
[data/processed_logs.csv]
           │
 ┌─────────┴─────────┐
 ▼                   ▼
Risk Scoring     Threat Hunting
Engine           Module
 │                   │
 └─────────┬─────────┘
           ▼
┌─────────────────────┐
│ MITRE & Alert Engine│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Streamlit Dashboard │
└─────────────────────┘
```

---

## Project Structure

```text
insider-threat-detection/
├── data/
│   └── processed_logs.csv
│
├── notebooks/
│   ├── exploration.ipynb
│   ├── risk_scoring.ipynb
│   └── threat_hunting.ipynb
│
├── src/
│   ├── risk_engine.py
│   ├── hunting.py
│   ├── alerts.py
│   └── mitre_mapping.py
│
├── dashboard/
│   └── app.py
│
├── reports/
│
└── README.md
```

---

## Data Schema

All modules operate on a standardized schema generated during preprocessing.

| Field          | Type     | Description                                         |
| -------------- | -------- | --------------------------------------------------- |
| user_id        | String   | Unique employee identifier                          |
| timestamp      | Datetime | Event timestamp (`YYYY-MM-DD HH:MM:SS`)             |
| ip_address     | String   | IPv4 address                                        |
| action         | String   | User activity category                              |
| login_attempts | Integer  | Number of authentication attempts                   |
| file_size      | Float    | File size in KB                                     |
| anomaly_type   | String   | Detected anomaly classification                     |
| label          | Integer  | Ground-truth label (`0 = Normal`, `1 = Suspicious`) |

### Supported Actions

```text
login
remote login
file access
usb insert
exfiltration
network traffic
```

---

## Risk Scoring Engine

The risk engine aggregates user activity and computes a normalized risk score between **0 and 100**.

### Risk Weights

| Event                  | Points |
| ---------------------- | ------ |
| Data Exfiltration      | +50    |
| Brute Force Detection  | +40    |
| USB Insertion          | +20    |
| Repeated Failed Logins | +15    |
| Remote Login           | +10    |

### Risk Categories

| Score Range | Category |
| ----------- | -------- |
| 0–39        | Low      |
| 40–69       | Medium   |
| 70–100      | High     |

---

## Threat Hunting Logic

The threat-hunting module identifies suspicious sequences of events across a user's activity timeline.

### Brute Force Chain

```text
Failed Login
      ↓
Failed Login
      ↓
Failed Login
      ↓
Successful Login
```

### Exfiltration Chain

```text
USB Insert
      ↓
Large File Access
      ↓
Data Exfiltration
```

---

## MITRE ATT&CK Mapping

| Activity          | Technique |
| ----------------- | --------- |
| Brute Force       | T1110     |
| Remote Login      | T1021     |
| USB Insertion     | T1091     |
| Data Exfiltration | T1041     |

### References

* T1110 – Brute Force
* T1021 – Remote Services
* T1091 – Replication Through Removable Media
* T1041 – Exfiltration Over C2 Channel

---

## Dashboard Features

The Streamlit dashboard provides:

* User search and filtering
* Risk score visualization
* Alert investigation
* Threat timeline analysis
* MITRE ATT&CK enrichment
* Security monitoring interface

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sarrazer24/insider-threat-detection.git
cd insider-threat-detection
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn streamlit
```

---

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Tech Stack

* Python
* Pandas
* NumPy
* Streamlit
* Matplotlib
* Seaborn

---

## Future Improvements

* Real-time log ingestion
* SIEM integration
* User behavior analytics (UBA)
* Advanced threat correlation
* Automated incident reporting
* Machine learning anomaly detection

```
```
