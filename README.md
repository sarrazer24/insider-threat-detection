# Insider Threat Detection & User Risk Scoring Platform

## 📊 Project Overview
This platform analyzes user activity logs to detect potentially risky or malicious insider behavior. It combines rule-based detection, risk scoring, and behavior pattern sequences to flag suspicious users and generate security alerts mapped to the MITRE ATT&CK framework.

## 📂 Project Structure
insider-threat-detection/
├── data/                  # Cleaned & processed activity logs
├── notebooks/             # Step-by-step development sandboxes
│   ├── exploration.ipynb   # (Data Eng) EDA & Schema definition
│   ├── risk_scoring.ipynb  # (Risk Eng) User aggregation experiments
│   └── threat_hunting.ipynb# (Hunting Eng) Sequential pattern design
├── src/                   # Production-ready backend modules
│   ├── risk_engine.py     # Quantifies risk score algorithms
│   ├── hunting.py         # Detects multi-event attack chains
│   ├── alerts.py          # Generates SOC-style JSON metrics
│   └── mitre_mapping.py   # Maps behaviors to ATT&CK matrix
├── dashboard/             # Front-end user interface
│   └── app.py             # Streamlit visual dashboard
└── reports/               # Final group writeups and presentations


## 🔄 Integration Workflow (Contract)
1. **Data Engineer** maps raw logs to standard lowercase schemas in `data/processed_logs.csv`.
2. **Risk Scoring Engineer** ingests logs and outputs user-level danger ratings (0-100).
3. **Threat Hunting Engineer** groups sequences to extract multi-stage malicious behavior chains.
4. **MITRE & Alerts Module** enriches the findings with security contexts.
5. **Dashboard Lead** merges all components into the final Streamlit app.