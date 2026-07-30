# PillVault AI

Smart medication management for patients, caregivers, and pharmacies.

A full-stack medication cabinet agent with AI-powered label scanning, adherence tracking, automated refill ordering, and recurring payment mandates.

## Features

- **Medication Dashboard** — Web-based UI to view, add, and manage medications across multiple family member profiles
- **AI Label Scanning** — Scan prescription labels via webcam or photo upload using OpenCV + EasyOCR (local, no API key), or Claude Vision (optional)
- **Dose Logging & Adherence** — Log doses taken, track daily/weekly adherence rates, export CSV reports
- **Drug Interaction Warnings** — Automatically detects conflicting medications in a profile
- **Predictive Refills** — Auto-drafts refill requests when stock runs low, routes to caregiver for approval
- **Prava Payment Mandates** — One-time payments or recurring mandates for automatic refill ordering (no passkey needed after setup)
- **Multi-Profile** — Separate medication cabinets for each family member
- **CLI Agent** — Full-featured terminal interface for all workflows
- **Public Tunnel** — One-click ngrok sharing for remote caregiver access

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Frontend | Tailwind CSS, Chart.js, vanilla JS |
| OCR | OpenCV, EasyOCR (optional: Claude Vision) |
| Payments | [Prava API](https://docs.prava.space) (sandbox & production) |
| Tunneling | pyngrok |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the web server
python app.py
# or: uvicorn app:app --reload --port 8000

# Open http://127.0.0.1:8000
```

### CLI Agent

```bash
python pillvault_agent.py
```

### Public tunnel (optional)

```bash
pip install pyngrok
python start_public.py
```

## Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `PRAVA_SECRET_KEY` | For payments | [Prava](https://prava.space) sandbox/production key |
| `PRAVA_ENV` | No | `sandbox` (default) or `production` |
| `PRAVA_USER_ID` | No | User identifier for Prava |
| `PRAVA_USER_EMAIL` | No | User email for Prava |
| `ANTHROPIC_API_KEY` | For Claude Vision | Falls back to local OCR if not set |

## Project Structure

```
PILLVAULT/
├── app.py                 # FastAPI web application
├── pillvault_agent.py     # Autonomous CLI agent
├── opencv_scanner.py      # OpenCV + EasyOCR label scanner
├── prava.py               # Prava payments integration
├── start_public.py        # ngrok tunnel launcher
├── templates/index.html   # Single-page web UI
├── pillvault_data.json    # Local data store
├── requirements.txt
└── .env
```

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/medications` | List medications (filter by profile, search, low stock) |
| POST | `/api/medications` | Add medication (with optional image scan) |
| PUT | `/api/medications/{id}` | Update medication details |
| POST | `/api/medications/{id}/dose` | Log a dose taken |
| GET | `/api/medications/{id}/history` | 30-day dose history chart data |
| GET | `/api/dashboard` | Caregiver dashboard summary |
| GET | `/api/notifications` | List refill notifications |
| POST | `/api/notifications/{id}/approve` | Approve & process refill |
| POST | `/api/notifications/{id}/decline` | Decline refill |
| POST | `/api/scan` | Scan medication label from image |
| GET | `/api/reports/adherence` | Adherence report (JSON) |
| GET | `/api/reports/adherence/csv` | Adherence report (CSV download) |
| POST | `/api/mandates/setup/{id}` | Set up Prava payment mandate |
| GET | `/api/mandates` | List Prava mandates |
| GET | `/api/profiles` | List profiles |
| POST | `/api/profiles` | Create profile |

## License

MIT
