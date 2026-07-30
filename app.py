from __future__ import annotations

import base64
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from opencv_scanner import scan_medication_label

_OPENCV_AVAILABLE = True
try:
    from opencv_scanner import capture_from_camera
except ImportError:
    capture_from_camera = None

try:
    from prava import PravaClient, PravaConfig
    _PRAVA_AVAILABLE = True
except ImportError:
    _PRAVA_AVAILABLE = False

DATA_FILE = Path(__file__).parent / "pillvault_data.json"
DUPLICATE_ORDER_LOCK_DAYS = 14
LOW_STOCK_THRESHOLD_DAYS = 5

INTERACTING_GROUPS = {
    "anticoagulant": {"nsaid"},
    "nsaid": {"anticoagulant"},
}


def load_store():
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())
    profiles = {
        "p1": {"id": "p1", "name": "John Doe", "avatar": "JD", "color": "indigo", "created_at": datetime.now().isoformat()},
        "p2": {"id": "p2", "name": "Sarah Doe", "avatar": "SD", "color": "emerald", "created_at": datetime.now().isoformat()},
    }
    seed = {"profiles": profiles, "active_profile": "p1", "meds": {}, "notifications": {}, "dose_history": [], "log": []}
    defaults = [
        {"id": "m1", "name": "Lisinopril", "dosage_mg": "10", "rx_number": "RX-88213",
         "schedule": "1 tablet at 8:00 AM", "doses_per_day": 1, "qty_remaining": 6, "initial_qty": 30,
         "expiration_date": "2027-06-15", "doctor_name": "Dr. Alvarez", "interaction_group": "ace_inhibitor",
         "refills_left": 2, "pharmacy_name": "MedPlus Pharmacy", "cost_estimate": 12.0,
         "last_order_date": None, "mandate_id": None, "mandate_amount": None, "profile_id": "p1"},
        {"id": "m2", "name": "Metformin", "dosage_mg": "500", "rx_number": "RX-77410",
         "schedule": "1 tablet at 8:00 AM and 6:00 PM", "doses_per_day": 2, "qty_remaining": 42, "initial_qty": 90,
         "expiration_date": "2027-08-20", "doctor_name": "Dr. Alvarez", "interaction_group": "biguanide",
         "refills_left": 1, "pharmacy_name": "MedPlus Pharmacy", "cost_estimate": 9.0,
         "last_order_date": None, "mandate_id": None, "mandate_amount": None, "profile_id": "p1"},
        {"id": "m3", "name": "Atorvastatin", "dosage_mg": "20", "rx_number": "RX-90055",
         "schedule": "1 tablet at 9:00 PM", "doses_per_day": 1, "qty_remaining": 3, "initial_qty": 30,
         "expiration_date": "2026-12-01", "doctor_name": "Dr. Chen", "interaction_group": "statin",
         "refills_left": 0, "pharmacy_name": "Riverside Pharmacy", "cost_estimate": 15.0,
         "last_order_date": None, "mandate_id": None, "mandate_amount": None, "profile_id": "p1"},
        {"id": "m4", "name": "Ibuprofen", "dosage_mg": "200", "rx_number": "RX-45123",
         "schedule": "1 tablet as needed", "doses_per_day": 1, "qty_remaining": 60, "initial_qty": 100,
         "expiration_date": "2028-01-10", "doctor_name": "Dr. Chen", "interaction_group": "nsaid",
         "refills_left": 3, "pharmacy_name": "Riverside Pharmacy", "cost_estimate": 8.0,
         "last_order_date": None, "mandate_id": None, "mandate_amount": None, "profile_id": "p2"},
        {"id": "m5", "name": "Vitamin D", "dosage_mg": "1000", "rx_number": "RX-67890",
         "schedule": "1 capsule at 8:00 AM", "doses_per_day": 1, "qty_remaining": 90, "initial_qty": 90,
         "expiration_date": "2027-11-30", "doctor_name": "Dr. Chen", "interaction_group": "unclassified",
         "refills_left": 5, "pharmacy_name": "MedPlus Pharmacy", "cost_estimate": 6.0,
         "last_order_date": None, "mandate_id": None, "mandate_amount": None, "profile_id": "p2"},
    ]
    for m in defaults:
        seed["meds"][m["id"]] = m
    save_store(seed)
    return seed


def save_store(store: dict) -> None:
    DATA_FILE.write_text(json.dumps(store, indent=2, default=str))


def get_medication(meds: dict, med_id: str) -> dict:
    m = meds.get(med_id)
    if not m:
        raise HTTPException(404, "Medication not found")
    return m


def enrich_medication(m: dict, store: dict) -> dict:
    return {
        **m,
        "days_of_supply": days_of_supply(m),
        "zero_stock_date": zero_stock_date(m),
        "adherence": adherence_rate(m["id"], store["dose_history"], m["doses_per_day"], 7),
        "low_stock": days_of_supply(m) <= LOW_STOCK_THRESHOLD_DAYS,
        "stock_pct": min(100, round((m["qty_remaining"] / max(m.get("initial_qty", 90), 1)) * 100)),
    }


def days_of_supply(m: dict) -> int:
    if m["doses_per_day"] <= 0:
        return 999
    return m["qty_remaining"] // m["doses_per_day"]


def zero_stock_date(m: dict) -> str:
    d = datetime.now() + timedelta(days=days_of_supply(m))
    return d.strftime("%b %d")


def adherence_rate(med_id: str, dose_history: list, doses_per_day: int, days: int = 7) -> str:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    actual = sum(1 for d in dose_history if d["med_id"] == med_id and d["timestamp"] >= cutoff)
    expected = doses_per_day * days
    rate = (actual / expected * 100) if expected > 0 else 0
    return f"{actual}/{expected} ({rate:.0f}%)"


def actual_burn_rate(med_id: str, dose_history: list, days: int = 14) -> float:
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    history = [d for d in dose_history if d["med_id"] == med_id and d["timestamp"] >= cutoff]
    if not history:
        return 0.0
    last = datetime.fromisoformat(history[-1]["timestamp"])
    span = min(days, max(1, (datetime.now() - last).days + 1))
    return len(history) / span


def trigger_notification(store: dict, recipient_role: str, message_type: str, content: str) -> None:
    now = datetime.now()
    stamp = now.strftime("%I:%M %p").lstrip("0")
    if stamp.startswith(":"):
        stamp = "0" + stamp
    banner = "URGENT" if message_type == "urgent_alert" else message_type.upper()
    line = f"[{stamp}] -> {recipient_role.upper()} ({banner}): {content}"
    store["log"].insert(0, line)


def pharmacy_reorder_request(rx_number: str, patient_info: dict, pharmacy_id: str) -> dict:
    return {
        "status": "submitted",
        "confirmation_number": f"CONF-{abs(hash(rx_number)) % 999999:06d}",
        "pharmacy_id": pharmacy_id,
        "submitted_at": datetime.now().isoformat(),
    }


app = FastAPI(title="PillVault AI")


@app.on_event("startup")
def startup():
    Path("uploads").mkdir(exist_ok=True)


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# ── Profile helpers / endpoints ────────────────────────────────────────────


def get_profile_meds(store: dict, profile_id: str | None = None) -> list:
    if not profile_id:
        profile_id = store.get("active_profile", "")
    return [m for m in store["meds"].values() if m.get("profile_id") == profile_id]


def get_profile(store: dict, profile_id: str) -> dict:
    p = store.get("profiles", {}).get(profile_id)
    if not p:
        raise HTTPException(404, "Profile not found")
    return p


@app.get("/api/profiles")
def list_profiles():
    store = load_store()
    profiles_list = list(store.get("profiles", {}).values())
    active_id = store.get("active_profile", "")
    for p in profiles_list:
        p["med_count"] = sum(1 for m in store["meds"].values() if m.get("profile_id") == p["id"])
        p["is_active"] = p["id"] == active_id
    return {"profiles": profiles_list, "active_profile": active_id}


@app.post("/api/profiles")
def create_profile(name: str = Form(...), avatar: str = Form(""), color: str = Form("indigo")):
    store = load_store()
    pid = f"p_{int(datetime.now().timestamp())}"
    av = avatar or "".join(w[0].upper() for w in name.split()[:2]) or "?"
    profile = {"id": pid, "name": name, "avatar": av, "color": color, "created_at": datetime.now().isoformat()}
    store.setdefault("profiles", {})[pid] = profile
    save_store(store)
    return {"success": True, "profile": profile}


@app.put("/api/profiles/{profile_id}")
def update_profile(profile_id: str, name: str = Form(""), avatar: str = Form(""), color: str = Form("")):
    store = load_store()
    p = get_profile(store, profile_id)
    if name:
        p["name"] = name
        p["avatar"] = avatar or "".join(w[0].upper() for w in name.split()[:2]) or "?"
    if color:
        p["color"] = color
    save_store(store)
    return {"success": True, "profile": p}


@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str):
    store = load_store()
    get_profile(store, profile_id)
    if len(store.get("profiles", {})) <= 1:
        raise HTTPException(400, "Cannot delete the last profile")
    store["profiles"].pop(profile_id, None)
    med_ids = [mid for mid, m in store["meds"].items() if m.get("profile_id") == profile_id]
    for mid in med_ids:
        store["meds"].pop(mid, None)
    store["dose_history"] = [d for d in store["dose_history"] if d.get("med_id") not in med_ids]
    if store.get("active_profile") == profile_id:
        remaining = list(store.get("profiles", {}).keys())
        store["active_profile"] = remaining[0] if remaining else ""
    save_store(store)
    return {"success": True}


@app.post("/api/profiles/{profile_id}/activate")
def activate_profile(profile_id: str):
    store = load_store()
    get_profile(store, profile_id)
    store["active_profile"] = profile_id
    save_store(store)
    return {"success": True, "active_profile": profile_id}


# ── API Endpoints ─────────────────────────────────────────────────────────


@app.get("/api/medications")
def list_medications(profile_id: str = "", search: str = "", low_stock: bool = False):
    store = load_store()
    meds_list = get_profile_meds(store, profile_id or None)
    if search:
        s = search.lower()
        meds_list = [m for m in meds_list if s in m["name"].lower() or s in m.get("doctor_name", "").lower() or s in m.get("rx_number", "").lower()]
    if low_stock:
        meds_list = [m for m in meds_list if days_of_supply(m) <= LOW_STOCK_THRESHOLD_DAYS]
    meds = [enrich_medication(m, store) for m in meds_list]
    return {"medications": meds}


@app.get("/api/medications/{med_id}")
def get_medication_detail(med_id: str):
    store = load_store()
    m = get_medication(store["meds"], med_id)
    enriched = enrich_medication(m, store)
    enriched["burn_rate"] = actual_burn_rate(m["id"], store["dose_history"])
    enriched["dose_history"] = [d for d in store["dose_history"] if d["med_id"] == med_id][-50:]
    return enriched


@app.post("/api/medications")
def add_medication(
    name: str = Form(...),
    dosage_mg: str = Form(""),
    rx_number: str = Form(""),
    schedule: str = Form(""),
    qty_remaining: int = Form(0),
    doctor_name: str = Form("Unknown"),
    interaction_group: str = Form("unclassified"),
    pharmacy_name: str = Form("MedPlus Pharmacy"),
    cost_estimate: float = Form(12.0),
    refills_left: int = Form(2),
    profile_id: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    store = load_store()
    doses_per_day = 2 if ("&" in schedule or " and " in schedule) else 1
    med_id = f"m_{int(datetime.now().timestamp())}"
    pid = profile_id or store.get("active_profile", "")

    parsed = {}
    if image:
        img_path = f"uploads/{uuid.uuid4()}_{image.filename}"
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        if _OPENCV_AVAILABLE:
            parsed = scan_medication_label(img_path)
        name = name or parsed.get("medication_name", name)
        dosage_mg = dosage_mg or str(parsed.get("dosage_mg", dosage_mg))
        rx_number = rx_number or parsed.get("rx_number", rx_number)
        qty_remaining = qty_remaining or int(parsed.get("total_quantity", 0))
        doctor_name = doctor_name or parsed.get("doctor_name", "Unknown")

    med = {
        "id": med_id,
        "name": name,
        "dosage_mg": dosage_mg,
        "rx_number": rx_number or f"RX-{abs(hash(name)) % 99999:05d}",
        "schedule": schedule or "Not specified",
        "doses_per_day": doses_per_day,
        "qty_remaining": qty_remaining,
        "initial_qty": qty_remaining,
        "expiration_date": parsed.get("expiration_date", "") or "",
        "doctor_name": doctor_name,
        "interaction_group": interaction_group,
        "refills_left": refills_left,
        "pharmacy_name": pharmacy_name,
        "cost_estimate": cost_estimate,
        "last_order_date": None,
        "mandate_id": None,
        "mandate_amount": None,
        "profile_id": pid,
    }
    store["meds"][med_id] = med

    profile_meds = get_profile_meds(store, pid)
    conflicting = INTERACTING_GROUPS.get(interaction_group, set())
    for o in profile_meds:
        if o["id"] != med_id and o["interaction_group"] in conflicting:
            trigger_notification(
                store, "caregiver", "urgent_alert",
                f"{name} may interact with {o['name']} already in cabinet. Consult a physician."
            )

    trigger_notification(
        store, "patient", "summary",
        f"Added {name} ({dosage_mg}mg). {qty_remaining} pills."
    )
    save_store(store)
    return {"success": True, "medication": med}


@app.put("/api/medications/{med_id}")
def update_medication(
    med_id: str,
    name: str = Form(""),
    dosage_mg: str = Form(""),
    rx_number: str = Form(""),
    schedule: str = Form(""),
    qty_remaining: int = Form(-1),
    doctor_name: str = Form(""),
    interaction_group: str = Form(""),
    pharmacy_name: str = Form(""),
    cost_estimate: float = Form(-1.0),
    refills_left: int = Form(-1),
    expiration_date: str = Form(""),
):
    store = load_store()
    m = get_medication(store["meds"], med_id)
    if name: m["name"] = name
    if dosage_mg: m["dosage_mg"] = dosage_mg
    if rx_number: m["rx_number"] = rx_number
    if schedule:
        m["schedule"] = schedule
        m["doses_per_day"] = 2 if ("&" in schedule or " and " in schedule) else 1
    if qty_remaining >= 0: m["qty_remaining"] = qty_remaining
    if doctor_name: m["doctor_name"] = doctor_name
    if interaction_group: m["interaction_group"] = interaction_group
    if pharmacy_name: m["pharmacy_name"] = pharmacy_name
    if cost_estimate >= 0: m["cost_estimate"] = cost_estimate
    if refills_left >= 0: m["refills_left"] = refills_left
    if expiration_date: m["expiration_date"] = expiration_date
    store["meds"][med_id] = m
    trigger_notification(store, "patient", "summary", f"Updated {m['name']} details.")
    save_store(store)
    return {"success": True, "medication": enrich_medication(m, store)}


@app.post("/api/medications/{med_id}/dose")
def log_dose(med_id: str):
    store = load_store()
    m = get_medication(store["meds"], med_id)
    now = datetime.now()
    event = {
        "id": f"d_{int(now.timestamp())}_{uuid.uuid4().hex[:4]}",
        "med_id": med_id,
        "timestamp": now.isoformat(),
        "quantity": 1,
    }
    store["dose_history"].insert(0, event)
    m["qty_remaining"] = max(0, m["qty_remaining"] - 1)
    save_store(store)

    if m["qty_remaining"] <= 3:
        draft_refill(store, m)

    return {
        "success": True,
        "qty_remaining": m["qty_remaining"],
        "days_of_supply": days_of_supply(m),
        "adherence": adherence_rate(m["id"], store["dose_history"], m["doses_per_day"], 7),
    }


def draft_refill(store: dict, m: dict):
    if m.get("last_order_date"):
        days_since = (datetime.now() - datetime.fromisoformat(m["last_order_date"])).days
        if days_since < DUPLICATE_ORDER_LOCK_DAYS:
            return

    already_pending = any(
        n["med_id"] == m["id"] and n["kind"] == "refill" and n["status"] == "pending"
        for n in store["notifications"].values()
    )
    if already_pending:
        return

    notif_id = f"n_{int(datetime.now().timestamp())}"

    if m.get("mandate_id"):
        message = (
            f"{m['name']}'s supply runs out around {zero_stock_date(m)}. "
            f"${m['cost_estimate']:.2f} at {m['pharmacy_name']} — "
            f"will auto-charge mandate {m['mandate_id']}. "
            f"Reply APPROVE to authorize."
        )
    elif m.get("refills_left", 0) > 0:
        message = (
            f"{m['name']}'s supply runs out around {zero_stock_date(m)}. "
            f"A refill of ${m['cost_estimate']:.2f} at {m['pharmacy_name']} is ready. "
            f"Reply APPROVE to authorize."
        )
    else:
        message = (
            f"{m['name']} has 0 refills left. Supply runs out around "
            f"{zero_stock_date(m)}. Refill authorization from "
            f"{m['doctor_name']} needed."
        )

    store["notifications"][notif_id] = {
        "id": notif_id,
        "med_id": m["id"],
        "kind": "refill",
        "status": "pending",
        "message": message,
        "created_at": datetime.now().isoformat(),
    }
    trigger_notification(store, "caregiver", "refill_approval", message)
    save_store(store)


@app.get("/api/notifications")
def list_notifications():
    store = load_store()
    return {"notifications": list(store["notifications"].values())}


@app.post("/api/notifications/{notif_id}/approve")
def approve_refill(notif_id: str):
    store = load_store()
    n = store["notifications"].get(notif_id)
    if not n or n["status"] != "pending":
        raise HTTPException(400, "No pending notification with that id")

    m = get_medication(store["meds"], n["med_id"])
    success = False
    confirmation = None

    if m.get("mandate_id") and _PRAVA_AVAILABLE:
        key = os.environ.get("PRAVA_SECRET_KEY", "")
        if key:
            try:
                client = PravaClient(PravaConfig.from_env())
                result = client.auto_charge_refill(
                    mandate_id=m["mandate_id"],
                    amount=f"{m['cost_estimate']:.2f}",
                    medication_name=m["name"],
                    pharmacy_name=m["pharmacy_name"],
                    rx_number=m["rx_number"],
                )
                if result["success"]:
                    confirmation = result["confirmation"]
                    success = True
            except Exception:
                pass

    if not success and _PRAVA_AVAILABLE:
        key = os.environ.get("PRAVA_SECRET_KEY", "")
        if key:
            try:
                client = PravaClient(PravaConfig.from_env())
                pay_result = client.pay_pharmacy_order(
                    medication_name=m["name"],
                    amount=f"{m['cost_estimate']:.2f}",
                    pharmacy_name=m["pharmacy_name"],
                    rx_number=m["rx_number"],
                    open_browser=False,
                )
                if pay_result["success"]:
                    confirmation = pay_result["confirmation"]
                    success = True
            except Exception:
                pass

    if not success:
        result = pharmacy_reorder_request(
            rx_number=m["rx_number"],
            patient_info={"user_id": "patient_1"},
            pharmacy_id=m["pharmacy_name"],
        )
        confirmation = result["confirmation_number"]

    m["last_order_date"] = datetime.now().isoformat()
    if m.get("refills_left", 0) > 0:
        m["refills_left"] -= 1
    n["status"] = "approved"

    trigger_notification(
        store, "patient", "summary",
        f"Your {m['name']} refill was ordered (confirmation {confirmation})."
    )
    save_store(store)
    return {"success": True, "confirmation": confirmation}


@app.post("/api/notifications/{notif_id}/decline")
def decline_refill(notif_id: str):
    store = load_store()
    n = store["notifications"].get(notif_id)
    if n:
        n["status"] = "declined"
        save_store(store)
    return {"success": True}


@app.get("/api/dashboard")
def caregiver_dashboard(profile_id: str = ""):
    store = load_store()
    meds_list = get_profile_meds(store, profile_id or None)
    meds = []
    for m in meds_list:
        meds.append({
            **m,
            "days_of_supply": days_of_supply(m),
            "zero_stock_date": zero_stock_date(m),
            "adherence": adherence_rate(m["id"], store["dose_history"], m["doses_per_day"], 7),
            "low_stock": days_of_supply(m) <= LOW_STOCK_THRESHOLD_DAYS,
        })

    pending = [n for n in store["notifications"].values() if n["status"] == "pending"]
    recent_log = store["log"][:20]

    return {
        "medications": meds,
        "pending_approvals": pending,
        "recent_log": recent_log,
        "total_meds": len(meds),
        "low_stock_count": sum(1 for m in meds if m["low_stock"]),
        "pending_count": len(pending),
    }


@app.get("/api/reports/adherence")
def adherence_report(profile_id: str = ""):
    store = load_store()
    meds_list = get_profile_meds(store, profile_id or None)
    reports = []
    for m in meds_list:
        rate = adherence_rate(m["id"], store["dose_history"], m["doses_per_day"], 7)
        burn = actual_burn_rate(m["id"], store["dose_history"])
        reports.append({
            "name": m["name"],
            "dosage_mg": m["dosage_mg"],
            "adherence": rate,
            "burn_rate": round(burn, 1),
            "expected_per_day": m["doses_per_day"],
            "qty_remaining": m["qty_remaining"],
        })
    return {"reports": reports}


@app.get("/api/reports/adherence/csv")
def adherence_csv(profile_id: str = ""):
    store = load_store()
    meds_list = get_profile_meds(store, profile_id or None)
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Medication", "Dosage (mg)", "Adherence", "Actual/Day", "Expected/Day", "Remaining", "Days of Supply"])
    for m in meds_list:
        rate = adherence_rate(m["id"], store["dose_history"], m["doses_per_day"], 7)
        burn = actual_burn_rate(m["id"], store["dose_history"])
        writer.writerow([m["name"], m["dosage_mg"], rate, round(burn, 1), m["doses_per_day"], m["qty_remaining"], days_of_supply(m)])
    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=adherence_report.csv"})


@app.get("/api/medications/{med_id}/history")
def medication_history(med_id: str):
    store = load_store()
    m = get_medication(store["meds"], med_id)
    history = [d for d in store["dose_history"] if d["med_id"] == med_id]
    days = 30
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    recent = [d for d in history if d["timestamp"] >= cutoff]
    daily = {}
    for d in recent:
        day = d["timestamp"][:10]
        daily[day] = daily.get(day, 0) + 1
    labels = []
    data = []
    for i in range(days - 1, -1, -1):
        day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        labels.append(day)
        data.append(daily.get(day, 0))
    return {
        "medication": enrich_medication(m, store),
        "labels": labels,
        "data": data,
        "expected_per_day": m["doses_per_day"],
    }


@app.post("/api/scan")
async def scan_label(image: UploadFile = File(...)):
    if not _OPENCV_AVAILABLE:
        return {"success": False, "message": "OpenCV not available", "data": {}}

    img_path = f"uploads/{uuid.uuid4()}_{image.filename}"
    with open(img_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        result = scan_medication_label(img_path)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "message": str(e), "data": {}}


# ── Prava endpoints ──────────────────────────────────────────────────────

def get_prava_client():
    key = os.environ.get("PRAVA_SECRET_KEY", "")
    if not key:
        raise HTTPException(400, "Prava not configured. Set PRAVA_SECRET_KEY.")
    return PravaClient(PravaConfig.from_env())


@app.post("/api/medications/{med_id}/checkin")
def checkin_followup(med_id: str, hours_since_scheduled: float = 2.0):
    store = load_store()
    m = get_medication(store["meds"], med_id)
    if hours_since_scheduled >= 4:
        trigger_notification(
            store, "caregiver", "unresponsive",
            f"{m['name']} dose not confirmed — {hours_since_scheduled:.0f}h past schedule. Please check on patient."
        )
    elif hours_since_scheduled >= 2:
        trigger_notification(
            store, "patient", "checkin",
            f"Have you taken your {m['name']} ({m['dosage_mg']}mg) yet?"
        )
    save_store(store)
    return {"success": True, "hours": hours_since_scheduled, "message": f"Check-in sent for {m['name']}"}


@app.post("/api/mandates/setup/{med_id}")
def setup_mandate(med_id: str):
    store = load_store()
    m = get_medication(store["meds"], med_id)
    if not _PRAVA_AVAILABLE:
        raise HTTPException(400, "Prava library not available")

    client = get_prava_client()
    result = client.setup_mandate(
        total_amount=f"{m['cost_estimate']:.2f}",
        currency="USD",
        merchant_name=m["pharmacy_name"],
        merchant_url="https://medplus.example.com",
        product_description=f"Monthly refill: {m['name']}",
        unit_price=f"{m['cost_estimate']:.2f}",
        recurring_frequency="monthly",
        max_charges=12,
        product_id=m["rx_number"],
        open_browser=True,
    )

    if result["success"] and result.get("mandate_id"):
        m["mandate_id"] = result["mandate_id"]
        m["mandate_amount"] = f"{m['cost_estimate']:.2f}"
        store["meds"][med_id] = m
        save_store(store)

    return result


@app.get("/api/mandates")
def list_mandates():
    if not _PRAVA_AVAILABLE:
        return {"mandates": []}
    try:
        client = get_prava_client()
        mandates = client.list_mandates(standing_only=True)
        return {"mandates": [vars(m) for m in mandates]}
    except Exception:
        return {"mandates": []}


@app.post("/api/mandates/{mandate_id}/cancel")
def cancel_mandate(mandate_id: str):
    if not _PRAVA_AVAILABLE:
        raise HTTPException(400, "Prava not available")
    client = get_prava_client()
    try:
        client.cancel_mandate(mandate_id)
        store = load_store()
        for m in store["meds"].values():
            if m.get("mandate_id") == mandate_id:
                m["mandate_id"] = None
                m["mandate_amount"] = None
        save_store(store)
        return {"success": True}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/prava/status")
def prava_status():
    ok = bool(os.environ.get("PRAVA_SECRET_KEY", ""))
    return {
        "enabled": ok and _PRAVA_AVAILABLE,
        "sandbox": os.environ.get("PRAVA_ENV", "sandbox") == "sandbox",
        "email": os.environ.get("PRAVA_USER_EMAIL", ""),
    }


@app.get("/api/log")
def get_log():
    store = load_store()
    return {"log": store["log"][:50]}


# ── Serve Frontend ───────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    html = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
