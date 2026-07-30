#!/usr/bin/env python3
"""
PillVault AI — autonomous medication cabinet agent.

Workflows:
  A — Cabinet audit & ingestion (vision parsing of a label photo)
  B — Daily dose logging, history & adherence tracking
  C — Predictive refill drafting & caregiver approval with Prava payments

Run:  python pillvault_agent.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DATA_FILE = Path(__file__).with_name("pillvault_data.json")
DUPLICATE_ORDER_LOCK_DAYS = 14
LOW_STOCK_THRESHOLD_DAYS = 5

try:
    from prava import PravaClient, PravaConfig
    _PRAVA_AVAILABLE = True
except ImportError:
    _PRAVA_AVAILABLE = False

try:
    from opencv_scanner import scan_medication_label
    _OPENCV_AVAILABLE = True
except ImportError:
    _OPENCV_AVAILABLE = False


# ── Data model ──────────────────────────────────────────────────────────

@dataclass
class DoseEvent:
    id: str
    med_id: str
    timestamp: str
    quantity: int = 1


@dataclass
class Medication:
    id: str
    name: str
    dosage_mg: str
    rx_number: str
    schedule: str
    doses_per_day: int
    qty_remaining: int
    doctor_name: str = "Unknown"
    interaction_group: str = "unclassified"
    refills_left: int = 2
    pharmacy_name: str = "MedPlus Pharmacy"
    cost_estimate: float = 12.0
    last_order_date: Optional[str] = None
    mandate_id: Optional[str] = None
    mandate_amount: Optional[str] = None

    def days_of_supply(self) -> int:
        if self.doses_per_day <= 0:
            return 999
        return self.qty_remaining // self.doses_per_day

    def zero_stock_date(self) -> str:
        d = datetime.now() + timedelta(days=self.days_of_supply())
        return d.strftime("%b %d")

    def has_active_mandate(self) -> bool:
        return bool(self.mandate_id)


@dataclass
class Notification:
    id: str
    med_id: str
    kind: str
    status: str = "pending"
    message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Store:
    meds: dict = field(default_factory=dict)
    notifications: dict = field(default_factory=dict)
    dose_history: list = field(default_factory=list)
    log: list = field(default_factory=list)


INTERACTING_GROUPS = {
    "anticoagulant": {"nsaid"},
    "nsaid": {"anticoagulant"},
}


# ── Persistence ─────────────────────────────────────────────────────────

def load_store() -> Store:
    if DATA_FILE.exists():
        raw = json.loads(DATA_FILE.read_text())
        return Store(
            meds=raw.get("meds", {}),
            notifications=raw.get("notifications", {}),
            dose_history=raw.get("dose_history", []),
            log=raw.get("log", []),
        )
    seed = Store()
    for m in [
        Medication(id="m1", name="Lisinopril", dosage_mg="10", rx_number="RX-88213",
                    schedule="1 tablet at 8:00 AM", doses_per_day=1, qty_remaining=6,
                    doctor_name="Dr. Alvarez", interaction_group="ace_inhibitor",
                    refills_left=2, pharmacy_name="MedPlus Pharmacy", cost_estimate=12.0),
        Medication(id="m2", name="Metformin", dosage_mg="500", rx_number="RX-77410",
                    schedule="1 tablet at 8:00 AM and 6:00 PM", doses_per_day=2, qty_remaining=42,
                    doctor_name="Dr. Alvarez", interaction_group="biguanide",
                    refills_left=1, pharmacy_name="MedPlus Pharmacy", cost_estimate=9.0),
        Medication(id="m3", name="Atorvastatin", dosage_mg="20", rx_number="RX-90055",
                    schedule="1 tablet at 9:00 PM", doses_per_day=1, qty_remaining=3,
                    doctor_name="Dr. Chen", interaction_group="statin",
                    refills_left=0, pharmacy_name="Riverside Pharmacy", cost_estimate=15.0),
    ]:
        seed.meds[m.id] = asdict(m)
    return seed


def save_store(store: Store) -> None:
    DATA_FILE.write_text(json.dumps(asdict(store), indent=2, default=str))


# ── Tools ───────────────────────────────────────────────────────────────

def vision_parser(image_path: Optional[str] = None) -> dict:
    """Read a medication label using OpenCV+OCR, Claude vision, or manual entry."""

    # 1) OpenCV + EasyOCR (fast, local, no API key needed)
    if image_path and _OPENCV_AVAILABLE:
        print("  [cv] Scanning label with OpenCV + EasyOCR...")
        try:
            result = scan_medication_label(image_path)
            if result.get("medication_name"):
                print(f"  [cv] Detected: {result['medication_name']} "
                      f"{result.get('dosage_mg', '')}mg")
                return result
            print("  [cv] Could not parse label text.")
        except Exception as e:
            print(f"  [cv] Error: {e}")

    # 2) Claude vision (requires ANTHROPIC_API_KEY)
    if image_path and os.environ.get("ANTHROPIC_API_KEY"):
        print("  [vision] Calling Claude...")
        try:
            import anthropic
            client = anthropic.Anthropic()
            img_bytes = Path(image_path).read_bytes()
            media_type = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            b64 = base64.b64encode(img_bytes).decode()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": (
                            "Read this medication label photo. Reply with ONLY a JSON object, "
                            "no other text, with keys: medication_name, dosage_mg, rx_number, "
                            "total_quantity, expiration_date, doctor_name. Use null for anything "
                            "not visible on the label."
                        )},
                    ],
                }],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            text = text.strip().strip("`").removeprefix("json").strip()
            result = json.loads(text)
            if result.get("medication_name"):
                return result
        except Exception as e:
            print(f"  [vision] Claude failed: {e}")

    # 3) Manual entry (fallback)
    print("\n-- Manual label entry --")
    return {
        "medication_name": input("  Medication name: ").strip(),
        "dosage_mg": input("  Dosage (mg): ").strip(),
        "rx_number": input("  Rx number: ").strip() or None,
        "total_quantity": int(input("  Tablets in bottle: ").strip() or 0),
        "expiration_date": input("  Expiration date (YYYY-MM-DD): ").strip() or None,
        "doctor_name": input("  Doctor name: ").strip() or "Unknown",
    }


def db_read_inventory(store: Store) -> list[Medication]:
    return [Medication(**m) for m in store.meds.values()]


def db_update_inventory(store: Store, item_id: str, adjustment_type: str, quantity: int) -> Medication:
    m = Medication(**store.meds[item_id])
    if adjustment_type == "dose_taken":
        m.qty_remaining = max(0, m.qty_remaining - quantity)
    elif adjustment_type == "manual_add":
        m.qty_remaining += quantity
    elif adjustment_type == "discard_expired":
        m.qty_remaining = max(0, m.qty_remaining - quantity)
    store.meds[item_id] = asdict(m)
    return m


def trigger_notification(store: Store, recipient_role: str, message_type: str, content: str) -> None:
    now = datetime.now()
    stamp = now.strftime("%I:%M %p").lstrip("0")
    if stamp.startswith(":"):
        stamp = "0" + stamp
    banner = "URGENT" if message_type == "urgent_alert" else message_type.upper()
    line = f"[{stamp}] -> {recipient_role.upper()} ({banner}): {content}"
    print(line)
    store.log.insert(0, line)


def pharmacy_reorder_request(rx_number: str, patient_info: dict, pharmacy_id: str) -> dict:
    """Simulated pharmacy order. Replace with a real API call."""
    return {
        "status": "submitted",
        "confirmation_number": f"CONF-{abs(hash(rx_number)) % 999999:06d}",
        "pharmacy_id": pharmacy_id,
        "submitted_at": datetime.now().isoformat(),
    }


# ── CLI helpers ─────────────────────────────────────────────────────────

def fmt_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a simple aligned table."""
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    sep = "  ".join("-" * w for w in col_widths)
    header_line = "  ".join(h.ljust(w) for h, w in zip(header, col_widths))
    body = "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, col_widths))
        for row in rows
    )
    return f"{header_line}\n{sep}\n{body}"


def status_tag(status: str) -> str:
    tags = {
        "active": "[ACTIVE]",
        "paused": "[PAUSED]",
        "consumed": "[USED]",
        "expired": "[EXP]",
        "cancelled": "[CANCEL]",
        "pending": "[PEND]",
        "approved": "[OK]",
        "declined": "[NO]",
        "available": "[AVAIL]",
    }
    return tags.get(status, f"[{status.upper()}]")


# ── Agent ───────────────────────────────────────────────────────────────

class PillVaultAgent:
    def __init__(self, store: Store, user_id: str = "patient_1"):
        self.store = store
        self.user_id = user_id
        self.prava_enabled = False
        self._prava_client = None

    def _init_prava(self) -> bool:
        if self._prava_client is not None:
            return True
        if not _PRAVA_AVAILABLE:
            return False
        key = os.environ.get("PRAVA_SECRET_KEY", "")
        if not key:
            return False
        self._prava_client = PravaClient(PravaConfig.from_env())
        self.prava_enabled = True
        return True

    # ── Workflow A: Cabinet audit ───────────────────────────────────────

    def audit_photo(self, image_path: Optional[str] = None):
        parsed = vision_parser(image_path)
        name = parsed.get("medication_name")
        if not name:
            print("Could not read a medication name.")
            return

        existing = next((m for m in db_read_inventory(self.store) if m.name.lower() == name.lower()), None)

        if existing:
            existing.qty_remaining += int(parsed.get("total_quantity") or 0)
            self.store.meds[existing.id] = asdict(existing)
            med = existing
            print(f"  Updated {med.name}: now {med.qty_remaining} tablets.")
        else:
            group = input("  Interaction group (e.g. anticoagulant/nsaid/statin): ").strip() or "unclassified"
            schedule = input("  Daily schedule (e.g. '1 tablet at 8:00 AM'): ").strip()
            doses_per_day = 2 if ("&" in schedule or " and " in schedule) else 1
            med = Medication(
                id=f"m_{int(datetime.now().timestamp())}",
                name=name,
                dosage_mg=str(parsed.get("dosage_mg") or ""),
                rx_number=parsed.get("rx_number") or f"RX-{abs(hash(name)) % 99999:05d}",
                schedule=schedule or "Not specified",
                doses_per_day=doses_per_day,
                qty_remaining=int(parsed.get("total_quantity") or 0),
                doctor_name=parsed.get("doctor_name") or "Unknown",
                interaction_group=group,
            )
            self.store.meds[med.id] = asdict(med)
            print(f"  Added {med.name} ({med.dosage_mg}mg).")

            conflicting_with = INTERACTING_GROUPS.get(group, set())
            others = [m for m in db_read_inventory(self.store) if m.id != med.id]
            conflict = next((o for o in others if o.interaction_group in conflicting_with), None)
            if conflict:
                trigger_notification(
                    self.store, "caregiver", "urgent_alert",
                    f"{med.name} may interact with {conflict.name} already in cabinet. "
                    f"Dose logging paused — consult a physician."
                )
                save_store(self.store)
                return

        trigger_notification(
            self.store, "patient", "summary",
            f"Added {med.name} ({med.dosage_mg}mg). {med.qty_remaining} pills, "
            f"lasts until {med.zero_stock_date()}."
        )
        save_store(self.store)

    # ── Workflow B: Dose logging & adherence ────────────────────────────

    def log_dose(self, med_id: str):
        meds = {m.id: m for m in db_read_inventory(self.store)}
        med = meds.get(med_id)
        if not med:
            print("  Unknown medication.")
            return

        now = datetime.now()
        event = DoseEvent(
            id=f"d_{int(now.timestamp())}",
            med_id=med_id,
            timestamp=now.isoformat(),
            quantity=1,
        )
        self.store.dose_history.insert(0, asdict(event))

        med = db_update_inventory(self.store, med_id, "dose_taken", 1)
        print(f"  Logged: {med.name} — {med.qty_remaining} left "
              f"({med.days_of_supply()} days).")

        self._check_adherence(med)
        self._check_burn_rate(med)
        save_store(self.store)

    def dose_history_for(self, med_id: str, days: int = 30) -> list[DoseEvent]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        return [
            DoseEvent(**d) for d in self.store.dose_history
            if d["med_id"] == med_id and d["timestamp"] >= cutoff
        ]

    def adherence_rate(self, med_id: str, days: int = 7) -> str:
        med = next((m for m in db_read_inventory(self.store) if m.id == med_id), None)
        if not med or med.doses_per_day <= 0:
            return "N/A"
        expected = med.doses_per_day * days
        actual = len(self.dose_history_for(med_id, days))
        rate = (actual / expected * 100) if expected > 0 else 0
        return f"{actual}/{expected} ({rate:.0f}%)"

    def actual_burn_rate(self, med_id: str, days: int = 14) -> float:
        """Average tablets per day from dose history."""
        history = self.dose_history_for(med_id, days)
        if not history:
            return 0.0
        return len(history) / min(days, max(1, (datetime.now() - datetime.fromisoformat(history[-1].timestamp)).days + 1))

    def _check_adherence(self, med: Medication):
        rate = self.adherence_rate(med.id, 7)
        print(f"  Adherence (7d): {rate}")

    def _check_burn_rate(self, med: Medication):
        if med.days_of_supply() > LOW_STOCK_THRESHOLD_DAYS:
            return
        self._draft_refill(med)

    # ── Workflow C: Refill & mandates ───────────────────────────────────

    def _draft_refill(self, med: Medication):
        if med.last_order_date:
            days_since = (datetime.now() - datetime.fromisoformat(med.last_order_date)).days
            if days_since < DUPLICATE_ORDER_LOCK_DAYS:
                return

        already_pending = any(
            n["med_id"] == med.id and n["kind"] == "refill" and n["status"] == "pending"
            for n in self.store.notifications.values()
        )
        if already_pending:
            return

        notif_id = f"n_{int(datetime.now().timestamp())}"

        if med.has_active_mandate():
            message = (
                f"{med.name}'s supply runs out around {med.zero_stock_date()}. "
                f"${med.cost_estimate:.2f} at {med.pharmacy_name} — "
                f"will auto-charge mandate {med.mandate_id}. "
                f"Reply APPROVE to authorize."
            )
        elif med.refills_left > 0:
            message = (
                f"{med.name}'s supply runs out around {med.zero_stock_date()}. "
                f"A refill of ${med.cost_estimate:.2f} at {med.pharmacy_name} is ready. "
                f"Reply APPROVE to authorize the order."
            )
        else:
            message = (
                f"{med.name} has 0 refills left. Supply runs out around "
                f"{med.zero_stock_date()}. Refill authorization from "
                f"{med.doctor_name} needed."
            )

        self.store.notifications[notif_id] = asdict(Notification(
            id=notif_id, med_id=med.id, kind="refill", message=message,
        ))
        trigger_notification(self.store, "caregiver", "refill_approval", message)
        save_store(self.store)

    def approve_refill(self, notif_id: str):
        n = self.store.notifications.get(notif_id)
        if not n or n["status"] != "pending":
            print("  No pending notification with that id.")
            return
        med = Medication(**self.store.meds[n["med_id"]])

        success = False
        confirmation = None

        # Try charging an existing mandate first (no passkey needed)
        if med.has_active_mandate() and self._init_prava():
            try:
                result = self._prava_client.auto_charge_refill(
                    mandate_id=med.mandate_id,
                    amount=f"{med.cost_estimate:.2f}",
                    medication_name=med.name,
                    pharmacy_name=med.pharmacy_name,
                    rx_number=med.rx_number,
                )
                if result["success"]:
                    print(f"  [Prava] Auto-charged mandate: ${med.cost_estimate:.2f}")
                    confirmation = result["confirmation"]
                    success = True
                else:
                    print(f"  [Prava] Mandate charge failed: {result['message']}")
            except Exception as e:
                print(f"  [Prava] Error: {e}")

        # Fallback: try one-time Prava payment
        if not success and self._init_prava():
            try:
                pay_result = self._prava_client.pay_pharmacy_order(
                    medication_name=med.name,
                    amount=f"{med.cost_estimate:.2f}",
                    pharmacy_name=med.pharmacy_name,
                    rx_number=med.rx_number,
                    open_browser=True,
                )
                if pay_result["success"]:
                    confirmation = pay_result["confirmation"]
                    success = True
                else:
                    print(f"  [Prava] Payment failed: {pay_result['message']}")
            except Exception as e:
                print(f"  [Prava] Error: {e}")

        # Final fallback: simulated order
        if not success:
            print("  Using simulated pharmacy order.")
            result = pharmacy_reorder_request(
                rx_number=med.rx_number,
                patient_info={"user_id": self.user_id},
                pharmacy_id=med.pharmacy_name,
            )
            confirmation = result["confirmation_number"]

        med.last_order_date = datetime.now().isoformat()
        if med.refills_left > 0:
            med.refills_left -= 1
        self.store.meds[med.id] = asdict(med)
        n["status"] = "approved"
        trigger_notification(
            self.store, "patient", "summary",
            f"Your {med.name} refill was ordered (confirmation {confirmation})."
        )
        save_store(self.store)

    def decline_refill(self, notif_id: str):
        n = self.store.notifications.get(notif_id)
        if not n:
            return
        n["status"] = "declined"
        save_store(self.store)

    # ── Prava mandate management ────────────────────────────────────────

    def setup_pharmacy_mandate(self, med_id: str):
        """Set up a recurring Prava mandate for a medication's pharmacy."""
        med = next((m for m in db_read_inventory(self.store) if m.id == med_id), None)
        if not med:
            print("  Unknown medication.")
            return
        if not self._init_prava():
            print("  Prava not configured. Set PRAVA_SECRET_KEY.")
            return

        print(f"\n  Setting up monthly mandate for {med.name} at {med.pharmacy_name}")
        print(f"  Amount: ${med.cost_estimate:.2f}/month")

        result = self._prava_client.setup_mandate(
            total_amount=f"{med.cost_estimate:.2f}",
            currency="USD",
            merchant_name=med.pharmacy_name,
            merchant_url="https://medplus.example.com",
            product_description=f"Monthly refill: {med.name}",
            unit_price=f"{med.cost_estimate:.2f}",
            recurring_frequency="monthly",
            max_charges=12,
            product_id=med.rx_number,
            open_browser=True,
        )

        if result["success"] and result["mandate_id"]:
            med.mandate_id = result["mandate_id"]
            med.mandate_amount = f"{med.cost_estimate:.2f}"
            self.store.meds[med.id] = asdict(med)
            save_store(self.store)
            print(f"  Mandate {result['mandate_id']} linked to {med.name}.")
        else:
            print(f"  Mandate setup: {result['message']}")

    def list_prava_mandates(self):
        if not self._init_prava():
            print("  Prava not configured.")
            return
        try:
            mandates = self._prava_client.list_mandates(standing_only=True)
            if not mandates:
                print("  No standing mandates found.")
                return

            header = ["ID", "Merchant", "Amount/mo", "Left", "Status", "Freq"]
            rows = []
            for m in mandates:
                rows.append([
                    m.id[:20],
                    m.merchant_name[:18],
                    f"${m.approved_amount}",
                    f"${m.remaining}",
                    status_tag(m.status),
                    m.recurring_frequency[:6],
                ])
            print(f"\n{fmt_table(header, rows)}\n")
        except Exception as e:
            print(f"  Error: {e}")

    def cancel_prava_mandate(self, mandate_id: str):
        if not self._init_prava():
            print("  Prava not configured.")
            return
        try:
            result = self._prava_client.cancel_mandate(mandate_id)
            print(f"  Mandate {mandate_id} cancelled.")
            # Unlink from any medication
            for med_id, m in self.store.meds.items():
                if m.get("mandate_id") == mandate_id:
                    m["mandate_id"] = None
                    m["mandate_amount"] = None
                    self.store.meds[med_id] = m
            save_store(self.store)
        except Exception as e:
            print(f"  Error: {e}")

    # ── Unresponsive patient ────────────────────────────────────────────

    def checkin_followup(self, med_id: str, hours_since_scheduled: float):
        med = Medication(**self.store.meds[med_id])
        if hours_since_scheduled >= 4:
            trigger_notification(
                self.store, "caregiver", "unresponsive",
                f"Dose of {med.name} not confirmed yet."
            )
        elif hours_since_scheduled >= 2:
            trigger_notification(
                self.store, "patient", "checkin",
                f"Have you taken your {med.name} yet?"
            )
        save_store(self.store)

    # ── Reports ─────────────────────────────────────────────────────────

    def caregiver_summary(self) -> str:
        lines = ["\n=== Caregiver Dashboard ==="]
        for m in db_read_inventory(self.store):
            mandate_tag = f" [mandate:{m.mandate_id[:12]}]" if m.mandate_id else ""
            lines.append(
                f"  {m.name:<14} {m.qty_remaining:>3} tabs  |  "
                f"{m.days_of_supply():>2}d left (until {m.zero_stock_date()})  |  "
                f"refills: {m.refills_left}  |  {m.pharmacy_name}{mandate_tag}"
            )

        pending = [n for n in self.store.notifications.values() if n["status"] == "pending"]
        if pending:
            lines.append(f"\n  Pending approvals ({len(pending)}):")
            for n in pending:
                lines.append(f"    [{n['id']}] {n['message']}")
        else:
            lines.append("\n  No pending approvals.")
        return "\n".join(lines)

    def adherence_report(self) -> str:
        lines = ["\n=== Adherence Report (7-day) ==="]
        for m in db_read_inventory(self.store):
            rate = self.adherence_rate(m.id, 7)
            actual = self.actual_burn_rate(m.id)
            status = "[OK]" if "100" in rate else "[MISSED]"
            lines.append(
                f"  {m.name:<14} {status}  adherence: {rate:<10}  "
                f"actual: {actual:.1f}/day  expected: {m.doses_per_day}/day"
            )
        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    store = load_store()
    agent = PillVaultAgent(store)

    menu = """
+++ PillVault AI ++++++++++++++++++++++++++++++++++
 1. Scan new medicine       (Workflow A)
 2. Log a dose taken        (Workflow B)
 3. Caregiver dashboard     (Workflow C)
 4. Approve a refill
 5. Decline a refill
 6. Unresponsive check-in
 7. Set up pharmacy mandate (Prava)
 8. List Prava mandates
 9. Cancel a Prava mandate
 A. Adherence report
 P. Prava payment status
 X. Exit
+++++++++++++++++++++++++++++++++++++++++++++++++++
"""
    while True:
        print(menu)
        choice = input("Choice: ").strip().lower()

        # ── Workflow A ──
        if choice == "1":
            path = None
            _cv_ok = _OPENCV_AVAILABLE
            _cam_ok = False
            if not _cv_ok:
                try:
                    import cv2 as _cv2
                    _cv_ok = True
                except ImportError:
                    pass
            if _cv_ok:
                try:
                    from opencv_scanner import capture_from_camera
                    _cam_ok = True
                except ImportError:
                    pass
                src = input("  [C]amera, [F]ile path, or Enter for manual: ").strip().lower()
                if src == "c" and _cam_ok:
                    path = capture_from_camera()
                elif src == "f":
                    path = input("  Photo path: ").strip() or None
            else:
                path = input("  Photo path (blank for manual): ").strip() or None
            agent.audit_photo(path)

        # ── Workflow B ──
        elif choice == "2":
            meds = db_read_inventory(store)
            if not meds:
                print("  No medications — scan one first.")
                continue
            print(f"\n{fmt_table(['ID', 'Name', 'Dose', 'Remaining'],
                                 [[m.id, m.name, f"{m.dosage_mg}mg", str(m.qty_remaining)] for m in meds])}")
            med_id = input("  Medication id to log: ").strip()
            if med_id in store.meds:
                agent.log_dose(med_id)
            else:
                print("  Unknown id.")

        # ── Workflow C — dashboard ──
        elif choice == "3":
            print(agent.caregiver_summary())

        # ── Approve refill ──
        elif choice == "4":
            pending = [n for n in store.notifications.values() if n["status"] == "pending"]
            if not pending:
                print("  Nothing pending.")
                continue
            for n in pending:
                print(f"  [{n['id']}] {n['message']}")
            nid = input("  Notification id to approve: ").strip()
            agent.approve_refill(nid)

        # ── Decline refill ──
        elif choice == "5":
            pending = [n for n in store.notifications.values() if n["status"] == "pending"]
            if not pending:
                print("  Nothing pending.")
                continue
            for n in pending:
                print(f"  [{n['id']}] {n['message']}")
            nid = input("  Notification id to decline: ").strip()
            agent.decline_refill(nid)

        # ── Unresponsive check-in ──
        elif choice == "6":
            meds = db_read_inventory(store)
            for m in meds:
                print(f"  [{m.id}] {m.name}")
            med_id = input("  Medication id: ").strip()
            if med_id in store.meds:
                hrs = float(input("  Hours since scheduled dose: ").strip() or 0)
                agent.checkin_followup(med_id, hrs)

        # ── Prava: setup mandate ──
        elif choice == "7":
            meds = db_read_inventory(store)
            if not meds:
                print("  No medications.")
                continue
            print(f"\n{fmt_table(['ID', 'Name', 'Cost', 'Pharmacy', 'Mandate'],
                                 [[m.id, m.name, f"${m.cost_estimate:.2f}", m.pharmacy_name,
                                   status_tag("active") if m.mandate_id else "[none]"]
                                  for m in meds])}")
            med_id = input("  Medication id for mandate: ").strip()
            if med_id in store.meds:
                agent.setup_pharmacy_mandate(med_id)

        # ── Prava: list mandates ──
        elif choice == "8":
            agent.list_prava_mandates()

        # ── Prava: cancel mandate ──
        elif choice == "9":
            mid = input("  Mandate id to cancel: ").strip()
            if mid:
                agent.cancel_prava_mandate(mid)

        # ── Adherence report ──
        elif choice == "a":
            print(agent.adherence_report())

        # ── Prava status ──
        elif choice == "p":
            ok = agent._init_prava()
            if ok:
                print(f"\n  [Prava] ENABLED — {'sandbox' if agent._prava_client.config.sandbox else 'production'}")
                print(f"  [Prava] User: {agent._prava_client.config.user_email}")
            else:
                print(f"\n  [Prava] NOT CONFIGURED — set PRAVA_SECRET_KEY in .env")

        elif choice == "x":
            print("Goodbye.")
            sys.exit(0)

        else:
            print("  Not a valid option.")


if __name__ == "__main__":
    main()
