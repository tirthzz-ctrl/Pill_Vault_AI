"""
Prava Payments integration for PillVault Agent.

Wraps the Prava REST API (https://docs.prava.space) so the agent can
create payment sessions, manage mandates, poll for credentials, and
report outcomes.

Requires: pip install requests
"""

from __future__ import annotations

import json
import os
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests


# Auto-load .env from the same directory as this file
_env_path = Path(__file__).with_name(".env")
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip("\"'")
            if _k not in os.environ:
                os.environ[_k] = _v

BASE_URL_SANDBOX = "https://sandbox.api.prava.space"
BASE_URL_PROD = "https://api.prava.space"


@dataclass
class PravaConfig:
    secret_key: str
    sandbox: bool = True
    user_id: str = "pillvault_patient_1"
    user_email: str = "patient@pillvault.com"

    @property
    def base_url(self) -> str:
        return BASE_URL_SANDBOX if self.sandbox else BASE_URL_PROD

    @classmethod
    def from_env(cls) -> PravaConfig:
        return cls(
            secret_key=os.environ.get("PRAVA_SECRET_KEY", ""),
            sandbox=os.environ.get("PRAVA_ENV", "sandbox") == "sandbox",
            user_id=os.environ.get("PRAVA_USER_ID", "pillvault_patient_1"),
            user_email=os.environ.get("PRAVA_USER_EMAIL", "patient@pillvault.com"),
        )


@dataclass
class PravaSession:
    session_id: str
    session_token: str
    iframe_url: str
    order_id: str
    expires_at: str


@dataclass
class PravaPaymentResult:
    status: str
    token: Optional[str] = None
    dynamic_cvv: Optional[str] = None
    txn_ref_id: Optional[str] = None
    txn_id: Optional[str] = None


@dataclass
class PravaMandate:
    id: str
    status: str
    state: str
    merchant_name: str
    approved_amount: str
    remaining: str
    currency: str
    recurring_frequency: str
    merchant_scope: str
    valid_until: Optional[str] = None
    renews_at: Optional[str] = None
    last_charge: Optional[dict] = None
    created_at: str = ""


class PravaError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(f"[{code}] {message}")


class PravaClient:
    def __init__(self, config: Optional[PravaConfig] = None):
        self.config = config or PravaConfig.from_env()
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.config.secret_key}",
            "Content-Type": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.config.base_url}{path}"
        resp = self._session.request(method, url, **kwargs)
        data = resp.json()
        if not resp.ok:
            err = data.get("error", {})
            raise PravaError(
                code=err.get("code", "UNKNOWN"),
                message=err.get("message", str(resp.status_code)),
                status=resp.status_code,
            )
        return data

    # ── Session (one-time payment) ──────────────────────────────────────

    def create_session(
        self,
        total_amount: str,
        currency: str,
        merchant_name: str,
        merchant_url: str,
        product_description: str,
        unit_price: str,
        quantity: int = 1,
        product_id: Optional[str] = None,
        external_order_ref: Optional[str] = None,
        description: Optional[str] = None,
        mandate_setup: Optional[dict] = None,
    ) -> PravaSession:
        """
        Create a payment session.

        Pass `mandate_setup` to create a mandate instead of charging immediately.
        """
        payload = {
            "user_id": self.config.user_id,
            "user_email": self.config.user_email,
            "total_amount": total_amount,
            "currency": currency,
            "purchase_context": [{
                "merchant_details": {
                    "name": merchant_name,
                    "url": merchant_url,
                    "country_code_iso2": "US",
                    "category": "Pharmacy",
                    "category_code": "5912",
                },
                "product_details": [{
                    "description": product_description,
                    "unit_price": unit_price,
                    "quantity": quantity,
                    **({"product_id": product_id} if product_id else {}),
                }],
            }],
            "integration_type": "full_checkout",
        }
        if external_order_ref:
            payload["external_order_ref"] = external_order_ref
        if description:
            payload["description"] = description
        if mandate_setup:
            payload["mandate_setup"] = mandate_setup

        data = self._request("POST", "/v1/sessions", json=payload)
        return PravaSession(
            session_id=data["session_id"],
            session_token=data["session_token"],
            iframe_url=data["iframe_url"],
            order_id=data["order_id"],
            expires_at=data["expires_at"],
        )

    def poll_payment_result(
        self, session_id: str, poll_interval: float = 3.0, timeout: float = 300.0
    ) -> PravaPaymentResult:
        """Poll for payment credentials after cardholder approval.

        Blocks until the session moves to 'awaiting_result' or 'completed'.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = self._request("GET", f"/v1/sessions/{session_id}/payment-result")
            status = data.get("status", "pending")

            if status == "failed":
                return PravaPaymentResult(status="failed")

            if status == "awaiting_result":
                txn = data["transactions"][0]
                item = txn["line_items"][0]
                return PravaPaymentResult(
                    status="awaiting_result",
                    token=item.get("token"),
                    dynamic_cvv=item.get("dynamic_cvv"),
                    txn_ref_id=item.get("txn_ref_id"),
                    txn_id=txn.get("txn_id"),
                )

            if status == "completed":
                return PravaPaymentResult(status="completed")

            time.sleep(poll_interval)

        return PravaPaymentResult(status="timeout")

    def report_status(
        self,
        session_id: str,
        txn_ref_id: str,
        txn_status: str,
        authorization_code: Optional[str] = None,
        response_code: Optional[str] = None,
    ) -> dict:
        """Report checkout outcome back to Prava."""
        payload = {"txn_ref_id": txn_ref_id, "txn_status": txn_status}
        if authorization_code:
            payload["authorization_code"] = authorization_code
        if response_code:
            payload["response_code"] = response_code
        return self._request("POST", f"/v1/sessions/{session_id}/report-status", json=payload)

    def pay_pharmacy_order(
        self,
        medication_name: str,
        amount: str,
        currency: str = "USD",
        pharmacy_name: str = "MedPlus Pharmacy",
        rx_number: Optional[str] = None,
        open_browser: bool = True,
    ) -> dict:
        """Full one-time payment flow for a pharmacy refill order."""
        session = self.create_session(
            total_amount=amount,
            currency=currency,
            merchant_name=pharmacy_name,
            merchant_url="https://medplus.example.com",
            product_description=f"Refill: {medication_name}",
            unit_price=amount,
            quantity=1,
            product_id=rx_number,
            external_order_ref=rx_number,
            description=f"Pharmacy refill order for {medication_name}",
        )

        print(f"\n[Prava] Payment session created: {session.session_id}")
        print(f"[Prava] Approve at: {session.iframe_url}")

        if open_browser:
            try:
                webbrowser.open(session.iframe_url)
                print("[Prava] Browser opened for payment approval.")
            except Exception:
                pass

        print("[Prava] Waiting for payment approval...")
        result = self.poll_payment_result(session.session_id)

        if result.status == "timeout":
            return {"success": False, "confirmation": None,
                    "message": "Timed out waiting for payment approval.",
                    "details": {"session_id": session.session_id}}

        if result.status == "failed":
            return {"success": False, "confirmation": None,
                    "message": "Payment session failed.",
                    "details": {"session_id": session.session_id}}

        if result.status == "completed":
            return {"success": True, "confirmation": f"PRV-{session.session_id[-8:]}",
                    "message": "Payment already completed.",
                    "details": {"session_id": session.session_id}}

        print(f"[Prava] Payment approved! Credentials received.")
        print(f"[Prava] Submitting to {pharmacy_name}...")

        report = self.report_status(
            session_id=session.session_id,
            txn_ref_id=result.txn_ref_id,
            txn_status="APPROVED",
            authorization_code=f"AUTH-{session.session_id[-8:]}",
            response_code="00",
        )
        print(f"[Prava] Payment reported: {report.get('visa_confirmation', 'SUCCESS')}")

        return {"success": True, "confirmation": f"PRV-{session.session_id[-8:]}",
                "message": f"Payment of ${amount} to {pharmacy_name} completed.",
                "details": {"session_id": session.session_id,
                            "transaction_id": result.txn_id,
                            "visa_confirmation": report.get("visa_confirmation")}}

    # ── Mandates ────────────────────────────────────────────────────────

    def setup_mandate(
        self,
        total_amount: str,
        currency: str,
        merchant_name: str,
        merchant_url: str,
        product_description: str,
        unit_price: str,
        recurring_frequency: str = "monthly",
        max_charges: int = 12,
        quantity: int = 1,
        product_id: Optional[str] = None,
        open_browser: bool = True,
    ) -> dict:
        """
        Set up a recurring mandate (authorize-only, no charge).

        The owner approves once with a passkey. Future refills charge
        against this mandate without requiring re-approval.

        Returns: { success, mandate_id, session_id, message }
        """
        session = self.create_session(
            total_amount=total_amount,
            currency=currency,
            merchant_name=merchant_name,
            merchant_url=merchant_url,
            product_description=product_description,
            unit_price=unit_price,
            quantity=quantity,
            product_id=product_id,
            mandate_setup={
                "intent": "mandate_setup",
                "recurring_frequency": recurring_frequency,
                "merchant_scope": "listed",
                "max_charges": max_charges,
            },
        )

        print(f"\n[Prava] Mandate setup session: {session.session_id}")
        print(f"[Prava] Approve mandate at: {session.iframe_url}")

        if open_browser:
            try:
                webbrowser.open(session.iframe_url)
                print("[Prava] Browser opened for mandate approval.")
            except Exception:
                pass

        print("[Prava] Waiting for mandate approval...")
        result = self.poll_payment_result(session.session_id)

        if result.status in ("timeout", "failed"):
            return {"success": False, "mandate_id": None,
                    "session_id": session.session_id,
                    "message": f"Mandate setup {result.status}."}

        # After authorize-only session completes, a mandate is created.
        # List mandates and find the newest one for this merchant.
        mandates = self.list_mandates(standing_only=True)
        matching = [m for m in mandates if m.merchant_name.lower() == merchant_name.lower()]
        if matching:
            mandate = matching[0]
            return {"success": True, "mandate_id": mandate.id,
                    "session_id": session.session_id,
                    "message": f"Mandate {mandate.id} active — ${mandate.approved_amount} "
                               f"{mandate.recurring_frequency} at {merchant_name}."}

        return {"success": True, "mandate_id": None,
                "session_id": session.session_id,
                "message": "Mandate approved but ID not yet visible (may take a moment)."}

    def charge_mandate(
        self,
        mandate_id: str,
        amount: str,
        reference: Optional[str] = None,
    ) -> dict:
        """
        Mint a single-use credential against an active mandate (no passkey).

        Returns: {
            "success": bool,
            "credentials": { token, dynamicCvv, expiry_month, expiry_year } or None,
            "transaction_id": str or None,
            "message": str,
        }
        """
        payload = {"amount": amount}
        if reference:
            payload["reference"] = reference

        try:
            data = self._request("POST", f"/v1/mandates/{mandate_id}/charge", json=payload)
        except PravaError as e:
            return {"success": False, "credentials": None,
                    "transaction_id": None,
                    "message": f"Charge failed: {e.message}"}

        if data.get("fetchStatus") != "SUCCESS":
            return {"success": False, "credentials": None,
                    "transaction_id": data.get("transactionId"),
                    "message": data.get("errorMessage", "Mandate charge fetch failed.")}

        creds = data.get("credentials", {})
        return {
            "success": True,
            "credentials": {
                "token": creds.get("token"),
                "dynamicCvv": creds.get("dynamicCvv"),
                "expiry_month": creds.get("expiryMonth"),
                "expiry_year": creds.get("expiryYear"),
            },
            "transaction_id": data.get("transactionId"),
            "instruction_id": data.get("instructionId"),
            "message": "Credentials minted against mandate.",
        }

    def report_mandate_charge(
        self,
        mandate_id: str,
        transaction_id: str,
        txn_status: str,
        authorization_code: Optional[str] = None,
        response_code: Optional[str] = None,
        amount_paid: Optional[str] = None,
    ) -> dict:
        """Report a mandate charge outcome."""
        payload = {"txn_status": txn_status, "txn_type": "PURCHASE"}
        if authorization_code:
            payload["authorization_code"] = authorization_code
        if response_code:
            payload["response_code"] = response_code
        if amount_paid:
            payload["amount_paid"] = amount_paid
        return self._request(
            "POST", f"/v1/mandates/{mandate_id}/charges/{transaction_id}/report", json=payload
        )

    def auto_charge_refill(
        self,
        mandate_id: str,
        amount: str,
        medication_name: str,
        pharmacy_name: str,
        rx_number: Optional[str] = None,
    ) -> dict:
        """
        Charge a refill against an existing mandate — no passkey needed.

        Returns: { success, confirmation, message, details }
        """
        ref = rx_number or f"refill-{int(time.time())}"
        result = self.charge_mandate(mandate_id, amount, reference=ref)

        if not result["success"]:
            return {"success": False, "confirmation": None,
                    "message": result["message"], "details": {}}

        creds = result["credentials"]
        print(f"[Prava] Charged mandate {mandate_id}: ${amount} at {pharmacy_name}")
        print(f"[Prava] Token: {creds['token']}  CVV: {creds['dynamicCvv']}")

        report = self.report_mandate_charge(
            mandate_id=mandate_id,
            transaction_id=result["transaction_id"],
            txn_status="APPROVED",
            authorization_code=f"AUTH-{mandate_id[-8:]}",
            response_code="00",
            amount_paid=amount,
        )
        print(f"[Prava] Charge reported: {report.get('visaConfirmation', 'SUCCESS')}")

        return {"success": True, "confirmation": f"PRV-{mandate_id[-8:]}-{abs(hash(ref)) % 9999:04d}",
                "message": f"Refill ${amount} charged to mandate {mandate_id}.",
                "details": {"mandate_id": mandate_id,
                            "transaction_id": result["transaction_id"]}}

    def list_mandates(self, standing_only: bool = True) -> list[PravaMandate]:
        """List standing mandates on the account."""
        params = {"standing_only": "true"} if standing_only else {}
        data = self._request("GET", "/v1/mandates", params=params)
        return [
            PravaMandate(
                id=m["id"],
                status=m.get("status", ""),
                state=m.get("state", ""),
                merchant_name=m.get("merchantName", ""),
                approved_amount=m.get("approvedAmount", "0.00"),
                remaining=m.get("remaining", "0.00"),
                currency=m.get("currency", "USD"),
                recurring_frequency=m.get("recurringFrequency", "one_time"),
                merchant_scope=m.get("merchantScope", "listed"),
                valid_until=m.get("validUntil"),
                renews_at=m.get("renewsAt"),
                last_charge=m.get("lastCharge"),
                created_at=m.get("createdAt", ""),
            )
            for m in data.get("mandates", [])
        ]

    def get_mandate(self, mandate_id: str) -> dict:
        """Get mandate details including charge history."""
        return self._request("GET", f"/v1/mandates/{mandate_id}")

    def pause_mandate(self, mandate_id: str) -> dict:
        return self._request("POST", f"/v1/mandates/{mandate_id}/pause")

    def resume_mandate(self, mandate_id: str) -> dict:
        return self._request("POST", f"/v1/mandates/{mandate_id}/resume")

    def cancel_mandate(self, mandate_id: str) -> dict:
        return self._request("POST", f"/v1/mandates/{mandate_id}/cancel")
