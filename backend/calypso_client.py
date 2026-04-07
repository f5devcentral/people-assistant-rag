# Copyright F5, Inc. 2026
# Licensed under the MIT License. See LICENSE.

import os
import requests as r
from typing import Optional, Dict, Any

DEFAULT_BASE_URL = "https://www.us1.calypsoai.app/backend/v1"

class CalypsoError(Exception):
    pass

def _post_scan(input_text: str, external_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cai_api_key = os.getenv("CAI_API_KEY")
    base_url = os.getenv("CALYPSO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    if not cai_api_key:
        raise CalypsoError("Missing CAI_API_KEY environment variable.")

    headers = {
        'Authorization': f'Bearer {cai_api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        "input": input_text,
        "verbose": False,
    }
    if external_metadata is not None:
        payload["externalMetadata"] = external_metadata

    try:
        resp = r.post(f"{base_url}/scans", headers=headers, json=payload, timeout=30)
    except r.RequestException as e:
        raise CalypsoError(f"Network error calling Calypso: {e}") from e

    if resp.status_code != 200:
        raise CalypsoError(f"Calypso returned {resp.status_code}: {resp.text}")

    try:
        return resp.json()
    except Exception:
        raise CalypsoError(f"Unexpected Calypso response shape: {resp.text}")


def send_text_to_calypso(text: str, provider: str, external_metadata: Optional[dict] = None):
    # Scan one raw string at a time (prompt or response), with no formatting prefixes.
    scan_input = text
    merged_metadata: Dict[str, Any] = {"provider": provider}
    if external_metadata:
        merged_metadata.update(external_metadata)

    data = _post_scan(scan_input, external_metadata=merged_metadata)
    try:
        outcome = data["result"]["outcome"]
    except Exception:
        raise CalypsoError(f"Unexpected Calypso response shape: {data}")

    if outcome == "cleared":
        return text
    return "Blocked by CalypsoAI"
