import time
import json
import urllib.request
import sys
import os

URL = "http://127.0.0.1:8787/api/simulation"


def get_sim_state():
    try:
        with urllib.request.urlopen(URL, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        # print(f"Error fetching state: {e}")
        return None


def monitor():
    print("Monitoring simulation for 45 seconds...")
    start_time = time.time()

    last_particles = 0
    total_emitted = 0
    total_absorbed = 0

    while time.time() - start_time < 45:
        state = get_sim_state()
        if not state:
            print("Waiting for server...")
            time.sleep(2)
            continue

        dynamics = state.get("presence_dynamics", {})
        particles = dynamics.get("field_particles", [])
        resource_daimoi = dynamics.get("resource_daimoi", {})

        delivered = resource_daimoi.get("delivered_packets", 0)
        emitter_rows = resource_daimoi.get("emitter_rows", 0)
        total_transfer = resource_daimoi.get("total_transfer", 0.0)
        recipients = resource_daimoi.get("recipients", [])

        # Check specific presence wallets
        impacts = dynamics.get("presence_impacts", [])
        cpu_core = next((p for p in impacts if p["id"] == "presence.core.cpu"), {})
        cpu_wallet = cpu_core.get("resource_wallet", {}).get("cpu", 0.0)

        witness = next((p for p in impacts if p["id"] == "witness_thread"), {})
        witness_wallet = witness.get("resource_wallet", {}).get("cpu", 0.0)

        top_recipient = (
            recipients[0] if recipients else {"presence_id": "none", "credited": 0.0}
        )

        print(
            f"T+{time.time() - start_time:04.1f}s | Vol: {total_transfer:.4f} | Core: {cpu_wallet:.4f} | Witness: {witness_wallet:.4f} | Top: {top_recipient['presence_id']} ({top_recipient['credited']:.4f})"
        )

        time.sleep(1.0)


if __name__ == "__main__":
    monitor()
