#!/usr/bin/env python3
"""Start PillVault publicly accessible via ngrok tunnel."""
import subprocess
import sys
import time
import webbrowser

from pyngrok import ngrok, conf

conf.get_default().log_event_callback = lambda e: None

print("Starting PillVault server...")
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

time.sleep(3)

print("Opening ngrok tunnel...")
tunnel = ngrok.connect(8000, bind_tls=True)
public_url = tunnel.public_url
ngrok_url = "http://127.0.0.1:4040"

print(f"\n{'='*55}")
print(f"  PillVault is LIVE at: {public_url}")
print(f"  Ngrok dashboard:    {ngrok_url}")
print(f"{'='*55}")
print(f"\nShare the public URL above with anyone!")
print(f"Press Ctrl+C to stop both server and tunnel.\n")

try:
    webbrowser.open(public_url)
except Exception:
    pass

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")
    ngrok.disconnect(tunnel.public_url)
    ngrok.kill()
    server.terminate()
    server.wait()
    print("Done.")
