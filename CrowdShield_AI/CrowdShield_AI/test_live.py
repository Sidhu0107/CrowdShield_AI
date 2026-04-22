import sys
import os
import threading
import time

project_root = os.path.abspath("/Users/shriya/Downloads/CrowdShield_AI/CrowdShield_AI")
training_dir = os.path.join(project_root, "training")
if project_root not in sys.path:
    sys.path.append(project_root)
if training_dir not in sys.path:
    sys.path.append(training_dir)

from backend.api.main import run_live_worker, state

def test():
    try:
        print("Starting worker...")
        run_live_worker(0)
    except Exception as e:
        print("Exception caught:", e)

t = threading.Thread(target=test)
t.start()

time.sleep(10)
print(f"live_running: {state.live_running}, live_status: {state.live_status}")
state.live_stop_event.set()
t.join()
print("Done")
