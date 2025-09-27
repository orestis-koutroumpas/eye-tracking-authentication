# server.py
from flask import Flask, request, jsonify
import csv
import os
import time
from pupil_labs.realtime_api.simple import discover_one_device

app = Flask(__name__)

device = None
current_recording_id = None
current_recording_start_ns = None

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global device, current_recording_id, current_recording_start_ns
    try:
        device = discover_one_device()
        current_recording_id = device.recording_start()
        # record server time (ns) as close as possible after starting the recording
        current_recording_start_ns = time.time_ns()
        return jsonify({
            "status": "started",
            "recording_id": current_recording_id,
            "server_start_ns": current_recording_start_ns
        })
    except Exception as e:
        # fallback: return a local id and server_start_ns anyway
        current_recording_id = f"local_{int(time.time())}"
        current_recording_start_ns = time.time_ns()
        return jsonify({
            "status": "error_starting_device",
            "message": str(e),
            "recording_id": current_recording_id,
            "server_start_ns": current_recording_start_ns
        }), 500

@app.route("/save_keystrokes", methods=["POST"])
def save_keystrokes():
    """
    Optional endpoint: saves keystrokes to server if client requests save_on_server=True.
    Body: { "logs": [...], "save_on_server": bool }
    """
    global current_recording_id
    data = request.get_json(force=True)
    logs = data.get("logs", [])
    save_on_server = data.get("save_on_server", False)

    if not logs:
        return jsonify({"status": "no_logs", "rows": 0})

    if save_on_server:
        filename = f"keystrokes_{current_recording_id or int(time.time())}.csv"
        file_exists = os.path.isfile(filename)
        try:
            with open(filename, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["recording id", "timestamp [ns]", "name", "type"])
                for log in logs:
                    writer.writerow([log.get("recording_id"), log.get("timestamp"), log.get("name"), log.get("type")])
            return jsonify({"status": "saved", "file": filename, "rows": len(logs)})
        except Exception as e:
            return jsonify({"status": "error_writing", "message": str(e)}), 500
    else:
        return jsonify({"status": "received", "rows": len(logs)})

@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    global device
    try:
        if device:
            device.recording_stop_and_save()
            device.close()
            device = None
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"status": "error_stopping", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)