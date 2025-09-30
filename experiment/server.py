from flask import Flask, request, jsonify
import csv
import os
import time
from pupil_labs.realtime_api.simple import discover_one_device

app = Flask(__name__)

device = None
current_recording_start_ns = None

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global device, current_recording_id, current_recording_start_ns
    try:
        device = discover_one_device()
        recording_id = device.recording_start()
        current_recording_start_ns = time.time_ns()
        return jsonify({
            "status": "started",
            "server_start_ns": current_recording_start_ns
        })
    except Exception as e:
        # fallback: return a local id and server_start_ns anyway
        current_recording_id = f"local_{int(time.time())}"
        current_recording_start_ns = time.time_ns()
        return jsonify({
            "status": "error_starting_device",
            "message": str(e),
            "server_start_ns": current_recording_start_ns
        }), 500

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