#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T-800 Terminal TUI
==================
Live telemetry dashboard for the T-800 brain.
Connects to the embedded SocketIO server in t800_brain_v2.py.

Usage:
    python3 t800_tui.py                         # Pi running locally
    python3 t800_tui.py --host 192.168.1.42     # Connect over network
    python3 t800_tui.py --host pi.local --port 5000

Requirements (install on the machine running THIS script):
    pip3 install rich python-socketio[client] --break-system-packages
"""

import io
import os
import sys

# Force UTF-8 output so Rich box-drawing characters render correctly
# even on terminals that default to ASCII/Latin-1
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import threading
import time
from collections import deque

import socketio
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich import box

# ── Theme ────────────────────────────────────────────────────────
RED       = "bold red"
DIM_RED   = "red"
HEADER    = "bold bright_red on black"
BORDER    = "red"
MAX_LOG   = 30


# ── Shared state (updated from SocketIO callbacks) ────────────────
class T800State:
    def __init__(self):
        self._lock = threading.Lock()
        self.connected  = False
        self.state      = "—"
        self.identity   = "—"
        self.emotion    = "—"
        self.distance   = "—"
        self.present    = False
        self.heard      = "—"
        self.said       = "—"
        self.standby    = False
        self.pan        = 0.0
        self.tilt       = 0.0
        self.agent_type = "quick"
        self.personality= "terminator"
        self.log        = deque(maxlen=MAX_LOG)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def add_log(self, line):
        with self._lock:
            self.log.append(line)

    def snapshot(self):
        with self._lock:
            return {
                "connected": self.connected,
                "state":     self.state,
                "identity":  self.identity,
                "emotion":   self.emotion,
                "distance":  self.distance,
                "present":   self.present,
                "heard":     self.heard,
                "said":      self.said,
                "standby":   self.standby,
                "pan":        self.pan,
                "tilt":       self.tilt,
                "agent_type": self.agent_type,
                "personality":self.personality,
                "log":        list(self.log),
            }


# ── Layout renderer ───────────────────────────────────────────────
def make_layout(snap):
    connected = snap["connected"]
    conn_txt  = Text("● CONNECTED", style="bold green") if connected \
                else Text("○ DISCONNECTED", style="bold red dim")

    # ── Top bar ──────────────────────────────────────────────────
    header = Text()
    header.append("  T-800 CYBERDYNE SYSTEMS — NEURAL NET TELEMETRY  ",
                  style=HEADER)
    header.append("  ")
    header.append_text(conn_txt)
    agent_label = f"  [ {snap.get('agent_type','?').upper()} / {snap.get('personality','?')} ]"
    header.append(agent_label, style="bold cyan")
    if snap.get("standby"):
        header.append("  [ STANDBY ]", style="bold yellow")

    # ── State panel ──────────────────────────────────────────────
    state_color = {
        "IDLE":         "bright_black",
        "DETECTED":     "yellow",
        "IDENTIFYING":  "bright_yellow",
        "GREETING":     "bright_cyan",
        "LISTENING":    "bright_green",
        "PROCESSING":   "bright_magenta",
        "SPEAKING":     "bright_red",
    }.get(snap["state"], "white")

    state_txt = Text(snap["state"], style=f"bold {state_color}")
    state_panel = Panel(state_txt, title="[red]STATE[/]",
                        border_style=BORDER, box=box.HEAVY_HEAD)

    # ── Identity panel ───────────────────────────────────────────
    id_txt = Text()
    id_txt.append(snap["identity"] or "—", style=RED)
    id_txt.append("\n")
    id_txt.append(snap["emotion"] or "—",  style=DIM_RED)
    id_panel = Panel(id_txt, title="[red]IDENTITY / EMOTION[/]",
                     border_style=BORDER, box=box.HEAVY_HEAD)

    # ── LiDAR panel ──────────────────────────────────────────────
    lidar_txt = Text()
    if snap["present"]:
        lidar_txt.append(f"{snap['distance']} cm", style=RED)
        lidar_txt.append("\nTARGET ACQUIRED", style="bold red")
    else:
        lidar_txt.append("— cm", style="dim red")
        lidar_txt.append("\nNO TARGET", style="dim red")
    lidar_panel = Panel(lidar_txt, title="[red]LIDAR[/]",
                        border_style=BORDER, box=box.HEAVY_HEAD)

    # ── Servo panel ──────────────────────────────────────────────
    pan_deg  = (snap["pan"]  + 1.0) * 90.0
    tilt_deg = (snap["tilt"] + 1.0) * 90.0
    servo_txt = Text()
    servo_txt.append(f"PAN  {snap['pan']:+.2f}  ({pan_deg:.0f}\u00b0)\n", style=RED)
    servo_txt.append(f"TILT {snap['tilt']:+.2f}  ({tilt_deg:.0f}\u00b0)",  style=DIM_RED)
    servo_panel = Panel(servo_txt, title="[red]SERVO[/]",
                        border_style=BORDER, box=box.HEAVY_HEAD)

    # ── Speech panels ─────────────────────────────────────────────
    heard_panel = Panel(
        Text(snap["heard"] or "—", style="bright_red"),
        title="[red]HEARD[/]", border_style=BORDER, box=box.HEAVY_HEAD
    )
    said_panel = Panel(
        Text(snap["said"] or "—", style="bright_red"),
        title="[red]SAID[/]", border_style=BORDER, box=box.HEAVY_HEAD
    )

    # ── Log panel ────────────────────────────────────────────────
    log_lines = snap["log"]
    log_txt = Text()
    for line in log_lines:
        log_txt.append(line + "\n", style="bright_white")
    log_panel = Panel(log_txt, title="[red]LOG[/]",
                      border_style=BORDER, box=box.HEAVY_HEAD)

    # ── Compose layout ───────────────────────────────────────────
    layout = Layout()
    layout.split_column(
        Layout(header, name="header",  size=1),
        Layout(name="main"),
        Layout(log_panel, name="log", size=MAX_LOG // 2 + 4),
    )
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(state_panel,  name="state",   size=5),
        Layout(id_panel,     name="identity", size=7),
        Layout(lidar_panel,  name="lidar",   size=6),
        Layout(servo_panel,  name="servo",   size=5),
    )
    layout["right"].split_column(
        Layout(heard_panel, name="heard", ratio=1),
        Layout(said_panel,  name="said",  ratio=1),
    )
    return layout


# ── SocketIO client ───────────────────────────────────────────────
def run_client(host, port, t8state):
    url = f"http://{host}:{port}"
    sio = socketio.Client(reconnection=True, reconnection_delay=3,
                          reconnection_attempts=0)

    @sio.event
    def connect():
        t8state.update(connected=True)
        t8state.add_log(f"[TUI] Connected to {url}")

    @sio.event
    def disconnect():
        t8state.update(connected=False)
        t8state.add_log("[TUI] Disconnected — retrying...")

    @sio.on("state")
    def on_state(data):
        t8state.update(state=data.get("new", "—"))

    @sio.on("face")
    def on_face(data):
        t8state.update(
            identity=data.get("name", "—") or "—",
            emotion =data.get("emotion", "—") or "—",
        )

    @sio.on("sensor")
    def on_sensor(data):
        t8state.update(
            distance=str(data.get("distance", "—")),
            present =bool(data.get("present", False)),
        )

    @sio.on("speech_in")
    def on_speech_in(data):
        t8state.update(heard=data.get("text", "—"))

    @sio.on("speech_out")
    def on_speech_out(data):
        t8state.update(said=data.get("text", "—"))

    @sio.on("standby")
    def on_standby(data):
        t8state.update(standby=bool(data.get("active", False)))

    @sio.on("servo")
    def on_servo(data):
        t8state.update(
            pan =float(data.get("pan",  0.0)),
            tilt=float(data.get("tilt", 0.0)),
        )

    @sio.on("profile")
    def on_profile(data):
        t8state.update(
            agent_type  = data.get("agent", "quick"),
            personality = data.get("personality", "terminator"),
        )

    @sio.on("log")
    def on_log(data):
        line = data.get("line", "")
        if line:
            t8state.add_log(line)

    while True:
        try:
            sio.connect(url, transports=["websocket", "polling"])
            sio.wait()
        except Exception as e:
            t8state.update(connected=False)
            t8state.add_log(f"[TUI] Connection error: {e}")
            time.sleep(3)


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="T-800 Terminal TUI — connects to t800_brain_v2 dashboard"
    )
    parser.add_argument("--host", default="localhost",
                        help="Hostname/IP of the Pi running t800_brain_v2.py")
    parser.add_argument("--port", type=int, default=5000,
                        help="Dashboard port (default: 5000)")
    args = parser.parse_args()

    t8state = T800State()
    t8state.add_log(f"[TUI] Connecting to http://{args.host}:{args.port} ...")

    # SocketIO client runs in background thread
    client_thread = threading.Thread(
        target=run_client, args=(args.host, args.port, t8state), daemon=True
    )
    client_thread.start()

    console = Console()
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            snap = t8state.snapshot()
            live.update(make_layout(snap))
            time.sleep(0.25)


if __name__ == "__main__":
    main()
