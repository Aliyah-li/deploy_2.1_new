#!/usr/bin/env python3
"""
M20 Robot Monitor UI  —  library mode
Usage: pipe rl_deploy stdout into this script
  ros2 run m20_sdk_deploy rl_deploy 2>&1 | python3 monitor_ui.py
"""

import curses
import sys
import threading
import re
import time
from collections import deque

# ── shared state ─────────────────────────────────────────────────────────────
state = {
    "robot":       "---",
    "speed_mode":  "---",
    "direction":   "---",
    "phase":       "IDLE",
    "remain":      0.0,
    "vx":          0.0,
    "vy":          0.0,
    "yaw":         0.0,
    "spd":         0.0,
    "h":           0.40,
    "mix_hi":      "---",
    "robot_omega": [0.0, 0.0, 0.0],
    "robot_acc":   [0.0, 0.0, 0.0],
    "warnings":    deque(maxlen=5),
    "log":         deque(maxlen=200),
}
state_lock = threading.Lock()

# ── regex patterns ────────────────────────────────────────────────────────────
# [STATUS] robot=RLControl speed=LOW dir=x phase=IDLE remain=0.00 vx=0.0000 vy=0.0000 yaw=0.0000 spd=0.0000 h=0.400 mix_hi=y
RE_STATUS = re.compile(
    r'\[STATUS\] robot=(\S+) speed=(\S+) dir=(\S+) phase=(\S+) remain=([\d.]+)'
    r' vx=([\d.eE+\-]+) vy=([\d.eE+\-]+) yaw=([\d.eE+\-]+) spd=([\d.eE+\-]+) h=([\d.]+) mix_hi=(\S+)'
)
# [MODE] Joint Damping / Standing Up / RL Control
RE_MODE      = re.compile(r'\[MODE\]\s+(.+)')
# [DIRECTION] Switched to x direction
RE_DIRECTION = re.compile(r'\[DIRECTION\]\s+Switched to (\S+) direction')
# [MOVE] Direction: x, Velocity: 1.5
RE_MOVE      = re.compile(r'\[MOVE\] Direction:\s+(\S+),\s+Velocity:\s+([\d.eE+\-]+)')
# [ROBOT] omega=x y z acc=x y z
RE_ROBOT     = re.compile(
    r'\[ROBOT\] omega=([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)'
    r'\s+acc=([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)'
)
RE_WARN      = re.compile(r'(warning|error|Warning|Error|========)', re.IGNORECASE)

def parse_line(line: str):
    with state_lock:
        state["log"].append(line.rstrip())

        m = RE_STATUS.search(line)
        if m:
            state["robot"]      = m.group(1)
            state["speed_mode"] = m.group(2)
            state["direction"]  = m.group(3)
            state["phase"]      = m.group(4)
            state["remain"]     = float(m.group(5))
            state["vx"]         = float(m.group(6))
            state["vy"]         = float(m.group(7))
            state["yaw"]        = float(m.group(8))
            state["spd"]        = float(m.group(9))
            state["h"]          = float(m.group(10))
            state["mix_hi"]     = m.group(11)
            return  # [STATUS] is the canonical periodic update

        m = RE_DIRECTION.search(line)
        if m:
            state["direction"] = m.group(1)

        m = RE_MOVE.search(line)
        if m:
            state["direction"] = m.group(1)

        m = RE_ROBOT.search(line)
        if m:
            state["robot_omega"] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
            state["robot_acc"]   = [float(m.group(4)), float(m.group(5)), float(m.group(6))]

        if RE_WARN.search(line):
            state["warnings"].append(line.rstrip())

def stdin_reader():
    for line in sys.stdin:
        parse_line(line)

# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_box(win, title, y, x, h, w, color):
    try:
        sub = win.derwin(h, w, y, x)
        sub.attron(color)
        sub.box()
        sub.attroff(color)
        sub.addstr(0, 2, f" {title} ", curses.A_BOLD | color)
        return sub
    except curses.error:
        return None

def bar(val, lo, hi, width=20):
    frac = max(0.0, min(1.0, (val - lo) / (hi - lo) if hi != lo else 0))
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)

def speed_color(speed_mode, C_VAL, C_WARN, C_ERR):
    if speed_mode == "HIGH":   return C_ERR
    if speed_mode == "MIXED":  return C_WARN
    return C_VAL

# ── main UI ───────────────────────────────────────────────────────────────────
def ui_main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)
    curses.init_pair(2, curses.COLOR_GREEN,   -1)
    curses.init_pair(3, curses.COLOR_YELLOW,  -1)
    curses.init_pair(4, curses.COLOR_RED,     -1)
    curses.init_pair(5, curses.COLOR_WHITE,   -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)

    C_BOX  = curses.color_pair(1)
    C_VAL  = curses.color_pair(2)
    C_WARN = curses.color_pair(3)
    C_ERR  = curses.color_pair(4)
    C_LOG  = curses.color_pair(5)
    C_MODE = curses.color_pair(6)

    while True:
        stdscr.erase()
        H, W = stdscr.getmaxyx()

        title = " M20 Robot Monitor  [Ctrl-C] quit "
        stdscr.attron(C_BOX | curses.A_BOLD)
        stdscr.addstr(0, max(0, (W - len(title)) // 2), title)
        stdscr.attroff(C_BOX | curses.A_BOLD)

        with state_lock:
            robot       = state["robot"]
            speed_mode  = state["speed_mode"]
            direction   = state["direction"]
            phase       = state["phase"]
            remain      = state["remain"]
            vx          = state["vx"]
            vy          = state["vy"]
            yaw         = state["yaw"]
            spd         = state["spd"]
            h_cmd       = state["h"]
            mix_hi      = state["mix_hi"]
            robot_omega = state["robot_omega"]
            robot_acc   = state["robot_acc"]
            warnings    = list(state["warnings"])
            log_lines   = list(state["log"])

        C_SPD = speed_color(speed_mode, C_VAL, C_WARN, C_ERR)

        # ── Row 1: Robot state + Speed mode ──────────────────────────────────
        p = draw_box(stdscr, "ROBOT STATE", 1, 0, 3, 24, C_BOX)
        if p:
            p.addstr(1, 2, f"{robot:<18}", C_MODE | curses.A_BOLD)

        p = draw_box(stdscr, "SPEED MODE", 1, 24, 3, 22, C_BOX)
        if p:
            p.addstr(1, 2, f"{speed_mode:<16}", C_SPD | curses.A_BOLD)

        p = draw_box(stdscr, "DIRECTION", 1, 46, 3, 18, C_BOX)
        if p:
            p.addstr(1, 2, f"{direction:<12}", C_VAL | curses.A_BOLD)

        p = draw_box(stdscr, "MIX HI DIR", 1, 64, 3, 18, C_BOX)
        if p:
            p.addstr(1, 2, f"{mix_hi:<12}", C_WARN)

        # ── Row 2: Motion phase + countdown ──────────────────────────────────
        if phase == "IDLE":
            phase_label = "  IDLE"
            phase_color = curses.A_DIM
        else:
            phase_label = f"  {phase}  {remain:.2f}s left"
            phase_color = C_SPD | curses.A_BOLD

        p = draw_box(stdscr, "MOTION PHASE", 4, 0, 3, 46, C_BOX)
        if p:
            try:
                p.addstr(1, 2, f"{phase_label:<40}", phase_color)
            except curses.error:
                pass

        # ── Row 3: Velocities ─────────────────────────────────────────────────
        BAR_W = 18
        p = draw_box(stdscr, "COMMAND VELOCITY", 7, 0, 7, 64, C_BOX)
        if p:
            p.addstr(1, 2, "Vx  :", C_VAL)
            p.addstr(1, 8, f"{vx:+.4f}  [{bar(vx,  -3, 3, BAR_W)}]", C_VAL)
            p.addstr(2, 2, "Vy  :", C_VAL)
            p.addstr(2, 8, f"{vy:+.4f}  [{bar(vy,  -3, 3, BAR_W)}]", C_VAL)
            p.addstr(3, 2, "Yaw :", C_VAL)
            p.addstr(3, 8, f"{yaw:+.4f}  [{bar(yaw, -1, 1, BAR_W)}]", C_VAL)
            p.addstr(4, 2, "Spd :", C_VAL)
            p.addstr(4, 8, f"{spd:+.4f}  [{bar(spd,  0, 3, BAR_W)}]", C_SPD)
            p.addstr(5, 2, f"H    : {h_cmd:.3f} m", C_VAL | curses.A_BOLD)
            p.addstr(5, 27, "[height]", curses.A_DIM)
            p.addstr(5, 40, "Keys: [Q]=go  [A/D]=yaw  [W/S]=height  [T]=mode  [1/2/3]=dir", curses.A_DIM)

        # ── Row 4: Robot IMU ──────────────────────────────────────────────────
        p = draw_box(stdscr, "ROBOT IMU", 13, 0, 5, 64, C_BOX)
        if p:
            p.addstr(1, 2, "omega:", C_VAL)
            p.addstr(1, 9, f"x={robot_omega[0]:+.3f}  y={robot_omega[1]:+.3f}  z={robot_omega[2]:+.3f}", C_VAL)
            p.addstr(2, 2, "acc  :", C_VAL)
            p.addstr(2, 9, f"x={robot_acc[0]:+.3f}  y={robot_acc[1]:+.3f}  z={robot_acc[2]:+.3f}", C_VAL)

        # ── Warnings ──────────────────────────────────────────────────────────
        warn_h = 5
        p = draw_box(stdscr, "WARNINGS / ERRORS", 18, 0, warn_h, min(W, 80), C_BOX)
        if p:
            for i, w in enumerate(warnings[-(warn_h - 2):]):
                color = C_ERR if re.search(r'error', w, re.I) else C_WARN
                try:
                    p.addstr(i + 1, 2, w[:min(W, 80) - 4], color)
                except curses.error:
                    pass

        # ── Log ───────────────────────────────────────────────────────────────
        log_y = 23
        log_h = max(4, H - log_y - 1)
        p = draw_box(stdscr, "LOG", log_y, 0, log_h, min(W, 120), C_BOX)
        if p:
            for i, line in enumerate(log_lines[-(log_h - 2):]):
                try:
                    p.addstr(i + 1, 2, line[:min(W, 120) - 4], C_LOG)
                except curses.error:
                    pass

        stdscr.refresh()
        time.sleep(0.025)

if __name__ == "__main__":
    t = threading.Thread(target=stdin_reader, daemon=True)
    t.start()
    curses.wrapper(ui_main)
