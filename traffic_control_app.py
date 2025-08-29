import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict, deque

st.set_page_config(layout="wide")
st.title("🚦 Real-Time AI Traffic Control (Stable)")

model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = [2, 3, 5, 7]
YOLO_CONF = 0.35
YOLO_IOU = 0.5

lanes = ["North", "East", "West", "South"]
video_paths = {
    "North": r"D:\Downloads 2\ai_agent_TRAFFIC\vid1.mp4",
    "East":  r"D:\Downloads 2\ai_agent_TRAFFIC\vid2.mp4",
    "West":  r"D:\Downloads 2\ai_agent_TRAFFIC\vid3.mp4",
    "South": r"D:\Downloads 2\ai_agent_TRAFFIC\vid4.mp4",
}

LANE_ROIS = {
    "North": ((0, 0), (640, 480)),
    "East":  ((0, 0), (640, 480)),
    "West":  ((0, 0), (640, 480)),
    "South": ((0, 0), (640, 480)),
}

EMA_ALPHA = 0.3
ema_counts = defaultdict(float)
history = {ln: deque(maxlen=10) for ln in lanes}

MIN_GREEN   = 10
MAX_GREEN   = 45
MAX_WAIT    = 45
HYSTERESIS  = 0.15
REPEAT_PENALTY = 0.90
YELLOW_TIME   = 3
ALL_RED_TIME  = 1

def in_rect(cx, cy, rect):
    (x1, y1), (x2, y2) = rect
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def get_vehicle_count_and_frame(frame, lane_name):
    frame = cv2.resize(frame, (640, 480))
    results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU, classes=VEHICLE_CLASSES, verbose=False)[0]
    count = 0
    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if in_rect(cx, cy, LANE_ROIS[lane_name]):
            count += 1
    annotated = results.plot()
    (rx1, ry1), (rx2, ry2) = LANE_ROIS[lane_name]
    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
    cv2.putText(annotated, lane_name, (rx1 + 8, max(18, ry1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    ema_counts[lane_name] = EMA_ALPHA * count + (1 - EMA_ALPHA) * ema_counts[lane_name]
    history[lane_name].append(ema_counts[lane_name])
    return int(round(ema_counts[lane_name])), annotated

def lane_score(lane, now, smoothed_counts, last_green_time, current_lane):
    density = smoothed_counts.get(lane, 0)
    wait = now - last_green_time[lane]
    if len(history[lane]) >= 2:
        arr = (history[lane][-1] - history[lane][0]) / max(1, len(history[lane]) - 1)
        arrivals = max(0.0, arr)
    else:
        arrivals = 0.0
    score = 1.5 * density + 0.5 * wait + 2.0 * arrivals
    if lane == current_lane:
        score *= REPEAT_PENALTY
    return score

def pick_lane(smoothed_counts, last_green_time, current_lane, now):
    overdue = [ln for ln in lanes if (now - last_green_time[ln]) >= MAX_WAIT]
    if overdue:
        return max(overdue, key=lambda ln: now - last_green_time[ln])
    scores = {ln: lane_score(ln, now, smoothed_counts, last_green_time, current_lane) for ln in lanes}
    best_val = max(scores.values())
    cands = [ln for ln, s in scores.items() if abs(s - best_val) < 1e-6]
    cands.sort(key=lambda ln: (-(now - last_green_time[ln]), ln))
    return cands[0]

def phase_duration_for(lane, smoothed_counts):
    density = smoothed_counts.get(lane, 0)
    dur = 8 + 2 * int(max(0.0, density) ** 0.8)
    return int(max(MIN_GREEN, min(MAX_GREEN, dur)))

def main():
    cols = st.columns(2)
    frame_placeholders = {
        "North": cols[0].empty(),
        "East":  cols[1].empty(),
        "West":  cols[0].empty(),
        "South": cols[1].empty(),
    }
    stats_placeholder = st.empty()
    signal_placeholder = st.empty()

    caps = {ln: cv2.VideoCapture(video_paths[ln]) for ln in lanes}
    start = time.time()
    last_green_time = {ln: start for ln in lanes}

    TARGET_FPS = 15
    green_lane = None
    phase = "ALL_RED"
    phase_start = time.time()
    green_time = MIN_GREEN
    next_lane_queued = None

    while True:
        loop_start = time.time()
        counts_dict = {}

        for ln in lanes:
            cap = caps[ln]
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    continue
            c, disp = get_vehicle_count_and_frame(frame, ln)
            counts_dict[ln] = c
            frame_placeholders[ln].image(disp, channels="BGR", use_container_width=True)

        stats_placeholder.write(f"**Vehicle Counts (EMA):** {counts_dict}")
        now = time.time()

        def time_in_phase():
            return time.time() - phase_start

        def pick_next_lane():
            overdue = [ln for ln in lanes if (now - last_green_time[ln]) >= MAX_WAIT]
            if overdue:
                return max(overdue, key=lambda ln: now - last_green_time[ln])
            tmp = {ln: lane_score(ln, now, counts_dict, last_green_time, green_lane) for ln in lanes}
            best = max(tmp, key=tmp.get)
            if green_lane is not None and phase == "GREEN":
                if tmp[best] <= (1 + HYSTERESIS) * tmp.get(green_lane, 0.0):
                    return green_lane
            return best

        if phase == "GREEN":
            if time_in_phase() >= MIN_GREEN:
                overdue_exists = any((now - last_green_time[ln]) >= MAX_WAIT for ln in lanes)
                tmp_scores = {ln: lane_score(ln, now, counts_dict, last_green_time, green_lane) for ln in lanes}
                best_alt = max(tmp_scores, key=tmp_scores.get)
                big_gain = (best_alt != green_lane) and (tmp_scores[best_alt] > (1 + HYSTERESIS) * tmp_scores.get(green_lane, 0.0))
                if overdue_exists or big_gain or time_in_phase() >= green_time:
                    next_lane_queued = pick_next_lane()
                    phase = "YELLOW"
                    phase_start = time.time()

        elif phase == "YELLOW":
            if time_in_phase() >= YELLOW_TIME:
                phase = "ALL_RED"
                phase_start = time.time()

        elif phase == "ALL_RED":
            if time_in_phase() >= ALL_RED_TIME:
                if next_lane_queued is None:
                    green_lane = pick_next_lane()
                else:
                    green_lane = next_lane_queued
                    next_lane_queued = None
                green_time = phase_duration_for(green_lane, counts_dict)
                last_green_time[green_lane] = time.time()
                phase = "GREEN"
                phase_start = time.time()

        if phase == "GREEN":
            remaining = max(0, int(round(green_time - time_in_phase())))
            signal_placeholder.success(f"🟢 GREEN: {green_lane} • {remaining}s left")
        elif phase == "YELLOW":
            remaining = max(0, YELLOW_TIME - int(time_in_phase()))
            signal_placeholder.warning(f"🟡 YELLOW (clearing) • {remaining}s")
        else:
            remaining = max(0, ALL_RED_TIME - int(time_in_phase()))
            signal_placeholder.info(f"🔴 ALL-RED (interlock) • {remaining}s")

        elapsed_loop = time.time() - loop_start
        delay = max(0.0, (1.0 / TARGET_FPS) - elapsed_loop)
        time.sleep(delay)

    for cap in caps.values():
        cap.release()

if __name__ == "__main__":
    main()
