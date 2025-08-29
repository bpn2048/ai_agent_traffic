import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from collections import defaultdict
import time
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")  # <-- MUST be first Streamlit command

# Auto-refresh every 5000 ms (5 seconds)
st_autorefresh(interval=5000, limit=None, key="refresh")

st.title("🚦 AI Traffic Control Agent with Video (OpenRouter)")

api_key = st.text_input("Enter your OpenRouter API Key:", type="password")
model_choice = st.selectbox("Choose Model:", ["openai/gpt-4o-mini", "anthropic/claude-3.5", "meta-llama/llama-3-70b-instruct"])

model = YOLO("yolov8n.pt")
lanes = ["North", "East", "West", "South"]

video_paths = {
    "North": r"D:\Downloads 2\ai_agent_TRAFFIC\vid1.mp4",
    "East":  r"D:\Downloads 2\ai_agent_TRAFFIC\vid2.mp4",
    "West":  r"D:\Downloads 2\ai_agent_TRAFFIC\vid3.mp4",
    "South": r"D:\Downloads 2\ai_agent_TRAFFIC\vid4.mp4",
}

EMA_ALPHA = 0.3
if "ema_counts" not in st.session_state:
    st.session_state.ema_counts = defaultdict(float)
if "lane_status" not in st.session_state:
    st.session_state.lane_status = {ln: "RED" for ln in lanes}
if "last_decision_time" not in st.session_state:
    st.session_state.last_decision_time = 0
if "caps" not in st.session_state:
    st.session_state.caps = {ln: cv2.VideoCapture(video_paths[ln]) for ln in lanes}
if "frame_pos" not in st.session_state:
    st.session_state.frame_pos = {ln: 0 for ln in lanes}

LANE_ROIS = {ln: ((0, 0), (640, 480)) for ln in lanes}

cols = st.columns(2)
frame_placeholders = {
    "North": cols[0].empty(),
    "East":  cols[1].empty(),
    "West":  cols[0].empty(),
    "South": cols[1].empty(),
}
stats_placeholder = st.empty()
decision_placeholder = st.empty()

def in_rect(cx, cy, rect):
    (x1, y1), (x2, y2) = rect
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def draw_signal_overlay(img, lane):
    status = st.session_state.lane_status.get(lane, "RED")
    color = (0, 200, 0) if status == "GREEN" else (0, 0, 255)
    (x1, y1), (x2, y2) = LANE_ROIS[lane]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{lane} • {status}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    bx1, by1 = x1 + 8, max(8, y1 + 8)
    cv2.rectangle(img, (bx1 - 6, by1 + 6), (bx1 + tw + 6, by1 + th + 12), (0, 0, 0), -1)
    cv2.putText(img, label, (bx1, by1 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img

def get_vehicle_counts_and_show():
    counts = {}
    for ln in lanes:
        cap = st.session_state.caps[ln]
        # Set frame position and read next frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos[ln])
        ret, frame = cap.read()
        if not ret:
            # Loop video if end reached
            st.session_state.frame_pos[ln] = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                continue
        frame = cv2.resize(frame, (640, 480))
        results = model(frame, conf=0.35, iou=0.5, classes=[2, 3, 5, 7])[0]
        count = 0
        for r in results.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if in_rect(cx, cy, LANE_ROIS[ln]):
                count += 1
        annotated = results.plot()
        annotated = draw_signal_overlay(annotated, ln)
        st.session_state.ema_counts[ln] = EMA_ALPHA * count + (1 - EMA_ALPHA) * st.session_state.ema_counts[ln]
        counts[ln] = int(round(st.session_state.ema_counts[ln]))
        frame_placeholders[ln].image(annotated, channels="BGR", use_container_width=True)
        # Advance frame position for next run
        st.session_state.frame_pos[ln] += 1
    return counts

def get_vehicle_counts_batch(batch_size=20):
    counts = {}
    for ln in lanes:
        cap = st.session_state.caps[ln]
        lane_counts = []
        for _ in range(batch_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_pos[ln])
            ret, frame = cap.read()
            if not ret:
                st.session_state.frame_pos[ln] = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    continue
            frame = cv2.resize(frame, (640, 480))
            results = model(frame, conf=0.35, iou=0.5, classes=[2, 3, 5, 7])[0]
            count = 0
            for r in results.boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if in_rect(cx, cy, LANE_ROIS[ln]):
                    count += 1
            lane_counts.append(count)
            st.session_state.frame_pos[ln] += 1
        # Use EMA of batch counts for decision
        if lane_counts:
            batch_ema = EMA_ALPHA * np.mean(lane_counts) + (1 - EMA_ALPHA) * st.session_state.ema_counts[ln]
        else:
            batch_ema = st.session_state.ema_counts[ln]
        st.session_state.ema_counts[ln] = batch_ema
        counts[ln] = int(round(batch_ema))
        # Show last frame in batch
        if lane_counts:
            annotated = results.plot()
            annotated = draw_signal_overlay(annotated, ln)
            frame_placeholders[ln].image(annotated, channels="BGR", use_container_width=True)
    return counts

def get_vehicle_counts_tool(_):
    # Use batch of 20 frames and EMA for decision
    return get_vehicle_counts_batch(batch_size=20)

def change_signal(lane):
    for ln in lanes:
        st.session_state.lane_status[ln] = "RED"
    st.session_state.lane_status[lane] = "GREEN"
    return f"Signal changed: {lane} is GREEN, others RED."

def show_status(_):
    return st.session_state.lane_status

tools = [
    Tool(name="GetVehicleCounts", func=get_vehicle_counts_tool, description="Get the estimated vehicle count in each lane."),
    Tool(name="ChangeSignal", func=change_signal, description="Change the traffic light to green for a chosen lane."),
    Tool(name="ShowStatus", func=show_status, description="See the current traffic light status."),
]

TARGET_FPS = 15  # Set your comfortable frame rate here (e.g., 10-15 FPS)
frame_delay = 1.0 / TARGET_FPS

if api_key:
    llm = ChatOpenAI(
        model=model_choice,
        temperature=0,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    counts = get_vehicle_counts_batch(batch_size=20)
    stats_placeholder.write(f"**Vehicle Counts (EMA, 20-frame batch):** {counts}")

    now = time.time()
    if now - st.session_state.last_decision_time > 30:
        # The agent will use EMA counts for decision
        decision = agent.run("Use GetVehicleCounts, then select exactly one lane with ChangeSignal to minimize total waiting time using EMA vehicle counts.")
        decision_placeholder.success(decision)
        st.session_state.last_decision_time = now

    st.write("🚦 Current Signals:", st.session_state.lane_status)

    if st.button("Next Batch (20 Frames)"):
        pass  # Reruns and processes next batch

else:
    st.warning("Please enter your OpenRouter API key to start the AI agent.")
