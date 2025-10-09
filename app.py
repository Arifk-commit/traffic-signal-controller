import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# Load model
model = YOLO("best.pt")
class_names = model.names

st.set_page_config(page_title="Vehicle Detection - 4 Lane System", layout="wide")
st.title("Vehicle Detection - 4 Lane Upload System")

# Output file for simulation
DETECTION_FILE = "detected_vehicles.json"

# Initialize session state
if "all_detections" not in st.session_state:
    st.session_state.all_detections = {
        'right': [],
        'down': [],
        'left': [],
        'up': []
    }

if "uploader_keys" not in st.session_state:
    st.session_state.uploader_keys = {
        'right': str(np.random.randint(0, 1000000)),
        'down': str(np.random.randint(0, 1000000)),
        'left': str(np.random.randint(0, 1000000)),
        'up': str(np.random.randint(0, 1000000))
    }

# Lane mapping
directions = {
    'right': {'color': '🔴', 'emoji': '→'},
    'down': {'color': '🟢', 'emoji': '↓'},
    'left': {'color': '🔵', 'emoji': '←'},
    'up': {'color': '🟡', 'emoji': '↑'}
}

# Clear all button
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear All Detections", use_container_width=True):
        # Reset detections
        st.session_state.all_detections = {
            'right': [],
            'down': [],
            'left': [],
            'up': []
        }
        # Reset uploader keys (clears uploaded files)
        st.session_state.uploader_keys = {
            'right': str(np.random.randint(0, 1000000)),
            'down': str(np.random.randint(0, 1000000)),
            'left': str(np.random.randint(0, 1000000)),
            'up': str(np.random.randint(0, 1000000))
        }
        # Remove JSON file
        if os.path.exists(DETECTION_FILE):
            os.remove(DETECTION_FILE)
        st.rerun()

with col2:
    if st.button("Save & Send to Simulation", use_container_width=True):
        # Save detections to JSON file
        with open(DETECTION_FILE, 'w') as f:
            json.dump(st.session_state.all_detections, f, indent=2)
        st.success(f"✅ Detections saved! Run simulation.py to start traffic simulation")

st.divider()

# Create 4 columns for 4 lanes
cols = st.columns(4)

for idx, (direction, info) in enumerate(directions.items()):
    with cols[idx]:
        st.subheader(f"{info['emoji']} {direction.upper()}")
        
        # ✅ Allow multiple file uploads
        uploaded_files = st.file_uploader(
            f"Upload images for {direction} lane",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"{direction}_{st.session_state.uploader_keys[direction]}"
        )
        
        if uploaded_files:
            detection_data = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                img_np = np.array(image)

                # Run prediction
                results = model.predict(img_np, conf=0.5, verbose=False)
                annotated_img = results[0].plot()

                # Show annotated image
                st.image(annotated_img, caption=f"Detections in {direction}: {uploaded_file.name}", use_container_width=True)

                # Extract boxes
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for j, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = class_names[cls_id]
                        
                        detection_data.append({
                            'id': len(st.session_state.all_detections[direction]) + len(detection_data),
                            'class': cls_name,
                            'confidence': round(conf, 2),
                            'bbox': {
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2),
                                'y2': float(y2)
                            }
                        })
                else:
                    st.warning(f"No vehicles detected in {uploaded_file.name}.")

            st.session_state.all_detections[direction] = detection_data

            st.write(f"**Found {len(detection_data)} vehicle(s) in total:**")
            for det in detection_data:
                st.write(f"  • {det['class'].upper()} (Confidence: {det['confidence']})")

st.divider()

# Display all detections summary
st.subheader("📊 Detection Summary")
total_vehicles = sum(len(v) for v in st.session_state.all_detections.values())
st.metric("Total Vehicles Detected", total_vehicles)

# Show detections by lane
for direction, detections in st.session_state.all_detections.items():
    if detections:
        with st.expander(f"{direction.upper()} Lane - {len(detections)} vehicle(s)"):
            for det in detections:
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.write(f"**{det['class'].upper()}**")
                with col_b:
                    st.write(f"Conf: {det['confidence']}")
