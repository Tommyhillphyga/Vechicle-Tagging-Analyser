
import streamlit as st
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from engine.schemas import Snapshot
from engine.vision import SentinelProcessor

# Configuration
st.set_page_config(page_title="SENTINEL - Forensic Vision", page_icon="🛡️", layout="wide")

# Optimized State Management
if 'logs' not in st.session_state: st.session_state.logs = []
if 'results' not in st.session_state: st.session_state.results = []
if 'entry_staged' not in st.session_state: st.session_state.entry_staged = []
if 'exit_staged' not in st.session_state: st.session_state.exit_staged = []

def add_log(msg):
    st.session_state.logs.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Sidebar UI
with st.sidebar:
    st.markdown("### 🛡️ SENTINEL v4.1")
    st.caption("Local Forensic Identity Matcher")
    st.divider()
    st.subheader("System Logs")
    log_area = st.empty()
    with log_area.container(height=300):
        for log in st.session_state.logs:
            st.markdown(f"<small>`{log}`</small>", unsafe_allow_stdio=True)
    
    if st.button("Reset Pipeline", type="secondary"):
        st.session_state.clear()
        st.rerun()

# Main Header
st.title("Forensic Analysis Dashboard")
st.markdown("Deep verification of vehicle and occupant identity across checkpoints.")

# 1. Data Ingestion
in1, in2 = st.columns(2)
with in1:
    e_files = st.file_uploader("Batch Entry Stream", accept_multiple_files=True, type=['jpg','png'])
    if e_files:
        st.session_state.entry_staged = [Snapshot(file_name=f.name, image_data=f.read()) for f in e_files]
        add_log(f"Staged {len(e_files)} entry frames.")

with in2:
    x_files = st.file_uploader("Batch Exit Stream", accept_multiple_files=True, type=['jpg','png'])
    if x_files:
        st.session_state.exit_staged = [Snapshot(file_name=f.name, image_data=f.read()) for f in x_files]
        add_log(f"Staged {len(x_files)} exit frames.")

# 2. Execution Engine
if st.button("🚀 INITIATE FORENSIC AUDIT", use_container_width=True, type="primary"):
    if not st.session_state.entry_staged or not st.session_state.exit_staged:
        st.error("Incomplete data streams. Both checkpoints required.")
    else:
        processor = SentinelProcessor()
        
        # Step A: Parallel Feature Extraction
        add_log("Starting parallel feature extraction...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            entries = list(executor.map(processor.process_image, st.session_state.entry_staged))
            exits = list(executor.map(processor.process_image, st.session_state.exit_staged))
        
        # Step B: Clustering
        add_log("Clustering detection groups...")
        en_clusters = processor.cluster_snapshots(entries, is_entry=True)
        ex_clusters = processor.cluster_snapshots(exits, is_entry=False)
        
        # Step C: Forensic Matching
        add_log(f"Matching {len(ex_clusters)} exit vehicles against baseline...")
        st.session_state.results = processor.match_clusters(en_clusters, ex_clusters)
        add_log("Pipeline cycle completed.")

# 3. Visualization
if st.session_state.results:
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Processed Vehicles", len(st.session_state.results))
    m2.metric("Verified", len([r for r in st.session_state.results if r.status == 'VERIFIED']))
    m3.metric("Theft Alerts", len([r for r in st.session_state.results if r.status == 'MISMATCH']), delta_color="inverse")

    for res in st.session_state.results:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                st.caption("Entry Point")
                st.image(res.entry_sample_img, use_container_width=True)
            with c2:
                st.caption("Exit Point")
                st.image(res.exit_sample_img, use_container_width=True)
            with c3:
                color = "green" if res.status == "VERIFIED" else "red"
                st.markdown(f"### Status: :{color}[{res.status}]")
                st.progress(res.overall_score, text=f"Confidence: {res.overall_score:.2%}")
                st.write(f"**Forensic Note:** {res.reason}")
                
                det_cols = st.columns(2)
                det_cols[0].write(f"Vehicle Sim: `{res.vehicle_similarity:.2f}`")
                det_cols[1].write(f"Driver Sim: `{res.driver_similarity:.2f}`")
else:
    st.info("System Standby. Please ingest camera data.")
