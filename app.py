import streamlit as st
import os
import json
from src.pipeline_runner import run_full_pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="CineSense AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS (CINEMATIC DASHBOARD)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e6e6e6;
}
.block-container {
    padding-top: 2rem;
}
h1, h2 {
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background-color: #161b22;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 18px;
    box-shadow: 0 0 18px rgba(0,0,0,0.35);
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    margin-right: 8px;
}
.badge-blue { background: #2563eb; }
.badge-orange { background: #d97706; }
.badge-gray { background: #374151; }
.badge-green { background: #059669; }
.label {
    font-size: 13px;
    opacity: 0.75;
    margin-top: 8px;
}
.progress-container {
    background: #2a2f3a;
    border-radius: 999px;
    overflow: hidden;
    height: 10px;
    margin-top: 6px;
}
.progress-bar {
    height: 10px;
    background: linear-gradient(90deg, #3b82f6, #22c55e);
}
.footer {
    text-align: center;
    opacity: 0.45;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<h1>🎬 CineSense AI</h1>
<p style="opacity:0.75; max-width:900px;">
A professional AI-powered assistant that analyzes video scenes, motion, audio, emotion,
and director style to recommend intelligent cuts and transitions.
</p>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# CONTROL PANEL
# --------------------------------------------------
col1, col2, col3 = st.columns([3, 2, 2])

with col1:
    uploaded_video = st.file_uploader(
        "📤 Upload a video",
        type=["mp4", "mov", "mkv"]
    )

with col2:
    director_style = st.selectbox(
        "🎥 Director Style",
        ["nolan", "tarantino", "hirani"]
    )

with col3:
    run_button = st.button("🚀 Run Analysis")

st.divider()

# --------------------------------------------------
# PIPELINE EXECUTION
# --------------------------------------------------
if run_button and uploaded_video:
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_path = os.path.join("uploads", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    video_name = os.path.splitext(uploaded_video.name)[0]
    output_dir = os.path.join("outputs", video_name)

    with st.spinner("Analyzing video… This may take 1–2 minutes depending on length."):
        run_full_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            style=director_style
        )

    st.success("Analysis complete!")

    # --------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------
    def load_json(path):
        with open(path) as f:
            return json.load(f)

    scenes = load_json(os.path.join(output_dir, "scenes.json"))
    emotions = load_json(os.path.join(output_dir, "scene_emotions.json"))
    transitions = load_json(os.path.join(output_dir, "scene_transitions.json"))
    strengths = load_json(os.path.join(output_dir, "cut_strength.json"))

    st.subheader("📌 Scene-wise Editing Recommendations")

    # --------------------------------------------------
    # SCENE CARDS
    # --------------------------------------------------
    for scene in scenes:
        scene_id = scene["scene_id"]
        sid = f"scene_{scene_id}"
        thumb_path = os.path.join(
            output_dir,
            "scene_thumbnails",
            f"scene_{scene_id:02d}.jpg"
        )

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 3])

            with c1:
                if os.path.exists(thumb_path):
                    st.image(thumb_path, use_column_width=True)
                else:
                    st.write("Thumbnail not available")

            with c2:
                mood = emotions[sid]["color_mood"]
                transition = transitions[sid]["recommended_transition"]
                confidence = transitions[sid]["confidence"]
                cut_strength = strengths[sid]["cut_strength"]

                st.markdown(
                    f"""
                    <span class="badge badge-blue">{director_style.upper()}</span>
                    <span class="badge badge-gray">{mood.upper()}</span>
                    <span class="badge badge-orange">{transition.upper()}</span>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div class='label'>Transition Confidence</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width:{confidence * 100}%"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div class='label'>Cut Strength</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width:{cut_strength * 100}%"></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
CineSense AI • Smart Cut & Transition Recommendation Engine
</div>
""", unsafe_allow_html=True)
