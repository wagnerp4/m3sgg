import os
import tempfile
from typing import Any, Dict, Optional

import cv2
import streamlit as st


def render_video_analysis(ctx: Dict[str, Any]) -> Optional[bytes]:
    """Render the video upload and basic metadata analysis section.

    Updates session state with the uploaded file and video frame stats.

    :param ctx: Context with shared helpers
    :type ctx: dict
    :return: Uploaded file content if present, else None
    :rtype: bytes | None
    """
    st.header("Video Analysis")
    if "uploaded_video_file" not in st.session_state:
        st.session_state.uploaded_video_file = None
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Please upload a video file for scene graph generation and NLP analysis",
    )
    if uploaded_file is not None:
        st.session_state.uploaded_video_file = uploaded_file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                st.session_state["video_total_frames"] = total_frames
                cap.release()
                st.markdown("---")
                st.subheader("Video Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Frames", f"{total_frames:,}")
                with col2:
                    st.metric("FPS", f"{fps:.1f}")
                with col3:
                    st.metric("Duration", f"{duration:.1f}s")
            os.unlink(tmp_path)
        except Exception:
            pass
    return uploaded_file
