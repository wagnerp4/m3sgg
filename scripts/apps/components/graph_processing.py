import os
import time
import tempfile
from typing import Any, Dict, Optional

import streamlit as st


def render_graph_processing_button(
    uploaded_file: Any, model_path: Optional[str], max_frames: int, ctx: Dict[str, Any]
) -> None:
    """Render and handle the Scene Graph processing button and progress UI.

    :param ctx: Context with callable helpers from the host module
    :type ctx: dict
    :param uploaded_file: Streamlit uploaded file object
    :type uploaded_file: Any
    :param model_path: Path to the model checkpoint
    :type model_path: str | None
    :param max_frames: Maximum frames to process
    :type max_frames: int
    :return: None
    :rtype: None
    """
    process_video_with_sgg = ctx.get("process_video_with_sgg")
    create_processed_video_with_bboxes = ctx.get("create_processed_video_with_bboxes")
    create_processed_video_with_scene_graph = ctx.get(
        "create_processed_video_with_scene_graph"
    )

    if uploaded_file is not None and st.button("Generate Scene Graph", type="primary"):
        if not model_path:
            st.error(
                " No model checkpoint specified. Please select or provide a model path in the sidebar."
            )
        elif not os.path.exists(model_path):
            st.error(f" Model checkpoint not found at: `{model_path}`")
        else:
            start_time = time.time()
            progress_container = st.container()
            with progress_container:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("### Processing Progress")
                    timer_display = st.empty()
                    timer_display.markdown(
                        "<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>0.0s</div>",
                        unsafe_allow_html=True,
                    )
                progress_bar = st.progress(0)
                status_text = st.empty()
                st.markdown("#### Processing Log")
                log_container = st.container()
                with log_container:
                    log_display = st.empty()

            log_entries: list[str] = []

            def update_timer() -> float:
                current_time = time.time()
                elapsed = current_time - start_time
                timer_display.markdown(
                    f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>{elapsed:.1f}s</div>",
                    unsafe_allow_html=True,
                )
                return elapsed

            def update_progress(
                step: int,
                total_steps: int,
                message: str,
                log_message: Optional[str] = None,
            ) -> None:
                progress = step / total_steps
                for sub_tick in range(5):
                    sub_progress = progress + (sub_tick * 0.2 / total_steps)
                    if sub_progress <= 1.0:
                        progress_bar.progress(min(sub_progress, 1.0))
                        time.sleep(0.05)
                progress_bar.progress(min(progress, 1.0))
                status_text.text(message)
                update_timer()
                if log_message:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    log_entries.append(f"[{timestamp}] {log_message}")
                    log_text = "\n".join(log_entries[-50:])
                    log_display.markdown(
                        f"""
                        <div style="
                            height: 200px; 
                            overflow-y: auto; 
                            border: 1px solid #ccc; 
                            padding: 10px; 
                            background-color: #f8f9fa;
                            font-family: monospace;
                            font-size: 12px;
                            white-space: pre-wrap;
                        ">
                        {log_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            def update_progress_realtime(
                current_frame: int,
                total_frames: int,
                message: str,
                log_message: Optional[str] = None,
            ) -> None:
                progress = current_frame / total_frames
                progress_bar.progress(min(progress, 1.0))
                status_text.text(message)
                update_timer()
                if log_message:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    log_entries.append(f"[{timestamp}] {log_message}")
                    log_text = "\n".join(log_entries[-50:])
                    log_display.markdown(
                        f"""
                        <div style="
                            height: 200px; 
                            overflow-y: auto; 
                            border: 1px solid #ccc; 
                            padding: 10px; 
                            background-color: #f8f9fa;
                            font-family: monospace;
                            font-size: 12px;
                            white-space: pre-wrap;
                        ">
                        {log_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            def update_progress_initialization(stage: str, progress: float) -> None:
                progress_percent = int(progress * 100)
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Initializing model... {stage} ({progress_percent}%)")
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                log_entries.append(f"[{timestamp}] {stage} ({progress_percent}%)")
                log_text = "\n".join(log_entries[-50:])
                log_display.markdown(
                    f"""
                    <div style="
                        height: 200px; 
                        overflow-y: auto; 
                        border: 1px solid #ccc; 
                        padding: 10px; 
                        background-color: #f8f9fa;
                        font-family: monospace;
                        font-size: 12px;
                        white-space: pre-wrap;
                    ">
                    {log_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                update_progress(
                    0,
                    5,
                    "Initializing model and loading datasets...",
                    "Loading fasterRCNN, GloVe embeddings, and model weights...",
                )
                update_progress(
                    1,
                    5,
                    "Processing video with scene graph generation...",
                    "Starting video processing with scene graph generation",
                )

                results = None
                if callable(process_video_with_sgg):
                    results = process_video_with_sgg(
                        tmp_path,
                        model_path,
                        max_frames,
                        update_progress_initialization,
                        update_progress_realtime,
                    )

                if results:
                    update_progress(
                        2,
                        5,
                        "Video processing completed",
                        f"Video processed successfully! Analyzed {results['processed_frames']} frames",
                    )

                    update_progress(
                        3,
                        5,
                        "Creating video with bounding boxes...",
                        "Creating bounding box video...",
                    )
                    bbox_success, bbox_video_path = (False, "")
                    if callable(create_processed_video_with_bboxes):
                        bbox_success, bbox_video_path = (
                            create_processed_video_with_bboxes(
                                tmp_path, model_path, max_frames
                            )
                        )

                    if (
                        bbox_success
                        and bbox_video_path
                        and os.path.exists(bbox_video_path)
                    ):
                        file_size = os.path.getsize(bbox_video_path)
                        st.session_state["bbox_video_path"] = bbox_video_path
                        update_progress(
                            4,
                            5,
                            "Bounding box video created",
                            f"Bounding box video created successfully! Size: {file_size} bytes",
                        )
                        st.session_state["debug_bbox_path"] = bbox_video_path
                        try:
                            import cv2

                            test_cap = cv2.VideoCapture(bbox_video_path)
                            ret, test_frame = test_cap.read()
                            if ret:
                                log_entries.append(
                                    f"[{time.strftime('%H:%M:%S')}] Bounding box video file is readable by OpenCV"
                                )
                                log_text = "\n".join(log_entries[-15:])
                                log_display.text(log_text)
                            else:
                                log_entries.append(
                                    f"[{time.strftime('%H:%M:%S')}] ERROR: Bounding box video file created but not readable by OpenCV"
                                )
                                log_text = "\n".join(log_entries[-15:])
                                log_display.text(log_text)
                            test_cap.release()
                        except Exception as e:
                            log_entries.append(
                                f"[{time.strftime('%H:%M:%S')}] ERROR: Error verifying bbox video: {e}"
                            )
                            log_text = "\n".join(log_entries[-15:])
                            log_display.text(log_text)
                    else:
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] WARNING: Failed to create bounding box video"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)

                    update_progress(
                        5,
                        5,
                        "Creating video with scene graph overlay...",
                        "Creating scene graph video...",
                    )
                    sg_success, scene_graph_video_path = (False, "")
                    if callable(create_processed_video_with_scene_graph):
                        sg_success, scene_graph_video_path = (
                            create_processed_video_with_scene_graph(
                                tmp_path, model_path, max_frames
                            )
                        )

                    log_entries.append(
                        f"[{time.strftime('%H:%M:%S')}] Scene graph creation result: success={sg_success}, path_exists={os.path.exists(scene_graph_video_path) if scene_graph_video_path else False}"
                    )
                    log_text = "\n".join(log_entries[-15:])
                    log_display.text(log_text)

                    if sg_success and os.path.exists(scene_graph_video_path):
                        file_size = os.path.getsize(scene_graph_video_path)
                        st.session_state["scene_graph_video_path"] = (
                            scene_graph_video_path
                        )
                        st.session_state["debug_sg_path"] = scene_graph_video_path
                        progress_bar.progress(1.0)
                        status_text.text("Processing completed successfully!")
                        final_elapsed = update_timer()
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] Scene graph video created successfully! Size: {file_size} bytes"
                        )
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] All processing steps completed in {final_elapsed:.1f} seconds"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)
                        try:
                            import cv2

                            test_cap = cv2.VideoCapture(scene_graph_video_path)
                            ret, test_frame = test_cap.read()
                            if ret:
                                log_entries.append(
                                    f"[{time.strftime('%H:%M:%S')}] Scene graph video file is readable by OpenCV"
                                )
                                log_text = "\n".join(log_entries[-15:])
                                log_display.text(log_text)
                            else:
                                log_entries.append(
                                    f"[{time.strftime('%H:%M:%S')}] ERROR: Scene graph video file created but not readable by OpenCV"
                                )
                                log_text = "\n".join(log_entries[-15:])
                                log_display.text(log_text)
                            test_cap.release()
                        except Exception as e:
                            log_entries.append(
                                f"[{time.strftime('%H:%M:%S')}] ERROR: Error verifying scene graph video: {e}"
                            )
                            log_text = "\n".join(log_entries[-15:])
                            log_display.text(log_text)
                    else:
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] WARNING: Failed to create scene graph video"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)

                    if results.get("errors"):
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] WARNING: {len(results['errors'])} processing warnings occurred"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)
                        with st.expander("Processing Warnings", expanded=False):
                            for error in results["errors"][:5]:
                                st.warning(error)
                    st.session_state["results"] = results
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    log_entries.append(
                        f"[{time.strftime('%H:%M:%S')}] Cleaned up temporary input file"
                    )
                    log_text = "\n".join(log_entries[-15:])
                    log_display.text(log_text)
