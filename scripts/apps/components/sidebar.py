import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


def render_sidebar(ctx: Dict[str, Any]) -> Tuple[Optional[str], int]:
    """Render the sidebar controls for model configuration, processing, and export.

    This function operates on and updates Streamlit session state as needed.

    :param ctx: Context with references to shared helpers and flags
    :type ctx: dict
    :return: Tuple of (model_path, max_frames)
    :rtype: tuple[str | None, int]
    """
    MODEL_DETECTOR_AVAILABLE = ctx.get("MODEL_DETECTOR_AVAILABLE", False)
    get_model_info_from_checkpoint = ctx.get("get_model_info_from_checkpoint")
    find_available_checkpoints = ctx.get("find_available_checkpoints")

    model_path: Optional[str] = st.session_state.get("model_path")

    st.markdown("---")
    st.subheader("Model Configuration")
    uploaded_file = st.file_uploader(
        "Upload Model Checkpoint",
        type=["tar", "pth", "pt"],
        help="Drag and drop a model checkpoint file (.tar, .pth, or .pt). Large files (>200MB) are supported.",
        key="checkpoint_uploader",
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        model_path = str(temp_path)
        st.session_state["model_path"] = model_path

        if MODEL_DETECTOR_AVAILABLE and get_model_info_from_checkpoint is not None:
            try:
                model_info = get_model_info_from_checkpoint(model_path)
                st.success("Checkpoint uploaded successfully!")
                st.write(f"**File:** {uploaded_file.name}")
                st.write(f"**Model Type:** {model_info['model_type'] or 'Unknown'}")
                st.write(f"**Dataset:** {model_info['dataset'] or 'Unknown'}")
                st.write(f"**Model Class:** {model_info['model_class'] or 'Unknown'}")
            except Exception as e:
                st.error(f"Error analyzing checkpoint: {e}")
                model_path = None
        else:
            st.success("Checkpoint uploaded successfully!")
            st.write(f"**File:** {uploaded_file.name}")
            st.warning("Model analysis unavailable - model_detector module not found")
            st.info(
                "The checkpoint will still work, but automatic model detection is disabled."
            )
    else:
        checkpoints: Dict[str, str] = {}
        if callable(find_available_checkpoints):
            checkpoints = find_available_checkpoints()
        if checkpoints:
            selected_model = st.selectbox(
                "Or Select Existing Checkpoint",
                list(checkpoints.keys()),
                help="Available trained models",
            )
            model_path = checkpoints[selected_model]
            st.session_state["model_path"] = model_path
            if "default" in selected_model.lower():
                st.success(" Default checkpoint loaded")
        else:
            st.warning("No trained models found in expected locations")
            st.info(
                "Upload a checkpoint file above or place model at `data/checkpoints/action_genome/sgdet_test/model_best.tar`"
            )
            model_path = st.text_input(
                "Or Enter Model Path",
                value="data/checkpoints/action_genome/sgdet_test/model_best.tar",
                placeholder="Path to model checkpoint (.tar or .pth)",
                help="Provide path to a trained model checkpoint",
            )
            if model_path:
                st.session_state["model_path"] = model_path

    st.subheader("Processing Settings")
    max_slider_value = 1000
    total_frames_info = ""

    if "video_total_frames" in st.session_state:
        total_frames = st.session_state["video_total_frames"]
        max_slider_value = min(total_frames, 1000)
        total_frames_info = f" (Video has {total_frames} total frames)"

    max_frames = st.slider(
        "Max Frames to Process",
        min_value=1,
        max_value=max_slider_value,
        value=min(30, max_slider_value),
        help=f"Maximum number of frames to process{total_frames_info}",
    )

    st.markdown("---")
    st.subheader("Export")
    export_format = st.selectbox("Export Format", ["JSON", "CSV", "XML"])

    if st.button("Download Results"):
        if "results" in st.session_state:
            results = st.session_state["results"]
            export_data = {
                "video_metadata": {
                    "total_frames": results["total_frames"],
                    "fps": results["fps"],
                    "duration_seconds": results["total_frames"] / results["fps"]
                    if results["fps"] > 0
                    else 0,
                    "processed_frames": results["processed_frames"],
                },
                "statistics": {
                    "avg_objects_per_frame": np.mean(results["detections"])
                    if results["detections"]
                    else 0,
                    "avg_relationships_per_frame": np.mean(results["relationships"])
                    if results["relationships"]
                    else 0,
                    "avg_confidence": np.mean(results["confidences"])
                    if results["confidences"]
                    else 0,
                    "error_rate_percent": (
                        len(results.get("errors", []))
                        / results["processed_frames"]
                        * 100
                    )
                    if results["processed_frames"] > 0
                    else 0,
                    "total_processing_time": sum(results["frame_times"])
                    if results["frame_times"]
                    else 0,
                },
                "frame_details": [],
            }

            for i in range(len(results["detections"])):
                frame_data = {
                    "frame_number": i + 1,
                    "objects_detected": results["detections"][i]
                    if i < len(results["detections"])
                    else 0,
                    "relationships_found": results["relationships"][i]
                    if i < len(results["relationships"])
                    else 0,
                    "confidence_score": results["confidences"][i]
                    if i < len(results["confidences"])
                    else 0,
                    "processing_time_ms": results["frame_times"][i] * 1000
                    if i < len(results["frame_times"])
                    else 0,
                }
                export_data["frame_details"].append(frame_data)

            if results.get("errors"):
                export_data["errors"] = results["errors"]

            if export_format == "JSON":
                import json

                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"scene_graph_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                )
            elif export_format == "CSV":
                summary_df = pd.DataFrame([export_data["video_metadata"]])
                stats_df = pd.DataFrame([export_data["statistics"]])
                frames_df = pd.DataFrame(export_data["frame_details"])
                csv_buffer = []
                csv_buffer.append("# Video Metadata")
                csv_buffer.append(summary_df.to_csv(index=False))
                csv_buffer.append("\n# Statistics Summary")
                csv_buffer.append(stats_df.to_csv(index=False))
                csv_buffer.append("\n# Frame-by-Frame Results")
                csv_buffer.append(frames_df.to_csv(index=False))
                if export_data.get("errors"):
                    errors_df = pd.DataFrame({"errors": export_data["errors"]})
                    csv_buffer.append("\n# Processing Errors")
                    csv_buffer.append(errors_df.to_csv(index=False))
                csv_data = "\n".join(csv_buffer)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"scene_graph_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            elif export_format == "XML":
                import xml.etree.ElementTree as ET
                import xml.dom.minidom

                root = ET.Element("scene_graph_results")
                root.set("export_date", datetime.now().isoformat())
                metadata_elem = ET.SubElement(root, "video_metadata")
                for key, value in export_data["video_metadata"].items():
                    elem = ET.SubElement(metadata_elem, key)
                    elem.text = str(value)
                stats_elem = ET.SubElement(root, "statistics")
                for key, value in export_data["statistics"].items():
                    elem = ET.SubElement(stats_elem, key)
                    elem.text = str(value)
                frames_elem = ET.SubElement(root, "frame_details")
                for frame in export_data["frame_details"]:
                    frame_elem = ET.SubElement(frames_elem, "frame")
                    for key, value in frame.items():
                        elem = ET.SubElement(frame_elem, key)
                        elem.text = str(value)
                if export_data.get("errors"):
                    errors_elem = ET.SubElement(root, "errors")
                    for error in export_data["errors"]:
                        error_elem = ET.SubElement(errors_elem, "error")
                        error_elem.text = str(error)
                xml_data = ET.tostring(root, encoding="unicode", method="xml")
                dom = xml.dom.minidom.parseString(xml_data)
                pretty_xml = dom.toprettyxml(indent="  ")
                st.download_button(
                    label="Download XML",
                    data=pretty_xml,
                    file_name=f"scene_graph_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
                    mime="application/xml",
                )

            st.success(f"{export_format} export ready for download!")
        else:
            st.warning("No results to export")

    st.markdown("---")
    st.subheader("Theme")
    dark_mode = st.button(
        "🌙 Toggle Dark Mode",
        key="dark_mode_toggle",
        help="Switch between light and dark themes",
    )
    if dark_mode:
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    return model_path, int(max_frames)
