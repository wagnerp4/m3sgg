import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_chat import message

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.apps.utils.components.dark_mode import apply_theme_styles
from scripts.apps.utils.display.processors import StreamlitVideoProcessor

try:
    from m3sgg.utils.model_detector import get_model_info_from_checkpoint
    MODEL_DETECTOR_AVAILABLE = True
    print("Model detector imported successfully")
except ImportError as e:
    print(f"Could not import model_detector: {e}")
    MODEL_DETECTOR_AVAILABLE = False

try:
    from m3sgg.language.conversation import SceneGraphChatInterface
    CHAT_INTERFACE_AVAILABLE = True
    print("Successfully imported SceneGraphChatInterface")
except ImportError as e:
    print(f"Warning: Could not import SceneGraphChatInterface: {e}")
    CHAT_INTERFACE_AVAILABLE = False

st.set_page_config(
    page_title="M3Sgg",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded", # collapsed
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
apply_theme_styles()


def get_best_video_format() -> str:
    """Get the best video format for browser compatibility
    
    :return: Best video format extension
    :rtype: str
    """
    formats_to_try = [".mp4", ".avi", ".mov"]
    return formats_to_try[0]  # Default to mp4


def convert_video_for_browser(video_path: str) -> str:
    """Convert video to browser-friendly format if needed
    
    :param video_path: Path to input video file
    :type video_path: str
    :return: Path to converted video file (or original if conversion not needed)
    :rtype: str
    """
    try:
        print(f"Starting video conversion for: {video_path}")
        
        # Check if video is already browser-friendly
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Check codec
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            print(f"Original video fourcc: {fourcc}")
            cap.release()
            
            # H264 and avc1 codecs (most browser-compatible)
            if fourcc == 875967048.0:  # H264
                print("Video is already H264, no conversion needed")
                return video_path
            elif fourcc == 875967080.0:  # avc1
                print("Video is already avc1, no conversion needed")
                return video_path
        
        # Convert to H264 if not already
        print(f"Converting video to browser-friendly format: {video_path}")
        
        # Create output path
        base_path = video_path.rsplit(".", 1)[0]
        output_path = f"{base_path}_browser_friendly.mp4"
        print(f"Output path: {output_path}")
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open input video")
            return video_path
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {fps} FPS, {width}x{height}, {frame_count} frames")
        
        # Create output with mp4v codec (case-sensitive)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            print("Video writer opened successfully")
            frame_num = 0
            # Copy all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_num += 1
                if frame_num % 10 == 0:
                    print(f"Processed {frame_num} frames")
            
            out.release()
            cap.release()
            print(f"Conversion completed, processed {frame_num} frames")
            
            # Verify output
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                test_cap.release()
                if ret and test_frame is not None:
                    print(f"Successfully converted video: {output_path}")
                    print(f"Converted video size: {os.path.getsize(output_path)} bytes")
                    return output_path
                else:
                    print("Converted video verification failed")
            else:
                print("Failed to open converted video for verification")
            
            # If conversion failed, return original
            try:
                os.unlink(output_path)
                print("Removed failed conversion file")
            except:
                pass
        else:
            print("Failed to open video writer")
        
        cap.release()
        print("Returning original video path")
        return video_path        
    except Exception as e:
        print(f"Error converting video: {e}")
        import traceback
        traceback.print_exc()
        return video_path


def validate_video_file(video_path: str) -> bool:
    """Validate that a video file is readable and contains valid frames
    
    :param video_path: Path to video file to validate
    :type video_path: str
    :return: True if video is valid, False otherwise
    :rtype: bool
    """
    try:
        if not os.path.exists(video_path):
            return False
            
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size < 1000:  # Minimum 1KB
            return False
            
        # Test with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Try to read at least one frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
        
    except Exception:
        return False


def create_processed_video_with_bboxes(
    video_path: str, model_path: str, max_frames: int = 30) -> tuple[bool, str]:
    """Create a new video file with bounding boxes drawn on frames"""
    try:
        import tempfile
        import os
        
        # Create project-local temp directory
        temp_dir = os.path.join("data", "temp_vid")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create temporary file in project temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir) as tmp_file:
            output_path = tmp_file.name
        
        processor = StreamlitVideoProcessor(model_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return False, ""

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try different codecs in order of web compatibility
        # Prioritize mp4v (case-sensitive) for better browser compatibility
        codecs_to_try = ["avc1", "mp4v", "XVID", "MJPG", "H264"]
        
        out = None
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"Successfully opened video writer with codec: {codec}")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"Failed to open video writer with codec {codec}: {e}")
                if out:
                    out.release()
                    out = None

        if not out or not out.isOpened():
            raise ValueError("Could not initialize video writer with any codec")

        frame_count = 0

        while frame_count < max_frames and cap.isOpened():
            if not (ret := cap.read())[0]:
                break
            frame = ret[1]

            # Process frame with SGG
            frame_results, entry, pred = processor.process_frame(frame)

            if entry is not None:
                frame_with_boxes = processor.draw_bounding_boxes(frame, entry)
                if frame_count < 3:  # Only log first few frames
                    print(f"Frame {frame_count}: Original shape: {frame.shape}, Processed shape: {frame_with_boxes.shape}")
                    print(f"Frame {frame_count}: Entry keys: {list(entry.keys()) if entry else 'None'}")
                    if "boxes" in entry and entry["boxes"] is not None:
                        print(f"Frame {frame_count}: Found {len(entry['boxes'])} boxes")
            else:
                frame_with_boxes = frame
                if frame_count < 3:
                    print(f"Frame {frame_count}: No entry data, using original frame")

            # Write frame (ignore write_success on Windows as it often returns False incorrectly)
            write_success = out.write(frame_with_boxes)
            # Note: On Windows, out.write() often returns False even when successful
            # We'll validate the final video file instead
            frame_count += 1

        cap.release()
        out.release()

        # Validate the created video
        if validate_video_file(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Bounding box video created successfully: {output_path}, size: {file_size} bytes")
            return True, output_path
        else:
            print(f"Bounding box video file created but validation failed: {output_path}")
            return False, ""

    except Exception as e:
        print(f"Error creating bounding box video: {e}")
        import traceback
        traceback.print_exc()
        return False, ""


def create_processed_video_with_scene_graph(
    video_path: str, model_path: str, max_frames: int = 30 ) -> tuple[bool, str]:
    """Create a new video file with scene graph overlay drawn on frames using simplified approach

    :param video_path: Path to input video file
    :type video_path: str
    :param model_path: Path to model checkpoint
    :type model_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :return: Tuple of (success, output_path) where success is bool and output_path is str
    :rtype: tuple[bool, str]
    """
    try:
        import cv2
        import tempfile
        import os
        
        # Create project-local temp directory
        temp_dir = os.path.join("data", "temp_vid")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create temporary file in project temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir) as tmp_file:
            output_path = tmp_file.name
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, ""
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {fps} FPS, {width}x{height}, {total_frames} total frames")
        
        # Try different codecs in order of web compatibility
        codecs_to_try = ["avc1", "mp4v", "XVID", "MJPG", "H264"]
        
        out = None
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                if out:
                    out.release()
                    out = None

        if not out or not out.isOpened():
            cap.release()
            return False, ""
        
        frame_count = 0
        processed_frames = 0
        
        # Create StreamlitVideoProcessor instance for proper scene graph generation
        processor = StreamlitVideoProcessor(model_path)
        
        # Process frames with scene graph overlay
        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with model to get predictions
            frame_results, entry, pred = processor.process_frame(frame)
            
            if entry is not None and pred is not None:
                # Use the proper scene graph drawing method
                frame_with_sg = processor.create_scene_graph_frame(frame, entry, pred)
            else:
                # Fallback to simple visualization if no model data
                frame_with_sg = frame.copy()
                h, w = frame_with_sg.shape[:2]
                
                # Add simple visual elements to represent scene graph
                cv2.putText(frame_with_sg, f"Scene Graph Frame {frame_count} (No Model Data)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw some simple shapes to represent objects and relationships
                cv2.rectangle(frame_with_sg, (50, 50), (150, 150), (255, 0, 0), 2)
                cv2.putText(frame_with_sg, "Object 1", (55, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                cv2.rectangle(frame_with_sg, (w-150, 50), (w-50, 150), (0, 0, 255), 2)
                cv2.putText(frame_with_sg, "Object 2", (w-145, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.line(frame_with_sg, (150, 100), (w-150, 100), (0, 255, 255), 2)
                cv2.putText(frame_with_sg, "relationship", (w//2-50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                cv2.circle(frame_with_sg, (w//2, h-100), 30, (255, 255, 0), 2)
                cv2.putText(frame_with_sg, "Node", (w//2-20, h-95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Write processed frame
            out.write(frame_with_sg)
            processed_frames += 1
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"✅ Video processing completed: {processed_frames} frames processed")
        
        # Verify output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Output file created: {file_size:,} bytes")
            
            # Test if output video is readable
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                test_cap.release()
                if ret:
                    print(f"✅ Output video is readable: shape={test_frame.shape}")
                    return True, output_path
                else:
                    print(f"❌ Output video created but not readable")
                    return False, ""
            else:
                print(f"❌ Output video created but cannot be opened")
                return False, ""
        else:
            print(f"❌ Output file was not created at: {output_path}")
            print(f"❌ Current working directory: {os.getcwd()}")
            print(f"❌ Directory contents: {os.listdir('.')}")
            return False, ""
            
    except Exception as e:
        print(f"❌ Error processing scene graph video: {e}")
        import traceback
        traceback.print_exc()
        return False, ""


def find_available_checkpoints() -> Dict[str, str]:
    """Find available model checkpoints"""
    checkpoints = {}
    default_checkpoint = Path(
        "data/checkpoints/action_genome/sgdet_test/model_best.tar"
    )

    if default_checkpoint.exists():
        checkpoints["action_genome/sgdet_test (default)"] = str(default_checkpoint)

    output_dir = Path("output")
    if output_dir.exists():
        checkpoints.update(
            {
                f"{dataset_dir.name}/{model_dir.name}": str(run_dir / "model_best.tar")
                for dataset_dir in output_dir.iterdir()
                if dataset_dir.is_dir()
                for model_dir in dataset_dir.iterdir()
                if model_dir.is_dir()
                for run_dir in model_dir.iterdir()
                if run_dir.is_dir()
                if (run_dir / "model_best.tar").exists()
            }
        )

    return checkpoints


def process_video_with_sgg(
    video_path: str, model_path: str, max_frames: int = 30, init_progress_callback=None, frame_progress_callback=None) -> Dict[str, Any]:
    """Process video using scene graph generation"""
    try:
        # Initialize video processor with progress callback
        processor = StreamlitVideoProcessor(model_path, init_progress_callback)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = {
            "total_frames": total_frames,
            "fps": fps,
            "processed_frames": 0,
            "detections": [],
            "relationships": [],
            "confidences": [],
            "frame_times": [],
            "errors": [],
            "frame_objects": [],  # Store object info for each frame
            "frame_relationships": [],  # Store relationship info for each frame
        }

        frame_count = 0

        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create frame-specific progress callback
            def frame_progress_callback_internal(stage, progress):
                if frame_progress_callback:
                    # Calculate overall progress: (frame_count / max_frames) + (progress / max_frames)
                    current_progress = (frame_count + progress) / max_frames
                    frame_progress_callback(current_progress, max_frames, f"Frame {frame_count + 1}/{max_frames} - {stage}", 
                                    f"Processing frame {frame_count + 1}: {stage}")

            # Process frame with SGG
            frame_results, entry, pred = processor.process_frame(frame, frame_progress_callback_internal)

            results["detections"].append(frame_results.get("objects", 0))
            results["relationships"].append(frame_results.get("relationships", 0))
            results["confidences"].append(frame_results.get("confidence", 0.0))
            results["frame_times"].append(frame_count / fps)

            # Extract bbox info for each frame
            if entry is not None:
                bbox_info = processor.extract_bbox_info(entry, confidence_threshold=0.1)
                results["frame_objects"].append(bbox_info)

                # Extract relationship info for each frame
                relationship_info = processor.extract_relationships(entry, pred)
                results["frame_relationships"].append(relationship_info)

                # Store first frame bbox info in session state for backward compatibility
                if frame_count == 0:
                    st.session_state["bbox_info"] = bbox_info
                    st.session_state["relationship_info"] = relationship_info
            else:
                results["frame_objects"].append([])
                results["frame_relationships"].append([])

            if "error" in frame_results:
                results["errors"].append(
                    f"Frame {frame_count}: {frame_results['error']}"
                )

            frame_count += 1

        cap.release()
        results["processed_frames"] = frame_count

        return results

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None


def cleanup_temp_videos():
    """Clean up temporary video files from data/temp_vid/ directory on exit"""
    import os
    import shutil
    
    # Clean up the project temp directory (output videos)
    temp_dir = os.path.join("data", "temp_vid")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary video directory: {temp_dir}")
        except Exception as e:
            print(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    # Clean up any remaining session state video files
    cleanup_paths = []
    if "bbox_video_path" in st.session_state:
        cleanup_paths.append(st.session_state["bbox_video_path"])
    if "scene_graph_video_path" in st.session_state:
        cleanup_paths.append(st.session_state["scene_graph_video_path"])
    
    for path in cleanup_paths:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
                print(f"Cleaned up temporary file: {path}")
        except Exception as e:
            print(f"Failed to clean up {path}: {e}")


def main():
    #--------------------------------
    # Header
    st.markdown(
        '<h1 class="main-header"> M3SGG</h1>',
        unsafe_allow_html=True,
    )
    # st.markdown("Video Scene Graph Generation with Deep Learning Models")
    st.markdown("---")

    #--------------------------------
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("Model Configuration")
        uploaded_file = st.file_uploader(
            "Upload Model Checkpoint",
            type=['tar', 'pth', 'pt'],
            help="Drag and drop a model checkpoint file (.tar, .pth, or .pt). Large files (>200MB) are supported.",
            key="checkpoint_uploader",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # TODO: Store better
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            model_path = str(temp_path)
            st.session_state["model_path"] = model_path
            
            # Display model info
            if MODEL_DETECTOR_AVAILABLE:
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
                st.info("The checkpoint will still work, but automatic model detection is disabled.")
        else:
            # TODO: Test with different checkpoints
            # TODO: Use better default path
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

        # TODO: Add back in
        # confidence_threshold = st.slider(
        #     "Confidence Threshold",
        #     0.0,
        #     1.0,
        #     0.5,
        #     help="Minimum confidence for object detection",
        # )

        # if st.button("Reset Settings"):
        #     st.rerun()

        st.markdown("---")
        st.subheader("Export")
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "XML"])

        # Download Section
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

                # Generate export based on format
                if export_format == "JSON":
                    # TODO: Modularize
                    import json
                    json_data = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"scene_graph_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

                elif export_format == "CSV":
                    # TODO: Modularize
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
                    # TODO: Modularize
                    import xml.etree.ElementTree as ET
                    root = ET.Element("scene_graph_results")
                    root.set("export_date", datetime.now().isoformat())
                    # Video metadata
                    metadata_elem = ET.SubElement(root, "video_metadata")
                    for key, value in export_data["video_metadata"].items():
                        elem = ET.SubElement(metadata_elem, key)
                        elem.text = str(value)
                    # Statistics
                    stats_elem = ET.SubElement(root, "statistics")
                    for key, value in export_data["statistics"].items():
                        elem = ET.SubElement(stats_elem, key)
                        elem.text = str(value)
                    # Frame details
                    frames_elem = ET.SubElement(root, "frame_details")
                    for frame in export_data["frame_details"]:
                        frame_elem = ET.SubElement(frames_elem, "frame")
                        for key, value in frame.items():
                            elem = ET.SubElement(frame_elem, key)
                            elem.text = str(value)
                    # Errors if any
                    if export_data.get("errors"):
                        errors_elem = ET.SubElement(root, "errors")
                        for error in export_data["errors"]:
                            error_elem = ET.SubElement(errors_elem, "error")
                            error_elem.text = str(error)
                    # Pretty format
                    import xml.dom.minidom
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

        # Dark mode toggle at bottom of sidebar
        st.markdown("---")
        st.subheader("Theme")
        dark_mode = st.button("🌙 Toggle Dark Mode", key="dark_mode_toggle", help="Switch between light and dark themes")
        if dark_mode:
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    #--------------------------------
    # Video Analysis
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
                
                # Display video info below upload component
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

    #--------------------------------
    # Graph Processing Button
    if uploaded_file is not None and st.button("Generate Scene Graph", type="primary"):
        if not model_path:
            st.error(
                " No model checkpoint specified. Please select or provide a model path in the sidebar."
            )
        elif not os.path.exists(model_path):
            st.error(f" Model checkpoint not found at: `{model_path}`")

        else:
            import time
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
                # Create a scrollable log container
                log_container = st.container()
                with log_container:
                    log_display = st.empty()

            log_entries = []

            def update_progress(step, total_steps, message, log_message=None):
                """Update progress bar and log display with sub-ticks for smoother updates"""
                # TODO: Adapt test function call for this
                progress = step / total_steps
                
                # Add sub-ticks for smoother progress bar updates
                # Update progress bar with sub-tick increments
                for sub_tick in range(5):  # 5 sub-ticks per main step
                    sub_progress = progress + (sub_tick * 0.2 / total_steps)
                    if sub_progress <= 1.0:
                        progress_bar.progress(min(sub_progress, 1.0))
                        time.sleep(0.05)  # Small delay for visual effect
                
                # Final progress update
                progress_bar.progress(min(progress, 1.0))
                status_text.text(message)

                # Update timer
                current_time = time.time()
                elapsed = current_time - start_time
                timer_display.markdown(
                    f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>{elapsed:.1f}s</div>",
                    unsafe_allow_html=True,
                )

                if log_message:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    log_entries.append(f"[{timestamp}] {log_message}")
                    # Display log with scrollable container
                    log_text = "\n".join(log_entries[-50:])  # Show last 50 entries for better scrolling
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
                        unsafe_allow_html=True
                    )

            def update_progress_realtime(current_frame, total_frames, message, log_message=None):
                """Update progress bar in real-time with frame-level granularity"""
                progress = current_frame / total_frames
                
                # Update progress bar immediately without artificial delays
                progress_bar.progress(min(progress, 1.0))
                status_text.text(message)

                # Update timer
                current_time = time.time()
                elapsed = current_time - start_time
                timer_display.markdown(
                    f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>{elapsed:.1f}s</div>",
                    unsafe_allow_html=True,
                )

                if log_message:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    log_entries.append(f"[{timestamp}] {log_message}")
                    # Display log with scrollable container
                    log_text = "\n".join(log_entries[-50:])  # Show last 50 entries for better scrolling
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
                        unsafe_allow_html=True
                    )

            def update_progress_initialization(stage, progress):
                """Update progress bar during model initialization"""
                # Convert progress (0.0-1.0) to percentage
                progress_percent = int(progress * 100)
                
                # Update progress bar immediately without artificial delays
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Initializing model... {stage} ({progress_percent}%)")

                # Update timer
                current_time = time.time()
                elapsed = current_time - start_time
                timer_display.markdown(
                    f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>{elapsed:.1f}s</div>",
                    unsafe_allow_html=True,
                )

                # Log the initialization step
                timestamp = time.strftime("%H:%M:%S", time.localtime())
                log_entries.append(f"[{timestamp}] {stage} ({progress_percent}%)")
                # Display log with scrollable container
                log_text = "\n".join(log_entries[-50:])  # Show last 50 entries for better scrolling
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
                    unsafe_allow_html=True
                )

            def update_timer():
                """Update the timer display"""
                current_time = time.time()
                elapsed = current_time - start_time
                timer_display.markdown(
                    f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>{elapsed:.1f}s</div>",
                    unsafe_allow_html=True,
                )
                return elapsed

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Step 0: Initialize model (this is where the overhead happens)
                update_progress(
                    0,
                    5,
                    "Initializing model and loading datasets...",
                    "Loading fasterRCNN, GloVe embeddings, and model weights...",
                )

                # Step 1: Process video with SGG
                update_progress(
                    1,
                    5,
                    "Processing video with scene graph generation...",
                    "Starting video processing with scene graph generation",
                )

                results = process_video_with_sgg(tmp_path, model_path, max_frames, update_progress_initialization, update_progress_realtime)

                if results:
                    update_progress(
                        2,
                        5,
                        "Video processing completed",
                        f"Video processed successfully! Analyzed {results['processed_frames']} frames",
                    )


                    # Step 2: Create bounding box video
                    update_progress(
                        3,
                        5,
                        "Creating video with bounding boxes...",
                        "Creating bounding box video...",
                    )

                    bbox_success, bbox_video_path = create_processed_video_with_bboxes(
                        tmp_path, model_path, max_frames
                    )

                    if bbox_success and bbox_video_path and os.path.exists(bbox_video_path):
                        print(f"Using original avc1 video: {bbox_video_path}")
                        file_size = os.path.getsize(bbox_video_path)
                        st.session_state["bbox_video_path"] = bbox_video_path
                        update_progress(
                            4,
                            5,
                            "Bounding box video created",
                            f"Bounding box video created successfully! Size: {file_size} bytes",
                        )
                        
                        # Debug: Store video path for display
                        st.session_state["debug_bbox_path"] = bbox_video_path

                        # Verify bbox video
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

                    # Step 3: Create scene graph video
                    update_progress(
                        5,
                        5,
                        "Creating video with scene graph overlay...",
                        "Creating scene graph video...",
                    )
                    sg_success, scene_graph_video_path = create_processed_video_with_scene_graph(
                        tmp_path, model_path, max_frames
                    )
                    
                    # Debug: Check what happened
                    log_entries.append(
                        f"[{time.strftime('%H:%M:%S')}] Scene graph creation result: success={sg_success}, path_exists={os.path.exists(scene_graph_video_path) if scene_graph_video_path else False}"
                    )
                    if sg_success and scene_graph_video_path and os.path.exists(scene_graph_video_path):
                        file_size = os.path.getsize(scene_graph_video_path)
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] Scene graph file exists: {scene_graph_video_path}, size: {file_size} bytes"
                        )
                    else:
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] Scene graph file missing: {scene_graph_video_path}"
                        )
                    log_text = "\n".join(log_entries[-15:])
                    log_display.text(log_text)

                    if sg_success and os.path.exists(scene_graph_video_path):
                        file_size = os.path.getsize(scene_graph_video_path)
                        st.session_state["scene_graph_video_path"] = scene_graph_video_path
                        
                        # Debug: Store video path for display
                        st.session_state["debug_sg_path"] = scene_graph_video_path

                        # Final progress update
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

                        # Verify scene graph video
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

                    # Show processing warnings if any
                    if results.get("errors"):
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] WARNING: {len(results['errors'])} processing warnings occurred"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)
                        with st.expander("Processing Warnings", expanded=False):
                            for error in results["errors"][:5]:  # Show first 5 errors
                                st.warning(error)

                    st.session_state["results"] = results
            finally:
                # Clean up temporary input file immediately after processing
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    log_entries.append(
                        f"[{time.strftime('%H:%M:%S')}] Cleaned up temporary input file"
                    )
                    log_text = "\n".join(log_entries[-15:])
                    log_display.text(log_text)

    #--------------------------------
    # Result View Tabs
    main_tab1, main_tab2 = st.tabs(["SGG View", "Advanced SGG View"])

    # SGG View Tab
    with main_tab1:

        st.header(" Video Players")
        if st.session_state.uploaded_video_file is not None:
            vid_col1, vid_col2, vid_col3 = st.columns(3)
            # Unprocessed Video
            with vid_col1:
                st.subheader("Original Video")
                st.video(st.session_state.uploaded_video_file)
            # Bounding Box Video
            with vid_col2:
                st.subheader("Object Detection")
                
                # Use the processed bbox video from session state (avc1 encoded)
                bbox_video_path = st.session_state.get("bbox_video_path") or st.session_state.get("debug_bbox_path")
                
                if bbox_video_path and os.path.exists(bbox_video_path):
                    bbox_size = os.path.getsize(bbox_video_path)
                    import cv2
                    cap = cv2.VideoCapture(bbox_video_path)
                    bbox_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
                    cap.release()
                    
                    # Validate and display video
                    if validate_video_file(bbox_video_path):
                        try:
                            # Try direct path first
                            st.video(bbox_video_path)
                        except Exception as e:
                            st.warning(f"Direct video display failed: {e}")
                            try:
                                # Fallback to bytes
                                with open(bbox_video_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                            except Exception as e2:
                                st.error(f"Error loading bbox video: {e2}")
                                st.video(st.session_state.uploaded_video_file)
                    else:
                        st.error("Bounding box video file is corrupted or unreadable")
                        st.video(st.session_state.uploaded_video_file)
                else:
                    st.video(st.session_state.uploaded_video_file)
                    st.warning("No bounding box video available - showing original")


                # Add bbox table if we have detection results
                if "results" in st.session_state and "bbox_info" in st.session_state:
                    st.markdown("---")
                    st.subheader("Detected Objects")
                    bbox_info = st.session_state["bbox_info"]
                    if bbox_info:
                        # Create DataFrame for the table
                        bbox_df = pd.DataFrame(
                            [
                                {
                                    "Object": bbox["object_name"],
                                    "Confidence": f"{bbox['confidence']:.3f}",
                                    "BBox": f"[{bbox['bbox'][0]:.0f}, {bbox['bbox'][1]:.0f}, {bbox['bbox'][2]:.0f}, {bbox['bbox'][3]:.0f}]",
                                }
                                for bbox in bbox_info
                            ]
                        )
                        st.dataframe(bbox_df, width="stretch", hide_index=True)
                    else:
                        st.info("No objects detected above confidence threshold")

            # Scene Graph Video
            with vid_col3:
                st.subheader("Scene Graph Analysis")
                
                # Use the processed scene graph video from session state (avc1 encoded)
                sg_video_path = st.session_state.get("scene_graph_video_path") or st.session_state.get("debug_sg_path")
                
                if sg_video_path and os.path.exists(sg_video_path):
                    # Debug information
                    sg_size = os.path.getsize(sg_video_path)
                    import cv2
                    cap = cv2.VideoCapture(sg_video_path)
                    sg_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
                    cap.release()
                    
                    # Validate and display video
                    if validate_video_file(sg_video_path):
                        try:
                            # Try direct path first
                            st.video(sg_video_path)
                        except Exception as e:
                            st.warning(f"Direct video display failed: {e}")
                            try:
                                # Fallback to bytes
                                with open(sg_video_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                            except Exception as e2:
                                st.error(f"Error loading scene graph video: {e2}")
                                st.video(st.session_state.uploaded_video_file)
                                st.caption(" Scene graph overlay failed to load")
                    else:
                        st.error("Scene graph video file is corrupted or unreadable")
                        st.video(st.session_state.uploaded_video_file)
                else:
                    st.video(st.session_state.uploaded_video_file)
                    st.warning("No scene graph video available - showing original")

                # Add relationship table if we have relationship results
                if "results" in st.session_state and "relationship_info" in st.session_state:
                    st.markdown("---")
                    st.subheader("Scene Graph Relationships")
                    relationship_info = st.session_state["relationship_info"]
                    if relationship_info:
                        # Create a temporary processor to get object/relationship names
                        temp_processor = None
                        if "model_path" in st.session_state:
                            try:
                                temp_processor = StreamlitVideoProcessor(st.session_state["model_path"])
                            except Exception:
                                pass
                        
                        relationship_data = []
                        for rel in relationship_info:
                            # Get subject and object names
                            subject_name = "person1"  # Default as requested
                            if "subject_class" in rel and temp_processor:
                                subject_name = temp_processor.get_object_name(rel["subject_class"])
                            
                            object_name = "object"
                            if "object_class" in rel and temp_processor:
                                object_name = temp_processor.get_object_name(rel["object_class"])
                            
                            # Get relationship name
                            relationship_name = "interacts_with"  # Default
                            if temp_processor:
                                if "attention_type" in rel and "attention_confidence" in rel:
                                    if rel["attention_confidence"] > 0.1:
                                        relationship_name = temp_processor.get_relationship_name(
                                            rel["attention_type"], "attention"
                                        )
                                elif "spatial_type" in rel and "spatial_confidence" in rel:
                                    if rel["spatial_confidence"] > 0.1:
                                        relationship_name = temp_processor.get_relationship_name(
                                            rel["spatial_type"], "spatial"
                                        )
                            
                            relationship_data.append({
                                "Subject": subject_name,
                                "Relation": relationship_name,
                                "Object": object_name,
                                "Confidence": f"{rel['confidence']:.3f}"
                            })
                        
                        if relationship_data:
                            relationship_df = pd.DataFrame(relationship_data)
                            st.dataframe(relationship_df, width="stretch", hide_index=True)
                        else:
                            st.info("No relationships detected above confidence threshold")
                    else:
                        st.info("No relationships detected above confidence threshold")

            # Chat
            st.markdown("---")
            st.header("Chat Assistant")
            
            if CHAT_INTERFACE_AVAILABLE:
                # Use the new LLM-based chat interface
                if "chat_interface" not in st.session_state:
                    st.session_state.chat_interface = SceneGraphChatInterface(
                        model_name="google/gemma-3-270m",
                        model_type="gemma"
                    )
                
                # Set scene graph context if results are available
                if "results" in st.session_state:
                    st.session_state.chat_interface.set_scene_graph_context(
                        st.session_state["results"]
                    )
                
                # Render the chat interface
                st.session_state.chat_interface.render_chat_interface()
                
            else:
                # Fallback to simple chat interface
                st.warning("Advanced chat interface not available. Using basic chat.")
                if "chat_messages" not in st.session_state:
                    st.session_state.chat_messages = []
                    st.session_state.chat_intro_started = False
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                if not st.session_state.get("chat_intro_started", False):
                    st.session_state.chat_intro_started = True
                    intro_messages = [
                        """Hello there! Welcome to VidSgg... 
                        I'm your personal AI assistant for video scene graph analysis. 
                        I can help you discover hidden relationships and objects in your videos!
                        Just upload a video above to start.""",
                    ]
                    st.session_state.chat_messages = [
                        {"message": intro_messages[0], "is_user": False}
                    ]
                    intro_container = st.empty()
                    with intro_container.container():
                        message(
                            st.session_state.chat_messages[0]["message"],
                            is_user=False,
                            key=f"intro_0_{uuid.uuid4().hex[:8]}",
                            allow_html=True,
                        )
                    intro_container.empty()

                def handle_chat_input():
                    user_input = st.session_state.chat_input
                    if user_input.strip():
                        # Add user message
                        st.session_state.chat_messages.append(
                            {"message": user_input, "is_user": True}
                        )

                        # Generate bot response (placeholder logic)
                        bot_response = generate_bot_response(user_input)
                        st.session_state.chat_messages.append(
                            {"message": bot_response, "is_user": False}
                        )

                        # Clear input
                        st.session_state.chat_input = ""

                def generate_bot_response(user_input: str) -> str:
                    """Generate bot response based on user input"""
                    user_input_lower = user_input.lower()

                    if "hello" in user_input_lower or "hi" in user_input_lower:
                        return "Hello! I'm your VidSgg assistant. I can help you understand scene graph generation results and answer questions about the analysis."
                    elif "help" in user_input_lower:
                        return "I can help you with:\n• Understanding scene graph results\n• Explaining object detections\n• Interpreting relationship data\n• Model configuration questions\n• Export options"
                    elif "object" in user_input_lower and "results" in st.session_state:
                        results = st.session_state["results"]
                        avg_objects = (
                            np.mean(results["detections"]) if results["detections"] else 0
                        )
                        return f"In your video analysis, I detected an average of {avg_objects:.1f} objects per frame across {results['processed_frames']} processed frames."
                    elif (
                        "relationship" in user_input_lower and "results" in st.session_state
                    ):
                        results = st.session_state["results"]
                        avg_relationships = (
                            np.mean(results["relationships"])
                            if results["relationships"]
                            else 0
                        )
                        return f"The analysis found an average of {avg_relationships:.1f} relationships per frame in your video."
                    elif "confidence" in user_input_lower and "results" in st.session_state:
                        results = st.session_state["results"]
                        avg_confidence = (
                            np.mean(results["confidences"]) if results["confidences"] else 0
                        )
                        return f"The average confidence score across all detections was {avg_confidence:.2f}."
                    elif "model" in user_input_lower:
                        return "The VidSgg model uses STTran (Spatial-Temporal Transformer) for scene graph generation. It processes video frames to detect objects and their relationships over time."
                    elif "export" in user_input_lower:
                        return "You can export your results in JSON, CSV, or XML format using the export options in the sidebar. The exported data will include all detection and relationship information."
                    else:
                        return "I'm here to help with your scene graph analysis! Ask me about objects, relationships, confidence scores, or how the model works."

                chat_container = st.container() # Display chat messages
                with chat_container:
                    for i, msg in enumerate(st.session_state.chat_messages):
                        message(
                            msg["message"],
                            is_user=msg["is_user"],
                            key=f"chat_msg_{i}",
                            allow_html=True,
                        )
                st.text_input( # Chat input
                    "Ask me about your scene graph analysis:",
                    key="chat_input",
                    on_change=handle_chat_input,
                    placeholder="Type your question here...",
                )
                if st.button("Clear Chat"): # Clear chat button
                    st.session_state.chat_messages = []
                    st.rerun()

            # Sub-tabs for Frame view and Temporal view
            st.markdown("---")
            sgg_tab1, sgg_tab2 = st.tabs(["Temporal View", "NLP View"])

            with sgg_tab1:
                st.header("Temporal Scene Graph Analysis")

                # Results visualization if available
                if "results" in st.session_state:
                    results = st.session_state["results"]

                    if results["detections"]:
                        df = pd.DataFrame(
                            {
                                "Frame": range(len(results["detections"])),
                                "Objects_Detected": results["detections"],
                                "Relationships": results["relationships"],
                                "Confidence": results["confidences"],
                                "Time_Seconds": results["frame_times"],
                            }
                        )

                        # Vertical Object Node Timeline Visualization
                        st.subheader("Object Node Timeline")

                        # Create a vertical timeline showing object presence across frames
                        if results.get("frame_objects") and any(
                            results["frame_objects"]
                        ):
                            # Collect all unique objects across all frames
                            all_objects = set()
                            for frame_objects in results["frame_objects"]:
                                for obj in frame_objects:
                                    all_objects.add(obj["object_name"])

                            if all_objects:
                                all_objects = sorted(list(all_objects))

                                # Create vertical timeline visualization with connected spheres
                                fig_timeline = go.Figure()

                                # Define positions: person at top (y=0), objects below (y=1,2,3...)
                                person_y = 0
                                object_y_positions = list(
                                    range(1, len(all_objects) + 1)
                                )

                                # Add person node (always present) - RED
                                fig_timeline.add_trace(
                                    go.Scatter(
                                        x=[0, len(results["frame_objects"]) - 1],
                                        y=[person_y, person_y],
                                        mode="lines",
                                        name="Person",
                                        line=dict(width=8, color="red"),
                                        showlegend=True,
                                    )
                                )

                                # Add person spheres for each frame
                                for frame_idx in range(len(results["frame_objects"])):
                                    fig_timeline.add_trace(
                                        go.Scatter(
                                            x=[frame_idx],
                                            y=[person_y],
                                            mode="markers",
                                            marker=dict(
                                                size=20,
                                                color="red",
                                                line=dict(width=2, color="white"),
                                            ),
                                            name=f"Person (Frame {frame_idx})",
                                            hovertemplate="<b>Person</b><br>"
                                            + f"Frame: {frame_idx}<br>"
                                            + "<extra></extra>",
                                            showlegend=False,
                                        )
                                    )

                                # Add object nodes and their spheres with vertical connections
                                # Define distinct colors for better visibility
                                distinct_colors = [
                                    "rgb(0, 100, 200)",      # Dark blue
                                    "rgb(200, 50, 50)",      # Red
                                    "rgb(50, 150, 50)",      # Green
                                    "rgb(150, 50, 150)",     # Purple
                                    "rgb(200, 100, 0)",      # Orange
                                    "rgb(0, 150, 150)",      # Teal
                                    "rgb(150, 100, 50)",     # Brown
                                    "rgb(100, 0, 200)",      # Violet
                                    "rgb(200, 150, 0)",      # Gold
                                    "rgb(50, 100, 150)",     # Steel blue
                                    "rgb(150, 50, 100)",     # Magenta
                                ]
                                
                                for i, obj_name in enumerate(all_objects):
                                    obj_y = object_y_positions[i]

                                    # Use distinct colors from predefined palette
                                    color_index = i % len(distinct_colors)
                                    object_color = distinct_colors[color_index]

                                    # Add object horizontal line
                                    fig_timeline.add_trace(
                                        go.Scatter(
                                            x=[0, len(results["frame_objects"]) - 1],
                                            y=[obj_y, obj_y],
                                            mode="lines",
                                            name=obj_name,
                                            line=dict(width=6, color=object_color),
                                            showlegend=True,
                                        )
                                    )

                                    # Collect frames where this object appears
                                    object_frames = []
                                    object_confidences = []

                                    for frame_idx, frame_objects in enumerate(
                                        results["frame_objects"]
                                    ):
                                        # Check if object is present in this frame
                                        obj_in_frame = None
                                        for obj in frame_objects:
                                            if obj["object_name"] == obj_name:
                                                obj_in_frame = obj
                                                break

                                        if obj_in_frame:
                                            object_frames.append(frame_idx)
                                            object_confidences.append(
                                                obj_in_frame["confidence"]
                                            )

                                    # Add spheres for this object
                                    if object_frames:
                                        fig_timeline.add_trace(
                                            go.Scatter(
                                                x=object_frames,
                                                y=[obj_y] * len(object_frames),
                                                mode="markers",
                                                marker=dict(
                                                    size=20,
                                                    color=object_color,
                                                    line=dict(width=2, color="white"),
                                                ),
                                                name=f"{obj_name} spheres",
                                                hovertemplate=f"<b>{obj_name}</b><br>"
                                                + "Frame: %{x}<br>"
                                                + "Confidence: %{customdata:.3f}<br>"
                                                + "<extra></extra>",
                                                customdata=object_confidences,
                                                showlegend=False,
                                            )
                                        )

                                        # Add vertical connecting lines between spheres
                                        if len(object_frames) > 1:
                                            fig_timeline.add_trace(
                                                go.Scatter(
                                                    x=object_frames,
                                                    y=[obj_y] * len(object_frames),
                                                    mode="lines",
                                                    line=dict(
                                                        width=3,
                                                        color=object_color,
                                                        dash="solid",
                                                    ),
                                                    name=f"{obj_name} connections",
                                                    showlegend=False,
                                                    hoverinfo="skip",
                                                )
                                            )

                                    # Add edges from person to object spheres (vertical lines)
                                    for frame_idx in object_frames:
                                        fig_timeline.add_trace(
                                            go.Scatter(
                                                x=[frame_idx, frame_idx],
                                                y=[person_y, obj_y],
                                                mode="lines",
                                                line=dict(
                                                    width=2, color="gray", dash="dot"
                                                ),
                                                name=f"Edge {frame_idx}",
                                                hovertemplate=f"<b>Person → {obj_name}</b><br>"
                                                + f"Frame: {frame_idx}<br>"
                                                + "<extra></extra>",
                                                showlegend=False,
                                            )
                                        )

                                fig_timeline.update_layout(
                                    title="Object Nodes Across Frames with Person-Object Relationships",
                                    xaxis=dict(
                                        title="Frame Number",
                                        tickmode="linear",
                                        tick0=0,
                                        dtick=1,
                                        showgrid=True,
                                        gridcolor="lightgray",
                                    ),
                                    yaxis=dict(
                                        title="Nodes",
                                        tickmode="array",
                                        tickvals=[person_y] + object_y_positions,
                                        ticktext=["Person"] + all_objects,
                                        side="left",
                                    ),
                                    height=400 + (len(all_objects) + 1) * 30,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="v",
                                        yanchor="top",
                                        y=1,
                                        xanchor="left",
                                        x=1.02,
                                    ),
                                )
                                st.plotly_chart(fig_timeline, width="stretch")

                                # Object statistics
                                st.subheader("Object Statistics")
                                obj_stats = []
                                for obj_name in all_objects:
                                    presence_count = sum(
                                        1
                                        for frame_objects in results["frame_objects"]
                                        if any(
                                            obj["object_name"] == obj_name
                                            for obj in frame_objects
                                        )
                                    )
                                    total_frames = len(results["frame_objects"])
                                    presence_percentage = (
                                        (presence_count / total_frames) * 100
                                        if total_frames > 0
                                        else 0
                                    )

                                    # Calculate average confidence for this object
                                    confidences = []
                                    for frame_objects in results["frame_objects"]:
                                        for obj in frame_objects:
                                            if obj["object_name"] == obj_name:
                                                confidences.append(obj["confidence"])
                                    avg_confidence = (
                                        np.mean(confidences) if confidences else 0.0
                                    )

                                    obj_stats.append(
                                        {
                                            "Object": obj_name,
                                            "Frames Present": presence_count,
                                            "Total Frames": total_frames,
                                            "Presence %": f"{presence_percentage:.1f}%",
                                            "Avg Confidence": f"{avg_confidence:.3f}",
                                        }
                                    )

                                stats_df = pd.DataFrame(obj_stats)
                                st.dataframe(
                                    stats_df, width="stretch", hide_index=True
                                )
                            else:
                                st.info("No objects detected in any frame")
                        else:
                            st.info(
                                "No detailed object information available for timeline visualization"
                            )

                        # Multi-line chart for detections and relationships
                        st.subheader("Scene Graph Metrics Over Time")
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=df["Time_Seconds"],
                                y=df["Objects_Detected"],
                                mode="lines+markers",
                                name="Objects Detected",
                                line=dict(color="blue"),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df["Time_Seconds"],
                                y=df["Relationships"],
                                mode="lines+markers",
                                name="Relationships",
                                line=dict(color="red"),
                                yaxis="y2",
                            )
                        )

                        fig.update_layout(
                            title="Scene Graph Analysis Over Time",
                            xaxis_title="Time (seconds)",
                            yaxis=dict(title="Objects Detected", side="left"),
                            yaxis2=dict(
                                title="Relationships", side="right", overlaying="y"
                            ),
                            legend=dict(x=0.02, y=0.98),
                        )
                        st.plotly_chart(fig, width="stretch")

                        # Confidence chart
                        fig3 = px.line(
                            df,
                            x="Time_Seconds",
                            y="Confidence",
                            title="Average Confidence Over Time",
                            labels={
                                "Time_Seconds": "Time (seconds)",
                                "Confidence": "Avg Confidence",
                            },
                        )
                        st.plotly_chart(fig3, width="stretch")

                        # Data table
                        st.subheader("Detection Details")
                        st.dataframe(df, width="stretch")
        
                # NLP View Tab Implementation
            
            with sgg_tab2:
                st.header(" NLP Analysis")

                # Single video player spanning full width
                if st.session_state.uploaded_video_file is not None:
                    st.subheader(" Video Analysis")
                    st.video(st.session_state.uploaded_video_file)

                    # NLP Analysis Results
                    st.markdown("---")
                    st.header("NLP Module Results")

                    # Create columns for different NLP outputs
                    nlp_col1, nlp_col2 = st.columns(2)

                    with nlp_col1:
                        st.subheader("Video Summarization")
                        summarization_text = """
                        **Automatically Generated Summary:**
                        
                        This video contains multiple scenes with various objects and activities. 
                        The analysis detected people interacting with objects in different spatial 
                        configurations. Key activities include movement patterns, object 
                        manipulations, and social interactions between detected entities.
                        
                        **Key Findings:**
                        • Multiple human subjects identified
                        • Various object interactions detected
                        • Temporal activity patterns observed
                        • Scene transitions and context changes noted
                        """
                        st.text_area("Summary", summarization_text, height=200, disabled=True)

                        st.subheader(" Video Semantic Search")
                        search_results = """
                        🔎 **Semantic Search Results:**
                        
                        **Query:** "People walking"
                        **Timestamps:** 0:12-0:18, 0:45-0:52, 1:23-1:30
                        
                        **Query:** "Object interaction"
                        **Timestamps:** 0:25-0:35, 1:05-1:15, 1:40-1:50
                        
                        **Query:** "Group activity"
                        **Timestamps:** 0:30-0:55, 1:10-1:35
                        """
                        st.text_area(
                            "Search Results", search_results, height=150, disabled=True
                        )

                    with nlp_col2:
                        st.subheader("Video Captioning")
                        captioning_text = """
                        **Frame-by-Frame Captions:**
                        
                        **00:05** - A person stands near a table with objects
                        **00:12** - Multiple people enter the scene from the left
                        **00:18** - Someone picks up an object from the surface
                        **00:25** - Two people engage in conversation
                        **00:32** - Group activity begins with shared focus
                        **00:40** - Objects are rearranged on the table
                        **00:48** - People move toward the background
                        **00:55** - Scene transitions to new activity
                        **01:02** - New objects appear in the frame
                        **01:10** - Final interactions before scene ends
                        """
                        st.text_area("Captions", captioning_text, height=200, disabled=True)

                        st.subheader("🔮 Action Anticipation")
                        anticipation_text = """
                        **Predicted Future Actions:**
                        
                        **Next 5 seconds:**
                        • Person likely to move towards door (85% confidence)
                        • Object manipulation probability: 72%
                        • Group dispersal expected: 68%
                        
                        **Next 10 seconds:**
                        • Scene change probability: 91%
                        • New person entry likelihood: 45%
                        • Activity continuation: 23%
                        
                        **Temporal Patterns:**
                        • Regular 15-second activity cycles detected
                        • Spatial movement patterns suggest routine behavior
                        """
                        st.text_area(
                            "Predictions", anticipation_text, height=150, disabled=True
                        )

                    # Additional NLP Features
                    st.markdown("---")
                    st.header("Advanced NLP Features")
                    feature_col1, feature_col2, feature_col3 = st.columns(3)

                    with feature_col1:
                        st.subheader("Emotion Analysis")
                        st.info("Detected emotions: Neutral (45%), Happy (30%), Focused (25%)")
                        emotion_data = pd.DataFrame(
                            {
                                "Emotion": ["Neutral", "Happy", "Focused", "Surprised"],
                                "Percentage": [45, 30, 25, 15],
                            }
                        )
                        fig_emotion = px.pie(
                            emotion_data,
                            values="Percentage",
                            names="Emotion",
                            title="Emotion Distribution",
                        )
                        st.plotly_chart(fig_emotion, width="stretch")

                    with feature_col2:
                        st.subheader("Scene Classification")
                        st.info("Scene type: Indoor Office Environment")
                        scene_data = pd.DataFrame(
                            {
                                "Scene Type": [
                                    "Office",
                                    "Meeting Room",
                                    "Kitchen",
                                    "Living Room",
                                ],
                                "Confidence": [0.89, 0.65, 0.23, 0.18],
                            }
                        )
                        fig_scene = px.bar(
                            scene_data,
                            x="Scene Type",
                            y="Confidence",
                            title="Scene Classification Confidence",
                        )
                        st.plotly_chart(fig_scene, width="stretch")

                    with feature_col3:
                        st.subheader("Activity Recognition")
                        st.info("Primary activity: Collaborative Work")
                        activity_data = pd.DataFrame(
                            {
                                "Activity": ["Meeting", "Discussion", "Presentation", "Break"],
                                "Duration": [45, 25, 20, 10],
                            }
                        )
                        fig_activity = px.bar(
                            activity_data,
                            x="Activity",
                            y="Duration",
                            title="Activity Duration (seconds)",
                        )
                        st.plotly_chart(fig_activity, width="stretch")

                else:
                    st.info("Please upload a video file above to see NLP analysis results.")
                    st.markdown("---")
                    st.subheader("Available NLP Features")
                    placeholder_features = [
                        "**Video Summarization** - Generate comprehensive summaries of video content",
                        "**Video Captioning** - Frame-by-frame natural language descriptions",
                        "**Semantic Search** - Find specific content using natural language queries",
                        "**Action Anticipation** - Predict future actions and activities",
                        "**Emotion Analysis** - Detect and analyze emotional states",
                        "**Scene Classification** - Identify and categorize different scene types",
                        "**Activity Recognition** - Recognize and track various activities",
                    ]
                    for feature in placeholder_features:
                        st.markdown(feature)

        else:
            st.info("Please upload a video file first to see the analysis results.")

    #--------------------------------
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, cleaning up...")
        cleanup_temp_videos()
    except Exception as e:
        print(f"Error occurred: {e}")
        cleanup_temp_videos()
        raise
    finally:
        cleanup_temp_videos()