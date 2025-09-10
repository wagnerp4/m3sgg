"""Video processing functions for the M3SGG Streamlit application.

This module contains functions for video validation, conversion, and processing
with scene graph generation and bounding box visualization.
"""

import os
from typing import Any, Dict, Tuple

import cv2
import streamlit as st

from scripts.apps.utils.display.processors import StreamlitVideoProcessor


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
            except OSError:
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
    video_path: str, model_path: str, max_frames: int = 30
) -> Tuple[bool, str]:
    """Create a new video file with bounding boxes drawn on frames

    :param video_path: Path to input video file
    :type video_path: str
    :param model_path: Path to model checkpoint
    :type model_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :return: Tuple of (success, output_path) where success is bool and output_path is str
    :rtype: Tuple[bool, str]
    """
    try:
        import tempfile
        import os

        # Create project-local temp directory
        temp_dir = os.path.join("data", "temp_vid")
        os.makedirs(temp_dir, exist_ok=True)

        # Create temporary file in project temp directory
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4", dir=temp_dir
        ) as tmp_file:
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
                    print(
                        f"Frame {frame_count}: Original shape: {frame.shape}, Processed shape: {frame_with_boxes.shape}"
                    )
                    print(
                        f"Frame {frame_count}: Entry keys: {list(entry.keys()) if entry else 'None'}"
                    )
                    if "boxes" in entry and entry["boxes"] is not None:
                        print(f"Frame {frame_count}: Found {len(entry['boxes'])} boxes")
            else:
                frame_with_boxes = frame
                if frame_count < 3:
                    print(f"Frame {frame_count}: No entry data, using original frame")

            # Write frame (ignore write_success on Windows as it often returns False incorrectly)
            out.write(frame_with_boxes)
            # Note: On Windows, out.write() often returns False even when successful
            # We'll validate the final video file instead
            frame_count += 1

        cap.release()
        out.release()

        # Validate the created video
        if validate_video_file(output_path):
            file_size = os.path.getsize(output_path)
            print(
                f"Bounding box video created successfully: {output_path}, size: {file_size} bytes"
            )
            return True, output_path
        else:
            print(
                f"Bounding box video file created but validation failed: {output_path}"
            )
            return False, ""

    except Exception as e:
        print(f"Error creating bounding box video: {e}")
        import traceback

        traceback.print_exc()
        return False, ""


def create_processed_video_with_scene_graph(
    video_path: str, model_path: str, max_frames: int = 30
) -> Tuple[bool, str]:
    """Create a new video file with scene graph overlay drawn on frames using simplified approach

    :param video_path: Path to input video file
    :type video_path: str
    :param model_path: Path to model checkpoint
    :type model_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :return: Tuple of (success, output_path) where success is bool and output_path is str
    :rtype: Tuple[bool, str]
    """
    try:
        import cv2
        import tempfile
        import os

        # Create project-local temp directory
        temp_dir = os.path.join("data", "temp_vid")
        os.makedirs(temp_dir, exist_ok=True)

        # Create temporary file in project temp directory
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4", dir=temp_dir
        ) as tmp_file:
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
        print(
            f"Video properties: {fps} FPS, {width}x{height}, {total_frames} total frames"
        )

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
            except Exception:
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
                cv2.putText(
                    frame_with_sg,
                    f"Scene Graph Frame {frame_count} (No Model Data)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Draw some simple shapes to represent objects and relationships
                cv2.rectangle(frame_with_sg, (50, 50), (150, 150), (255, 0, 0), 2)
                cv2.putText(
                    frame_with_sg,
                    "Object 1",
                    (55, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

                cv2.rectangle(
                    frame_with_sg, (w - 150, 50), (w - 50, 150), (0, 0, 255), 2
                )
                cv2.putText(
                    frame_with_sg,
                    "Object 2",
                    (w - 145, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                cv2.line(frame_with_sg, (150, 100), (w - 150, 100), (0, 255, 255), 2)
                cv2.putText(
                    frame_with_sg,
                    "relationship",
                    (w // 2 - 50, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )

                cv2.circle(frame_with_sg, (w // 2, h - 100), 30, (255, 255, 0), 2)
                cv2.putText(
                    frame_with_sg,
                    "Node",
                    (w // 2 - 20, h - 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )

            # Write processed frame
            out.write(frame_with_sg)
            processed_frames += 1
            frame_count += 1

        # Release resources
        cap.release()
        out.release()

        print(
            "✅ Video processing completed: {} frames processed".format(
                processed_frames
            )
        )

        # Verify output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print("✅ Output file created: {:,} bytes".format(file_size))

            # Test if output video is readable
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                test_cap.release()
                if ret:
                    print(f"✅ Output video is readable: shape={test_frame.shape}")
                    return True, output_path
                else:
                    print("❌ Output video created but not readable")
                    return False, ""
            else:
                print("❌ Output video created but cannot be opened")
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


def process_video_with_sgg(
    video_path: str,
    model_path: str,
    max_frames: int = 30,
    init_progress_callback=None,
    frame_progress_callback=None,
) -> Dict[str, Any]:
    """Process video using scene graph generation

    :param video_path: Path to input video file
    :type video_path: str
    :param model_path: Path to model checkpoint
    :type model_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :param init_progress_callback: Callback for initialization progress, defaults to None
    :type init_progress_callback: callable, optional
    :param frame_progress_callback: Callback for frame processing progress, defaults to None
    :type frame_progress_callback: callable, optional
    :return: Dictionary containing processing results
    :rtype: Dict[str, Any]
    """
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
                    frame_progress_callback(
                        current_progress,
                        max_frames,
                        f"Frame {frame_count + 1}/{max_frames} - {stage}",
                        f"Processing frame {frame_count + 1}: {stage}",
                    )

            # Process frame with SGG
            frame_results, entry, pred = processor.process_frame(
                frame, frame_progress_callback_internal
            )

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
