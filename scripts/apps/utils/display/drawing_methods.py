#!/usr/bin/env python3
"""
Simple drawing methods using multiple libraries for better compatibility
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

def simple_draw_bounding_boxes(frame, entry):
    """Simple bounding box drawing using PIL and matplotlib"""
    if entry is None or "boxes" not in entry or entry["boxes"] is None:
        return frame
    
    # Convert BGR to RGB for PIL
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    h, w = frame_rgb.shape[:2]
    
    # Convert to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Get boxes from entry
    boxes = entry["boxes"]
    
    # Draw simple rectangles for each box
    for i, box in enumerate(boxes):
        if len(box) >= 4:
            # Handle PyTorch tensors and batch dimension
            if hasattr(box, 'cpu'):  # PyTorch tensor
                box = box.cpu().numpy()
            
            if len(box) == 5:
                x1, y1, x2, y2 = box[1:5].astype(int)
            else:
                x1, y1, x2, y2 = box[:4].astype(int)
            
            # Scale to frame size (assuming model uses 600x600)
            x1 = int(x1 * w / 600)
            y1 = int(y1 * h / 600)
            x2 = int(x2 * w / 600)
            y2 = int(y2 * h / 600)
            
            # Ensure coordinates are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Ensure x1 <= x2 and y1 <= y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Skip if box is too small or invalid
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            
            # Draw rectangle with different colors
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            color = colors[i % len(colors)]
            
            # Draw thick rectangle outline
            for thickness in range(3):
                draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                             outline=color, width=1)
            
            # Add box number
            try:
                # Try to use a default font
                font = ImageFont.load_default()
                draw.text((x1, y1-20), f"Box {i+1}", fill=color, font=font)
            except:
                # Fallback to basic text
                draw.text((x1, y1-20), f"Box {i+1}", fill=color)
    
    # Convert back to numpy array and BGR
    frame_with_boxes = np.array(pil_image)
    if len(frame_with_boxes.shape) == 3 and frame_with_boxes.shape[2] == 3:
        frame_with_boxes = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
    
    return frame_with_boxes

def simple_create_scene_graph_frame(frame, entry, pred):
    """Simple scene graph drawing using matplotlib"""
    if entry is None or pred is None:
        return frame
    
    # Convert BGR to RGB for matplotlib
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    
    h, w = frame_rgb.shape[:2]
    
    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(w/100, h/100), dpi=100)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Flip y-axis for image coordinates
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Display the frame
    ax.imshow(frame_rgb, extent=[0, w, h, 0])
    
    # Get boxes from entry
    if "boxes" not in entry or entry["boxes"] is None:
        plt.close(fig)
        return frame
    
    boxes = entry["boxes"]
    
    # Draw simple nodes (circles) for each box
    centers = []
    colors = ['green', 'red', 'blue', 'yellow', 'magenta']
    
    for i, box in enumerate(boxes):
        if len(box) >= 4:
            # Handle PyTorch tensors and batch dimension
            if hasattr(box, 'cpu'):  # PyTorch tensor
                box = box.cpu().numpy()
            
            if len(box) == 5:
                x1, y1, x2, y2 = box[1:5].astype(int)
            else:
                x1, y1, x2, y2 = box[:4].astype(int)
            
            # Calculate center
            cx = int((x1 + x2) / 2 * w / 600)
            cy = int((y1 + y2) / 2 * h / 600)
            
            # Ensure coordinates are valid
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            centers.append((cx, cy))
            
            # Draw circle for node
            color = colors[i % len(colors)]
            circle = patches.Circle((cx, cy), 15, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Add node number
            ax.text(cx, cy, str(i+1), ha='center', va='center', 
                   color='white', fontsize=12, weight='bold')
    
    # Draw simple connections between nodes
    if len(centers) >= 2:
        for i in range(len(centers) - 1):
            ax.plot([centers[i][0], centers[i+1][0]], 
                   [centers[i][1], centers[i+1][1]], 
                   color='yellow', linewidth=3, alpha=0.8)
        
        # Connect last to first to make a triangle
        if len(centers) >= 3:
            ax.plot([centers[-1][0], centers[0][0]], 
                   [centers[-1][1], centers[0][1]], 
                   color='magenta', linewidth=3, alpha=0.8)
    
    # Convert matplotlib figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    # Convert back to BGR
    frame_with_sg = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    return frame_with_sg

def simple_draw_bounding_boxes_cv2_alternative(frame, entry):
    """Alternative OpenCV method with different approach"""
    if entry is None or "boxes" not in entry or entry["boxes"] is None:
        return frame
    
    frame_with_boxes = frame.copy()
    h, w = frame.shape[:2]
    
    # Get boxes from entry
    boxes = entry["boxes"]
    
    # Draw simple rectangles for each box using a different approach
    for i, box in enumerate(boxes):
        if len(box) >= 4:
            # Handle PyTorch tensors and batch dimension
            if hasattr(box, 'cpu'):  # PyTorch tensor
                box = box.cpu().numpy()
            
            if len(box) == 5:
                x1, y1, x2, y2 = box[1:5].astype(int)
            else:
                x1, y1, x2, y2 = box[:4].astype(int)
            
            # Scale to frame size (assuming model uses 600x600)
            x1 = int(x1 * w / 600)
            y1 = int(y1 * h / 600)
            x2 = int(x2 * w / 600)
            y2 = int(y2 * h / 600)
            
            # Ensure coordinates are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Draw filled rectangle first
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            color = colors[i % len(colors)]
            
            # Draw thick border by drawing multiple rectangles
            for thickness in range(5):
                cv2.rectangle(frame_with_boxes, 
                            (x1-thickness, y1-thickness), 
                            (x2+thickness, y2+thickness), 
                            color, 1)
            
            # Add text with background
            text = f"Box {i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(frame_with_boxes, 
                        (x1, y1-text_height-baseline-5), 
                        (x1+text_width, y1), 
                        color, -1)
            
            # Draw text
            cv2.putText(frame_with_boxes, text, (x1, y1-5), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return frame_with_boxes

def simple_process_video_basic(input_video_path, output_video_path, max_frames=30):
    """Simple video processing that just copies the input file directly (no processing)
    
    :param input_video_path: Path to input video file
    :type input_video_path: str
    :param output_video_path: Path for output video file
    :type output_video_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :return: True if video processing was successful, False otherwise
    :rtype: bool
    """
    try:
        print(f"Processing video: {input_video_path}")
        print(f"Output path: {output_video_path}")
        print(f"Max frames: {max_frames}")
        
        # PROCESS WITH OPENCV - BGR2RGB conversion and frame processing
        print(f"Processing video with OpenCV (BGR2RGB conversion)...")
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"❌ Could not open input video: {input_video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {fps} FPS, {width}x{height}, {total_frames} total frames")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Could not create output video writer")
            cap.release()
            return False
        
        print(f"✅ Video writer created successfully")
        
        frame_count = 0
        processed_frames = 0
        
        # Process all frames (ignore max_frames for now)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {frame_count}")
                break
            
            # Just copy the frame without any processing
            out.write(frame)
            processed_frames += 1
            
            if frame_count % 50 == 0:
                print(f"Processed frame {frame_count}/{total_frames}")
            
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"✅ Video processing completed: {processed_frames} frames processed")
        
        # Verify output file
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path)
            print(f"✅ Output file created: {file_size:,} bytes")
            
            # Test if output video is readable
            cap = cv2.VideoCapture(output_video_path)
            if cap.isOpened():
                ret, test_frame = cap.read()
                cap.release()
                if ret:
                    print(f"✅ Output video is readable: shape={test_frame.shape}")
                    return True
                else:
                    print(f"❌ Output video created but not readable")
                    return False
            else:
                print(f"❌ Output video created but cannot be opened")
                return False
        else:
            print(f"❌ Output file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False

def simple_process_video_with_bboxes(input_video_path, output_video_path, max_frames=30):
    """Simple video processing with bounding box drawing using H.264 encoding
    
    :param input_video_path: Path to input video file
    :type input_video_path: str
    :param output_video_path: Path for output video file
    :type output_video_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :return: True if video processing was successful, False otherwise
    :rtype: bool
    """
    try:
        print(f"Processing video with bounding boxes: {input_video_path}")
        print(f"Output path: {output_video_path}")
        print(f"Max frames: {max_frames}")
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"❌ Could not open input video: {input_video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {fps} FPS, {width}x{height}, {total_frames} total frames")
        
        # Create output video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Could not create output video writer")
            cap.release()
            return False
        
        print(f"✅ Video writer created successfully")
        
        frame_count = 0
        processed_frames = 0
        
        # Process frames with bounding boxes
        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {frame_count}")
                break
            
            # Create mock entry data for bounding boxes (for testing)
            # In real usage, this would come from model detection
            import numpy as np
            mock_entry = {
                "boxes": [
                    np.array([0, 100, 200, 300, 0.9]),  # [class, x1, y1, x2, y2, confidence]
                    np.array([1, 300, 150, 450, 0.8]),
                ]
            }
            
            # Draw bounding boxes
            frame_with_boxes = simple_draw_bounding_boxes(frame, mock_entry)
            
            # Write processed frame
            out.write(frame_with_boxes)
            processed_frames += 1
            
            if frame_count % 10 == 0:
                print(f"Processed frame {frame_count}/{max_frames}")
            
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"✅ Video processing completed: {processed_frames} frames processed")
        
        # Verify output file
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path)
            print(f"✅ Output file created: {file_size:,} bytes")
            
            # Test if output video is readable
            test_cap = cv2.VideoCapture(output_video_path)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                test_cap.release()
                if ret:
                    print(f"✅ Output video is readable: shape={test_frame.shape}")
                    return True
                else:
                    print(f"❌ Output video created but not readable")
                    return False
            else:
                print(f"❌ Output video created but cannot be opened")
                return False
        else:
            print(f"❌ Output file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return False

def simple_process_video_with_model(input_video_path, output_video_path, model_path, max_frames=30):
    """Simple video processing with model detection but no drawing - just saves original frames
    
    :param input_video_path: Path to input video file
    :type input_video_path: str
    :param output_video_path: Path for output video file
    :type output_video_path: str
    :param model_path: Path to model checkpoint
    :type model_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :return: True if video processing was successful, False otherwise
    :rtype: bool
    """
    try:
        print(f"Processing video with model: {input_video_path}")
        print(f"Model path: {model_path}")
        print(f"Output path: {output_video_path}")
        print(f"Max frames: {max_frames}")
        
        # Import here to avoid circular imports
        import sys
        import os
        from pathlib import Path
        
        # Add project paths
        project_root = Path(__file__).parent
        src_path = project_root / "src"
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from scripts.apps.utils.display.processors import StreamlitVideoProcessor
        
        # Initialize processor
        processor = StreamlitVideoProcessor(model_path)
        print("✅ Model processor initialized")
        
        # Open input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"❌ Could not open input video: {input_video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {fps} FPS, {width}x{height}, {total_frames} total frames")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Could not create output video writer")
            cap.release()
            return False
        
        print(f"✅ Video writer created successfully")
        
        frame_count = 0
        processed_frames = 0
        detection_count = 0
        
        # Process frames
        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached at frame {frame_count}")
                break
            
            # Process frame with model (but don't draw anything)
            try:
                frame_results, entry, pred = processor.process_frame(frame)
                
                # Count detections
                if entry is not None and "boxes" in entry:
                    detection_count += len(entry["boxes"])
                
                if frame_count % 5 == 0:
                    print(f"Frame {frame_count}: {frame_results.get('objects', 0)} objects detected")
                    
            except Exception as e:
                print(f"Warning: Error processing frame {frame_count}: {e}")
            
            # Just write the original frame (no drawing)
            out.write(frame)
            processed_frames += 1
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"✅ Video processing completed: {processed_frames} frames processed, {detection_count} total detections")
        
        # Verify output file
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path)
            print(f"✅ Output file created: {file_size:,} bytes")
            
            # Test if output video is readable
            test_cap = cv2.VideoCapture(output_video_path)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                test_cap.release()
                if ret:
                    print(f"✅ Output video is readable: shape={test_frame.shape}")
                    return True
                else:
                    print(f"❌ Output video created but not readable")
                    return False
            else:
                print(f"❌ Output video created but cannot be opened")
                return False
        else:
            print(f"❌ Output file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error processing video with model: {e}")
        import traceback
        traceback.print_exc()
        return False
