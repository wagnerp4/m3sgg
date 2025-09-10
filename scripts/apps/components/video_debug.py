# Made to be inserted into streamlit.py to debug different codecs, drawers, and video display methods.

# --------------------------------
# Video Display Debugging
# # Debug Component - Simple Video Processing Test
# st.markdown("### 🔧 Debug: Simple Video Processing Test")
# with st.expander("Test Simple Video Processing with Generated Videos", expanded=True):
#     st.markdown("**Testing simple video processing with pre-generated videos**")
#     st.markdown("""
#     **What this tests:**
#     - **Basic Video Processing**: Just copies frames without any modifications
#     - **Model Video Processing**: Runs model detection but saves original frames (no drawing)
#     - **Video Display**: Tests if the processed videos display properly in Streamlit

#     **Expected Results:**
#     - Both videos should be identical to the original 0MK2C.mp4
#     - Videos should display properly in all display methods
#     - File sizes should be similar to original
#     """)

#     # Show current video paths being used
#     st.markdown("**Current Video Paths:**")
#     st.code("""
#         Debug Component Videos:
#         - Original: 0MK2C.mp4
#         - Basic: simple_basic_full_0MK2C.mp4
#         - Model: simple_model_full_0MK2C.mp4
#         Main Display Videos:
#         - Bbox Video: simple_model_full_0MK2C.mp4
#         - Scene Graph Video: simple_basic_full_0MK2C.mp4
#     """)

#     # Use the generated simple videos (full length)
#     original_video = "0MK2C.mp4"
#     basic_video = "debug_bbox_c3aec751.mp4"
#     model_video = "simple_model_full_0MK2C.mp4"


#     # Check if videos exist
#     original_exists = os.path.exists(original_video)
#     basic_exists = os.path.exists(basic_video)
#     model_exists = os.path.exists(model_video)

#     # Show which videos are being used with detailed info
#     if original_exists:
#         original_size = os.path.getsize(original_video)
#         # Get frame count
#         import cv2
#         cap = cv2.VideoCapture(original_video)
#         original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
#         cap.release()
#         st.success(f"🎯 **Original Video**: `{original_video}` ({original_size:,} bytes, {original_frames} frames)")
#     else:
#         st.error(f"❌ **Original Video**: `{original_video}` not found")

#     if basic_exists:
#         basic_size = os.path.getsize(basic_video)
#         # Get frame count
#         cap = cv2.VideoCapture(basic_video)
#         basic_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
#         cap.release()
#         st.success(f"🎯 **Basic Video**: `{basic_video}` ({basic_size:,} bytes, {basic_frames} frames) - Simple frame copy")
#     else:
#         st.error(f"❌ **Basic Video**: `{basic_video}` not found")

#     if model_exists:
#         model_size = os.path.getsize(model_video)
#         # Get frame count
#         cap = cv2.VideoCapture(model_video)
#         model_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
#         cap.release()
#         st.success(f"🎯 **Model Video**: `{model_video}` ({model_size:,} bytes, {model_frames} frames) - Model processing without drawing")
#     else:
#         st.error(f"❌ **Model Video**: `{model_video}` not found")

#     # Additional debug info
#     st.markdown("---")
#     st.markdown("**Debug Information:**")
#     st.code(f"""
#     Video Paths in Debug Component:
#     - original_video = "{original_video}"
#     - basic_video = "{basic_video}"
#     - model_video = "{model_video}"

#     File Existence Check:
#     - Original exists: {original_exists}
#     - Basic exists: {basic_exists}
#     - Model exists: {model_exists}
#             """)

#     # Add basic processing function
#     st.markdown("---")
#     st.markdown("**Generate New Basic Processing Video**")

#     # Import the simple processing functions
#     try:
#         from simple_drawing_methods import simple_process_video_basic, simple_process_video_with_bboxes

#         # Create buttons for different processing methods
#         col_btn1, col_btn2 = st.columns(2)

#         with col_btn1:
#             # Button to generate new basic video
#             if st.button("🔄 Generate Basic Video", help="Create a new basic video by copying frames from original"):
#                 if original_exists:
#                     # Generate new basic video with timestamp
#                     import time
#                     timestamp = int(time.time())
#                     new_basic_video = f"debug_basic_{timestamp}.mp4"

#                     with st.spinner("Generating new basic video..."):
#                         success = simple_process_video_basic(original_video, new_basic_video, max_frames=30)

#                         if success:
#                             st.success(f"✅ New basic video generated: {new_basic_video}")
#                             # Store in session state for central display
#                             st.session_state["current_basic_video"] = new_basic_video
#                             st.rerun()
#                         else:
#                             st.error("❌ Failed to generate new basic video")
#                 else:
#                     st.error("❌ Original video not found - cannot generate basic video")

#         with col_btn2:
#             # Button to generate bounding box video
#             if st.button("📦 Generate Bbox Video", help="Create a new video with bounding boxes drawn"):
#                 if original_exists:
#                     # Generate new bbox video with timestamp
#                     import time
#                     timestamp = int(time.time())
#                     new_bbox_video = f"debug_bbox_{timestamp}.mp4"

#                     with st.spinner("Generating bounding box video..."):
#                         success = simple_process_video_with_bboxes(original_video, new_bbox_video, max_frames=30)

#                         if success:
#                             st.success(f"✅ New bbox video generated: {new_bbox_video}")
#                             # Store in session state for central display
#                             st.session_state["current_bbox_video"] = new_bbox_video
#                             st.rerun()
#                         else:
#                             st.error("❌ Failed to generate bounding box video")
#                 else:
#                     st.error("❌ Original video not found - cannot generate bbox video")

#     except ImportError as e:
#         st.error(f"❌ Could not import processing functions: {e}")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.markdown("**Original Video (0MK2C.mp4)**")
#         if original_exists:
#             st.success("✅ Original video found")
#             st.write(f"File size: {original_size:,} bytes")

#             # Method 1: Direct st.video
#             st.markdown("**Method 1: Direct st.video**")
#             try:
#                 st.video(original_video)
#                 st.success("✅ Original video works!")
#             except Exception as e:
#                 st.error(f"❌ Original video failed: {e}")

#             # Method 2: st.video with bytes
#             st.markdown("**Method 2: st.video with bytes**")
#             try:
#                 with open(original_video, "rb") as f:
#                     video_bytes = f.read()
#                 st.video(video_bytes)
#                 st.success("✅ Original video with bytes works!")
#             except Exception as e:
#                 st.error(f"❌ Original video with bytes failed: {e}")
#         else:
#             st.error("❌ Original video not found")

#     with col2:
#         st.markdown("**Central Display - All Results**")
#         st.caption("This column shows the most recently generated videos from the buttons above")

#         # Display current basic video if available
#         current_basic = st.session_state.get("current_basic_video")
#         if current_basic and os.path.exists(current_basic):
#             st.markdown("**🔄 Basic Video (Frame Copy)**")
#             st.success(f"✅ Basic video: {current_basic}")
#             basic_size = os.path.getsize(current_basic)
#             cap = cv2.VideoCapture(current_basic)
#             basic_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
#             cap.release()
#             st.write(f"File size: {basic_size:,} bytes, Frames: {basic_frames}")

#             # Display the video
#             try:
#                 st.video(current_basic)
#                 st.success("✅ Basic video displays correctly!")
#             except Exception as e:
#                 st.error(f"❌ Basic video display failed: {e}")

#         # Display current bbox video if available
#         current_bbox = st.session_state.get("current_bbox_video")
#         if current_bbox and os.path.exists(current_bbox):
#             st.markdown("**📦 Bounding Box Video**")
#             st.success(f"✅ Bbox video: {current_bbox}")
#             bbox_size = os.path.getsize(current_bbox)
#             cap = cv2.VideoCapture(current_bbox)
#             bbox_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
#             cap.release()
#             st.write(f"File size: {bbox_size:,} bytes, Frames: {bbox_frames}")

#             # Display the video
#             try:
#                 st.video(current_bbox)
#                 st.success("✅ Bbox video displays correctly!")
#             except Exception as e:
#                 st.error(f"❌ Bbox video display failed: {e}")

#         # If no current videos, show default
#         if not current_basic and not current_bbox:
#             st.info("Click the buttons above to generate videos and see them here")
#             if basic_exists:
#                 st.markdown("**Default Basic Video**")
#                 st.write(f"File size: {basic_size:,} bytes")
#                 try:
#                     st.video(basic_video)
#                 except Exception as e:
#                     st.error(f"❌ Video display failed: {e}")

#             # Method 1: Direct st.video
#             st.markdown("**Method 1: Direct st.video**")
#             try:
#                 st.video(basic_video)
#                 st.success("✅ Direct st.video works!")
#             except Exception as e:
#                 st.error(f"❌ Direct st.video failed: {e}")

#             # Method 2: st.video with bytes
#             st.markdown("**Method 2: st.video with bytes**")
#             try:
#                 with open(basic_video, "rb") as f:
#                     video_bytes = f.read()
#                 st.video(video_bytes)
#                 st.success("✅ st.video with bytes works!")
#             except Exception as e:
#                 st.error(f"❌ st.video with bytes failed: {e}")

#             # Method 3: HTML5 video
#             st.markdown("**Method 3: HTML5 video**")
#             try:
#                 import base64
#                 video_html = f"""
#                 <div style="text-align: center;">
#                     <video width="100%" height="200" controls autoplay muted loop>
#                         <source src="data:video/mp4;base64,{base64.b64encode(open(basic_video, 'rb').read()).decode()}" type="video/mp4">
#                         Your browser does not support the video tag.
#                     </video>
#                 </div>
#                 """
#                 st.markdown(video_html, unsafe_allow_html=True)
#                 st.success("✅ HTML5 video works!")
#             except Exception as e:
#                 st.error(f"❌ HTML5 video failed: {e}")

#             # Download button
#             with open(basic_video, "rb") as f:
#                 st.download_button(
#                     label="Download Basic Video",
#                     data=f.read(),
#                     file_name=basic_video,
#                     mime="video/mp4"
#                 )
#         else:
#             st.error(f"❌ Basic video not found: {basic_video}")

#     with col3:
#         st.markdown("**Model Processing Video**")
#         if model_exists:
#             st.success(f"✅ Model video exists: {model_video}")
#             st.write(f"File size: {model_size:,} bytes")

#             # Method 1: Direct st.video
#             st.markdown("**Method 1: Direct st.video**")
#             try:
#                 st.video(model_video)
#                 st.success("✅ Direct st.video works!")
#             except Exception as e:
#                 st.error(f"❌ Direct st.video failed: {e}")

#             # Method 2: st.video with bytes
#             st.markdown("**Method 2: st.video with bytes**")
#             try:
#                 with open(model_video, "rb") as f:
#                     video_bytes = f.read()
#                 st.video(video_bytes)
#                 st.success("✅ st.video with bytes works!")
#             except Exception as e:
#                 st.error(f"❌ st.video with bytes failed: {e}")

#             # Method 3: HTML5 video
#             st.markdown("**Method 3: HTML5 video**")
#             try:
#                 import base64
#                 video_html = f"""
#                 <div style="text-align: center;">
#                     <video width="100%" height="200" controls autoplay muted loop>
#                         <source src="data:video/mp4;base64,{base64.b64encode(open(model_video, 'rb').read()).decode()}" type="video/mp4">
#                         Your browser does not support the video tag.
#                     </video>
#                 </div>
#                 """
#                 st.markdown(video_html, unsafe_allow_html=True)
#                 st.success("✅ HTML5 video works!")
#             except Exception as e:
#                 st.error(f"❌ HTML5 video failed: {e}")

#             # Download button
#             with open(model_video, "rb") as f:
#                 st.download_button(
#                     label="Download Model Video",
#                     data=f.read(),
#                     file_name=model_video,
#                     mime="video/mp4"
#                 )
#         else:
#             st.error(f"❌ Model video not found: {model_video}")

#     # Method 4: streamlit-player (full width)
#     st.markdown("**Method 4: streamlit-player**")
#     if STREAMLIT_PLAYER_AVAILABLE:
#         col4, col5 = st.columns(2)
#         with col4:
#             st.markdown("**Basic Video (streamlit-player)**")
#             if basic_exists:
#                 try:
#                     file_url = f"file:///{os.path.abspath(basic_video).replace(os.sep, '/')}"
#                     st_player(file_url)
#                     st.success("✅ streamlit-player works!")
#                 except Exception as e:
#                     st.error(f"❌ streamlit-player failed: {e}")
#         with col5:
#             st.markdown("**Model Video (streamlit-player)**")
#             if model_exists:
#                 try:
#                     file_url = f"file:///{os.path.abspath(model_video).replace(os.sep, '/')}"
#                     st_player(file_url)
#                     st.success("✅ streamlit-player works!")
#                 except Exception as e:
#                     st.error(f"❌ streamlit-player failed: {e}")
#     else:
#         st.info("ℹ️ streamlit-player not available - install with: pip install streamlit-player")

# # Debug Component - Video Display
# st.markdown("### 🔧 Debug: Video Files")
# with st.expander("Debug Video Display", expanded=False):
#     st.markdown("**Debugging video display issues - showing debug files from project root:**")

#     # Look for debug video files in project root (including browser-friendly versions)
#     debug_files = []
#     for file in os.listdir("."):
#         if file.startswith("debug_") and file.endswith(".mp4"):
#             debug_files.append(file)

#     # Prioritize browser-friendly versions
#     browser_friendly_files = [f for f in debug_files if "browser_friendly" in f]
#     if browser_friendly_files:
#         debug_files = browser_friendly_files + [f for f in debug_files if "browser_friendly" not in f]

#     if debug_files:
#         st.info(f"Found {len(debug_files)} debug video files in project root")

#         # Group files by type (prioritize browser-friendly versions)
#         bbox_files = [f for f in debug_files if f.startswith("debug_bbox_")]
#         sg_files = [f for f in debug_files if f.startswith("debug_scene_graph_")]

#         # Sort to prioritize browser-friendly versions
#         bbox_files.sort(key=lambda x: ("browser_friendly" not in x, x))
#         sg_files.sort(key=lambda x: ("browser_friendly" not in x, x))

#         debug_col1, debug_col2 = st.columns(2)

#         with debug_col1:
#             st.subheader("Debug: Bounding Box Videos")
#             if bbox_files:
#                 # Show the most recent bbox file
#                 latest_bbox = max(bbox_files, key=lambda x: os.path.getctime(x))
#                 st.info(f"Latest bbox video: {latest_bbox}")

#                 if os.path.exists(latest_bbox):
#                     st.success("✅ Bbox video file exists!")

#                     # Use HTML video element for better control
#                     video_html = f"""
#                     <div style="text-align: center;">
#                         <video width="100%" height="300" controls autoplay muted loop>
#                             <source src="data:video/mp4;base64,{base64.b64encode(open(latest_bbox, 'rb').read()).decode()}" type="video/mp4">
#                             Your browser does not support the video tag.
#                         </video>
#                     </div>
#                     """
#                     st.markdown(video_html, unsafe_allow_html=True)

#                     # Alternative: File download button
#                     with open(latest_bbox, "rb") as f:
#                         video_bytes = f.read()
#                     st.download_button(
#                         label="📥 Download Bbox Video",
#                         data=video_bytes,
#                         file_name=latest_bbox,
#                         mime="video/mp4"
#                     )

#                     # Video info
#                     st.info(f"📁 File: {latest_bbox}")
#                     st.info(f"📏 Size: {os.path.getsize(latest_bbox):,} bytes")

#                     # Alternative: Use st.video with bytes for comparison
#                     st.markdown("**Alternative Display (st.video with bytes):**")
#                     try:
#                         st.video(video_bytes)
#                         st.success("✅ st.video with bytes works!")
#                     except Exception as e:
#                         st.error(f"❌ st.video with bytes failed: {e}")

#                     # Third approach: Use streamlit-player if available
#                     if STREAMLIT_PLAYER_AVAILABLE:
#                         st.markdown("**Third Display (streamlit-player):**")
#                         try:
#                             # Convert to file:// URL for local file
#                             file_url = f"file:///{os.path.abspath(latest_bbox).replace(os.sep, '/')}"
#                             st_player(file_url)
#                             st.success("✅ streamlit-player works!")
#                         except Exception as e:
#                             st.error(f"❌ streamlit-player failed: {e}")
#                     else:
#                         st.info("ℹ️ streamlit-player not available - install with: pip install streamlit-player")
#                 else:
#                     st.error("❌ Bbox video file does not exist!")

#                 # Show all bbox files
#                 if len(bbox_files) > 1:
#                     st.markdown("**All bbox files:**")
#                     for file in sorted(bbox_files):
#                         file_size = os.path.getsize(file) if os.path.exists(file) else 0
#                         st.text(f"  • {file} ({file_size} bytes)")
#             else:
#                 st.info("No bbox video files found")

#         with debug_col2:
#             st.subheader("Debug: Scene Graph Videos")
#             if sg_files:
#                 # Show the most recent scene graph file
#                 latest_sg = max(sg_files, key=lambda x: os.path.getctime(x))
#                 st.info(f"Latest scene graph video: {latest_sg}")

#                 if os.path.exists(latest_sg):
#                     st.success("✅ Scene graph video file exists!")

#                     # Use HTML video element for better control
#                     video_html = f"""
#                     <div style="text-align: center;">
#                         <video width="100%" height="300" controls autoplay muted loop>
#                             <source src="data:video/mp4;base64,{base64.b64encode(open(latest_sg, 'rb').read()).decode()}" type="video/mp4">
#                             Your browser does not support the video tag.
#                         </video>
#                     </div>
#                     """
#                     st.markdown(video_html, unsafe_allow_html=True)

#                     # Alternative: File download button
#                     with open(latest_sg, "rb") as f:
#                         video_bytes = f.read()
#                     st.download_button(
#                         label="📥 Download Scene Graph Video",
#                         data=video_bytes,
#                         file_name=latest_sg,
#                         mime="video/mp4"
#                     )

#                     # Video info
#                     st.info(f"📁 File: {latest_sg}")
#                     st.info(f"📏 Size: {os.path.getsize(latest_sg):,} bytes")

#                     # Alternative: Use st.video with bytes for comparison
#                     st.markdown("**Alternative Display (st.video with bytes):**")
#                     try:
#                         st.video(video_bytes)
#                         st.success("✅ st.video with bytes works!")
#                     except Exception as e:
#                         st.error(f"❌ st.video with bytes failed: {e}")

#                     # Third approach: Use streamlit-player if available
#                     if STREAMLIT_PLAYER_AVAILABLE:
#                         st.markdown("**Third Display (streamlit-player):**")
#                         try:
#                             # Convert to file:// URL for local file
#                             file_url = f"file:///{os.path.abspath(latest_sg).replace(os.sep, '/')}"
#                             st_player(file_url)
#                             st.success("✅ streamlit-player works!")
#                         except Exception as e:
#                             st.error(f"❌ streamlit-player failed: {e}")
#                     else:
#                         st.info("ℹ️ streamlit-player not available - install with: pip install streamlit-player")
#                 else:
#                     st.error("❌ Scene graph video file does not exist!")

#                 # Show all scene graph files
#                 if len(sg_files) > 1:
#                     st.markdown("**All scene graph files:**")
#                     for file in sorted(sg_files):
#                         file_size = os.path.getsize(file) if os.path.exists(file) else 0
#                         st.text(f"  • {file} ({file_size} bytes)")
#             else:
#                 st.info("No scene graph video files found")
#     else:
#         st.info("No debug video files found in project root")
#         st.markdown("**Expected files:** `debug_bbox_*.mp4` and `debug_scene_graph_*.mp4`")

#     # Installation instructions for streamlit-player
#     if not STREAMLIT_PLAYER_AVAILABLE:
#         st.markdown("---")
#         st.subheader("🔧 Enhanced Video Display")
#         st.info("For better video debugging, install streamlit-player:")
#         st.code("pip install streamlit-player", language="bash")
#         st.markdown("This will enable additional video display methods in the debug component.")

#     # Additional debug info
#     st.markdown("---")
#     st.subheader("Debug: File System Info")

#     # Show temp file info
#     if "debug_temp_path" in st.session_state:
#         temp_path = st.session_state["debug_temp_path"]
#         st.markdown("**Original Temp File:**")
#         st.text(f"Path: {temp_path}")
#         st.text(f"Exists: {os.path.exists(temp_path)}")
#         if os.path.exists(temp_path):
#             st.text(f"Size: {os.path.getsize(temp_path)} bytes")
#             st.text(f"Readable: {os.access(temp_path, os.R_OK)}")

#     if "bbox_video_path" in st.session_state and "scene_graph_video_path" in st.session_state:
#         bbox_path = st.session_state["bbox_video_path"]
#         sg_path = st.session_state["scene_graph_video_path"]

#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("**Bbox Video:**")
#             st.text(f"Path: {bbox_path}")
#             st.text(f"Exists: {os.path.exists(bbox_path)}")
#             if os.path.exists(bbox_path):
#                 st.text(f"Size: {os.path.getsize(bbox_path)} bytes")
#                 st.text(f"Readable: {os.access(bbox_path, os.R_OK)}")

#         with col2:
#             st.markdown("**Scene Graph Video:**")
#             st.text(f"Path: {sg_path}")
#             st.text(f"Exists: {os.path.exists(sg_path)}")
#             if os.path.exists(sg_path):
#                 st.text(f"Size: {os.path.getsize(sg_path)} bytes")
#                 st.text(f"Readable: {os.access(sg_path, os.R_OK)}")

#     # Add cleanup button for debugging
#     st.markdown("---")
#     st.subheader("Debug: Cleanup")
#     if st.button("🗑️ Clean Up Debug Files", help="Remove all debug files and reset session state"):
#         cleanup_paths = []

#         # Add temp files from session state
#         if "debug_temp_path" in st.session_state:
#             cleanup_paths.append(st.session_state["debug_temp_path"])
#         if "bbox_video_path" in st.session_state:
#             cleanup_paths.append(st.session_state["bbox_video_path"])
#         if "scene_graph_video_path" in st.session_state:
#             cleanup_paths.append(st.session_state["scene_graph_video_path"])

#         # Add debug files from project root
#         for file in os.listdir("."):
#             if file.startswith("debug_") and file.endswith(".mp4"):
#                 cleanup_paths.append(file)

#         cleaned_count = 0
#         for path in cleanup_paths:
#             try:
#                 if os.path.exists(path):
#                     os.unlink(path)
#                     cleaned_count += 1
#                     st.info(f"Deleted: {path}")
#             except Exception as e:
#                 st.error(f"Failed to delete {path}: {e}")

#         # Clear session state
#         keys_to_clear = ["debug_temp_path", "bbox_video_path", "scene_graph_video_path", "results"]
#         for key in keys_to_clear:
#             if key in st.session_state:
#                 del st.session_state[key]

#         st.success(f"Cleaned up {cleaned_count} debug files and reset session state")
#         st.rerun()
# st.markdown("---")
