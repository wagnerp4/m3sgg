import sys
from pathlib import Path

# Add project root to path before importing modules
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st

from scripts.apps.components.dark_mode import apply_theme_styles
from scripts.apps.components.sidebar import render_sidebar
from scripts.apps.components.video_analysis import render_video_analysis
from scripts.apps.components.graph_processing import render_graph_processing_button
from scripts.apps.components.results_tabs import render_results_tabs
from scripts.apps.components.video_processing import (
    convert_video_for_browser,
    validate_video_file,
    create_processed_video_with_bboxes,
    create_processed_video_with_scene_graph,
    process_video_with_sgg,
    cleanup_temp_videos,
)
from scripts.apps.components.model_utils import find_available_checkpoints

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
    page_title="M3Sgg", page_icon="", layout="wide", initial_sidebar_state="expanded"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
apply_theme_styles()


def main():
    # Header
    st.markdown(
        '<h1 class="main-header"> M3SGG</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        model_path, max_frames = render_sidebar(
            {
                "MODEL_DETECTOR_AVAILABLE": MODEL_DETECTOR_AVAILABLE,
                "get_model_info_from_checkpoint": get_model_info_from_checkpoint
                if MODEL_DETECTOR_AVAILABLE
                else None,
                "find_available_checkpoints": find_available_checkpoints,
            }
        )

    # Video Analysis
    uploaded_file = render_video_analysis({})

    # Graph Processing Button
    render_graph_processing_button(
        uploaded_file,
        model_path,
        max_frames,
        {
            "process_video_with_sgg": process_video_with_sgg,
            "create_processed_video_with_bboxes": create_processed_video_with_bboxes,
            "create_processed_video_with_scene_graph": create_processed_video_with_scene_graph,
            "validate_video_file": validate_video_file,
            "convert_video_for_browser": convert_video_for_browser,
        },
    )

    # Result View Tabs
    render_results_tabs(
        {
            "CHAT_INTERFACE_AVAILABLE": CHAT_INTERFACE_AVAILABLE,
            "SceneGraphChatInterface": SceneGraphChatInterface
            if CHAT_INTERFACE_AVAILABLE
            else None,
            "validate_video_file": validate_video_file,
            "convert_video_for_browser": convert_video_for_browser,
        }
    )

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
