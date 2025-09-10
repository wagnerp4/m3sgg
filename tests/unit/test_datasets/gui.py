"""
Mock GUI module for testing purposes
"""


class VideoProcessor:
    """Mock VideoProcessor class for testing"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.models_loaded = False
        self.current_video_index = 0
        self.current_frame_index = 0
        self.use_expanded_video_selection = False
        self.gt_video_selection_probability = 0.5
        self.available_video_indices = list(range(10))
        self.dataset = None
        self.manual_navigation = False

    def setup_models(self):
        """Mock model setup method"""
        self.models_loaded = True
        return True

    def setup_dataset_video(self):
        """Mock dataset video setup method"""
        self.dataset = MockDataset()
        return True

    def _initialize_video_cycling(self):
        """Mock video cycling initialization"""
        if self.use_expanded_video_selection:
            self.available_video_indices = list(range(20))
        else:
            self.available_video_indices = list(range(10))

    def get_frame_info(self):
        """Mock frame info method"""
        return {
            "frame_index": self.current_frame_index,
            "video_index": self.current_video_index,
            "total_frames": 100,
        }

    def navigate_frame(self, direction):
        """Mock frame navigation method"""
        if direction == "next":
            self.current_frame_index = min(self.current_frame_index + 1, 99)
        elif direction == "previous":
            self.current_frame_index = max(self.current_frame_index - 1, 0)
        elif direction == "first":
            self.current_frame_index = 0
        elif direction == "last":
            self.current_frame_index = 99
        return True

    def skip_to_next_video(self):
        """Mock skip to next video method"""
        if self.current_video_index < len(self.available_video_indices) - 1:
            self.current_video_index += 1
            self.current_frame_index = 0
            return True
        return False

    def set_manual_navigation(self, enabled):
        """Mock manual navigation setting"""
        self.manual_navigation = enabled

    def process_video(self, video_path):
        """Mock video processing method"""
        return {"frames": 100, "duration": 10.0, "width": 640, "height": 480}

    def get_frame(self, frame_index):
        """Mock frame extraction method"""
        import numpy as np

        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def close(self):
        """Mock cleanup method"""
        self.models_loaded = False


class MockDataset:
    """Mock dataset class for testing"""

    def __init__(self):
        self.video_list = [f"video_{i}.mp4" for i in range(10)]
        self.gt_annotations = [{"objects": [], "relationships": []} for _ in range(10)]
