#!/usr/bin/env python3
"""
Quick test to verify SceneLLM training initialization works without errors.
"""

import torch
import sys

sys.path.append("lib")
sys.path.append("fasterRCNN/lib")


def test_scenellm_training_init():
    """Test SceneLLM initialization with placeholder LLM for training."""
    print("Testing SceneLLM training initialization...")

    # Mock dataset
    class MockDataset:
        def __init__(self):
            self.attention_relationships = ["looking_at", "not_looking_at"]
            self.spatial_relationships = [
                "behind",
                "in_front_of",
                "next_to",
                "on",
                "above",
                "under",
            ]
            self.contacting_relationships = [
                "touching",
                "holding",
                "drinking_from",
                "eating",
                "writing_on",
                "wearing",
                "wiping",
                "pointing",
                "sitting_on",
                "lying_on",
                "standing_on",
                "opening",
                "pouring",
                "throwing",
                "carrying",
                "covered_by",
                "reading",
            ]
            self.object_classes = [
                "__background__",
                "person",
                "bag",
                "bed",
                "blanket",
                "book",
                "box",
                "broom",
                "chair",
                "closet/cabinet",
                "clothes",
                "cup/glass/bottle",
                "dish",
                "door",
                "doorknob",
                "doorway",
                "floor",
                "food",
                "groceries",
                "laptop",
                "light",
                "medicine",
                "mirror",
                "paper/notebook",
                "phone/camera",
                "picture",
                "pillow",
                "refrigerator",
                "sandwich",
                "shelf",
                "shoe",
                "sofa/couch",
                "table",
                "television",
                "towel",
                "vacuum",
                "window",
            ]

    # Mock config
    class MockConfig:
        def __init__(self):
            self.mode = "predcls"
            self.embed_dim = 512  # Smaller for testing
            self.codebook_size = 1024  # Smaller for testing
            self.commitment_cost = 0.25
            self.llm_name = "placeholder"  # Use placeholder to avoid download
            self.lora_r = 8
            self.lora_alpha = 16
            self.lora_dropout = 0.1
            self.ot_step = 256
            self.alpha_obj = 1.0
            self.alpha_rel = 1.0
            self.scenellm_training_stage = "vqvae"
            self.enc_layer = 1
            self.dec_layer = 2

    try:
        from lib.scenellm.scenellm import SceneLLM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create model
        dataset = MockDataset()
        config = MockConfig()

        print("Creating SceneLLM model...")
        model = SceneLLM(config, dataset).to(device)
        print("✓ Model created successfully")

        # Test all training stages
        stages = ["vqvae", "stage1", "stage2"]
        for stage in stages:
            print(f"Testing training stage: {stage}")
            try:
                model.set_training_stage(stage)
                print(f"✓ Stage {stage} set successfully")
            except Exception as e:
                print(f"✗ Stage {stage} failed: {e}")
                return False

        # Test parameter counting
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

        print("✓ All training initialization tests passed!")
        return True

    except Exception as e:
        print(f"✗ Training initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_scenellm_training_init()
    if success:
        print("\n🎉 SceneLLM is ready for training!")
        print("You can now run:")
        print(
            "python train.py -model scenellm -scenellm_training_stage vqvae -nepoch 1 -llm_name placeholder"
        )
    else:
        print("\n❌ SceneLLM training initialization failed!")

    exit(0 if success else 1)
