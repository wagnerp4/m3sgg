# lib/vlm.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class VLMSceneGraphGenerator(nn.Module):
    def __init__(
        self,
        mode="sgdet",
        attention_class_num=3,
        spatial_class_num=6,
        contact_class_num=17,
        obj_classes=None,
        model_name="apple/FastVLM-0.5B",  # Salesforce/blip-vqa-base
        device="cuda",
        few_shot_examples=None,
        use_chain_of_thought=True,
        use_tree_of_thought=False,
        confidence_threshold=0.5,
    ):
        """
        Initialize VLM Scene Graph Generator.

        :param mode: Scene graph generation mode (sgdet, sgcls, predcls)
        :param attention_class_num: Number of attention relationship classes
        :param spatial_class_num: Number of spatial relationship classes
        :param contact_class_num: Number of contact relationship classes
        :param obj_classes: List of object classes
        :param model_name: HuggingFace model name for VLM
        :param device: Device to run inference on
        :param few_shot_examples: Few-shot examples for prompting
        :param use_chain_of_thought: Whether to use chain-of-thought reasoning
        :param use_tree_of_thought: Whether to use tree-of-thought reasoning
        :param confidence_threshold: Threshold for relationship confidence
        """
        super().__init__()
        self.mode = mode
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.obj_classes = obj_classes or []
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_chain_of_thought = use_chain_of_thought
        self.use_tree_of_thought = use_tree_of_thought

        # TODO: add VLM model interface as submodule for sgg module
        supported_models = ["Salesforce/blip-vqa-base", "apple/FastVLM-0.5B"]
        if model_name not in supported_models:
            raise ValueError(
                f"Unsupported model_name: '{model_name}'. "
                f"Supported models are: {supported_models}. "
                f"Please use one of the supported models."
            )

        self.model_name = model_name
        if model_name == "apple/FastVLM-0.5B":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model_type = "causal_lm"
        elif model_name == "Salesforce/blip-vqa-base":
            from transformers import BlipProcessor, BlipForQuestionAnswering

            self.tokenizer = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(
                model_name,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto",
            )
            self.model_type = "blip"
        else:
            raise ValueError(f"Unexpected model_name: {model_name}")

        # AG Relationship mapping
        # TODO: move to config or data-dependent
        self.attention_relationships = ["looking_at", "not_looking_at", "unsure"]
        self.spatial_relationships = [
            "above",
            "beneath",
            "in_front_of",
            "behind",
            "on_the_side_of",
            "in",
        ]
        self.contact_relationships = [
            "carrying",
            "covered_by",
            "drinking_from",
            "eating",
            "have_it_on_the_back",
            "holding",
            "leaning_on",
            "lying_on",
            "not_contacting",
            "other_relationship",
            "sitting_on",
            "standing_on",
            "touching",
            "twisting",
            "wearing",
            "wiping",
            "writing_on",
        ]
        self.few_shot_examples = (
            few_shot_examples or self._get_default_few_shot_examples()
        )
        self.object_classifier = self._create_object_classifier()

    def _create_object_classifier(self):
        """Create a simple object classifier head."""
        return nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, len(self.obj_classes))
        )

    def _get_default_few_shot_examples(self):
        """Get default few-shot examples for prompting."""
        return [
            {
                "image_description": "A person sitting on a chair looking at a laptop",
                "objects": ["person", "chair", "laptop"],
                "relationships": [
                    ("person", "sitting_on", "chair"),
                    ("person", "looking_at", "laptop"),
                ],
            },
            {
                "image_description": "A person holding a cup and standing near a table",
                "objects": ["person", "cup", "table"],
                "relationships": [
                    ("person", "holding", "cup"),
                    ("person", "standing_on", "table"),
                ],
            },
        ]

    def _create_prompt(
        self,
        image_description: str,
        objects: List[str],
        use_chain_of_thought: bool = True,
    ) -> str:
        """Create a prompt for the VLM."""

        # Base prompt
        prompt = f"""Analyze this image and identify all relationships between objects.
                    Image description: {image_description}
                    Objects detected: {', '.join(objects)}

                    Please identify all relationships between these objects. For each relationship, specify:
                    1. Subject object
                    2. Relationship type  
                    3. Object object

                    Available relationship types:
                    - Attention: {', '.join(self.attention_relationships)}
                    - Spatial: {', '.join(self.spatial_relationships)}  
                    - Contact: {', '.join(self.contact_relationships)}
                    """

        if use_chain_of_thought:
            prompt += """
                        Think step by step:
                        1. First, identify all possible object pairs
                        2. For each pair, determine what relationships exist
                        3. Consider both spatial and contact relationships
                        4. Be specific about the relationship type
                        """

        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples[:2]):
            prompt += f"""
                        Example {i+1}:
                        Image: {example['image_description']}
                        Objects: {', '.join(example['objects'])}
                        Relationships:
                        """
            for subj, rel, obj in example["relationships"]:
                prompt += f"- {subj} {rel} {obj}\n"
            prompt += "\n"

        prompt += """Now analyze the current image and 
        provide the relationships in the same format:"""

        return prompt

    def _extract_relationships_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relationships from VLM text output."""
        relationships = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("-") and " " in line:
                # Parse format: "- subject relationship object"
                parts = line[1:].strip().split()
                if len(parts) >= 3:
                    subject = parts[0]
                    relationship = parts[1]
                    obj = " ".join(parts[2:])
                    relationships.append((subject, relationship, obj))

        return relationships

    def _convert_relationships_to_distributions(
        self,
        relationships: List[Tuple[str, str, str]],
        pair_idx: torch.Tensor,
        pred_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Convert text relationships to distribution tensors."""
        num_pairs = pair_idx.shape[0]
        device = pair_idx.device

        # Initialize dists (without gradients initially to allow in-place operations)
        attention_dist = torch.zeros(
            num_pairs, self.attention_class_num, device=device, requires_grad=False
        )
        spatial_dist = torch.zeros(
            num_pairs, self.spatial_class_num, device=device, requires_grad=False
        )
        contact_dist = torch.zeros(
            num_pairs, self.contact_class_num, device=device, requires_grad=False
        )

        # Create fuzzy matching for object names
        def fuzzy_match_object(vlm_name, dataset_names):
            """Fuzzy match VLM output to dataset object names."""
            vlm_name = vlm_name.lower().strip()

            # Direct match
            if vlm_name in dataset_names:
                return vlm_name

            # Handle common variations based on Action Genome object classes
            variations = {
                # Exact matches from Action Genome object classes
                "person": "person",
                "bag": "bag",
                "bed": "bed",
                "blanket": "blanket",
                "book": "book",
                "box": "box",
                "broom": "broom",
                "chair": "chair",
                "clothes": "clothes",
                "dish": "dish",
                "door": "door",
                "doorknob": "doorknob",
                "doorway": "doorway",
                "floor": "floor",
                "food": "food",
                "groceries": "groceries",
                "laptop": "laptop",
                "light": "light",
                "medicine": "medicine",
                "mirror": "mirror",
                "picture": "picture",
                "pillow": "pillow",
                "refrigerator": "refrigerator",
                "sandwich": "sandwich",
                "shelf": "shelf",
                "shoe": "shoe",
                "table": "table",
                "television": "television",
                "towel": "towel",
                "vacuum": "vacuum",
                "window": "window",
                # TODO: use regex
                # Common variations that map to Action Genome classes
                "closet": "closetcabinet",
                "cabinet": "closetcabinet",
                "cup": "cupglassbottle",
                "glass": "cupglassbottle",
                "bottle": "cupglassbottle",
                "paper": "papernotebook",
                "notebook": "papernotebook",
                "phone": "phonecamera",
                "camera": "phonecamera",
                "sofa": "sofacouch",
                "couch": "sofacouch",
                "tv": "television",
                "fridge": "refrigerator",
                "vacuum cleaner": "vacuum",
                "vacuumcleaner": "vacuum",
            }

            if vlm_name in variations:
                return variations[vlm_name]

            # Partial matching
            for dataset_name in dataset_names:
                if vlm_name in dataset_name or dataset_name in vlm_name:
                    return dataset_name

            return None

        # Create fuzzy matching for relationships
        def fuzzy_match_relationship(vlm_rel, dataset_rels):
            """Fuzzy match VLM relationship to dataset relationship."""
            vlm_rel = vlm_rel.lower().strip().replace(" ", "")

            # Direct match
            if vlm_rel in dataset_rels:
                return vlm_rel

            # Handle common variations based on Action Genome relationship classes
            variations = {
                # Attention relationships (0:3)
                "lookingat": "looking_at",
                "looking at": "looking_at",
                "notlookingat": "not_looking_at",
                "not looking at": "not_looking_at",
                "unsure": "unsure",
                # Spatial relationships (3:9)
                "above": "above",
                "beneath": "beneath",
                "infrontof": "in_front_of",
                "in front of": "in_front_of",
                "behind": "behind",
                "onthesideof": "on_the_side_of",
                "on the side of": "on_the_side_of",
                "in": "in",
                # Contacting relationships (9:)
                "carrying": "carrying",
                "coveredby": "covered_by",
                "covered by": "covered_by",
                "drinkingfrom": "drinking_from",
                "drinking from": "drinking_from",
                "eating": "eating",
                "haveitontheback": "have_it_on_the_back",
                "have it on the back": "have_it_on_the_back",
                "holding": "holding",
                "leaningon": "leaning_on",
                "leaning on": "leaning_on",
                "lyingon": "lying_on",
                "lying on": "lying_on",
                "notcontacting": "not_contacting",
                "not contacting": "not_contacting",
                "otherrelationship": "other_relationship",
                "other relationship": "other_relationship",
                "sittingon": "sitting_on",
                "sitting on": "sitting_on",
                "standingon": "standing_on",
                "standing on": "standing_on",
                "touching": "touching",
                "twisting": "twisting",
                "wearing": "wearing",
                "wiping": "wiping",
                "writingon": "writing_on",
                "writing on": "writing_on",
            }

            if vlm_rel in variations:
                # Only return if the variation exists in the target dataset_rels
                if variations[vlm_rel] in dataset_rels:
                    return variations[vlm_rel]

            # Partial matching
            for dataset_rel in dataset_rels:
                if vlm_rel in dataset_rel or dataset_rel in vlm_rel:
                    return dataset_rel

            return None

        # Process relationships with fuzzy matching
        for subj, rel, obj in relationships:
            # Find matching pairs using fuzzy matching
            for i, (subj_idx, obj_idx) in enumerate(pair_idx):
                subj_name = self.obj_classes[pred_labels[subj_idx]]
                obj_name = self.obj_classes[pred_labels[obj_idx]]

                # Fuzzy match objects
                matched_subj = fuzzy_match_object(subj, [subj_name])
                matched_obj = fuzzy_match_object(obj, [obj_name])

                if matched_subj == subj_name and matched_obj == obj_name:
                    # Fuzzy match relationship - check all relationship types
                    matched_rel = None
                    rel_idx = None

                    # Check attention relationships first
                    matched_rel = fuzzy_match_relationship(
                        rel, self.attention_relationships
                    )
                    if matched_rel:
                        rel_idx = self.attention_relationships.index(matched_rel)
                        attention_dist[i, rel_idx] = 0.8  # Soft assignment
                        continue

                    # Check spatial relationships
                    matched_rel = fuzzy_match_relationship(
                        rel, self.spatial_relationships
                    )
                    if matched_rel:
                        rel_idx = self.spatial_relationships.index(matched_rel)
                        spatial_dist[i, rel_idx] = 0.8  # Soft assignment
                        continue

                    # Check contacting relationships
                    matched_rel = fuzzy_match_relationship(
                        rel, self.contact_relationships
                    )
                    if matched_rel:
                        rel_idx = self.contact_relationships.index(matched_rel)
                        contact_dist[i, rel_idx] = 0.8  # Soft assignment
                        continue

        # Add small random noise to prevent zero gradients
        attention_dist = attention_dist + torch.randn_like(attention_dist) * 0.01
        spatial_dist = spatial_dist + torch.randn_like(spatial_dist) * 0.01
        contact_dist = contact_dist + torch.randn_like(contact_dist) * 0.01

        # Normalize to probability distributions (create new tensors to avoid in-place operations)
        attention_dist = torch.softmax(attention_dist, dim=1)
        spatial_dist = torch.softmax(spatial_dist, dim=1)
        contact_dist = torch.softmax(contact_dist, dim=1)

        # Enable gradients for the final distributions
        attention_dist = attention_dist.requires_grad_(True)
        spatial_dist = spatial_dist.requires_grad_(True)
        contact_dist = contact_dist.requires_grad_(True)

        return {
            "attention_distribution": attention_dist,
            "spatial_distribution": spatial_dist,
            "contact_distribution": contact_dist,
        }

    def _crop_objects_from_image(
        self, image: torch.Tensor, boxes: torch.Tensor, im_info: torch.Tensor
    ) -> List[Image.Image]:
        """Crop objects from image based on bounding boxes."""
        # Convert tensor to PIL Image
        if image.dim() == 4:
            image = image.squeeze(0)

        # Denormalize and convert to PIL
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)

        cropped_objects = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4].cpu().numpy()
            cropped = pil_image.crop((x1, y1, x2, y2))
            cropped_objects.append(cropped)

        return cropped_objects

    def forward(self, entry: Dict, im_data: torch.Tensor = None) -> Dict:
        """
        Forward pass through VLM Scene Graph Generator.

        :param entry: Input dictionary containing image data and bounding boxes
        :return: Dictionary with scene graph predictions
        """
        # Extract input data
        boxes = entry["boxes"]
        features = entry["features"]
        pair_idx = entry["pair_idx"]
        im_idx = entry["im_idx"]

        # Object classification
        obj_distribution = self.object_classifier(features)
        pred_labels = torch.argmax(obj_distribution, dim=1)

        # Get unique objects for this batch
        unique_objects = []
        for i in range(len(pred_labels)):
            obj_name = self.obj_classes[pred_labels[i]]
            if obj_name not in unique_objects:
                unique_objects.append(obj_name)

        # Create image description (simplified for now)
        image_description = f"Image containing: {', '.join(unique_objects)}"

        # Create prompt
        prompt = self._create_prompt(
            image_description,
            unique_objects,
            use_chain_of_thought=self.use_chain_of_thought,
        )

        # Prepare VLM input based on model type
        if self.model_type == "causal_lm":
            # Handle causal LM models (FastVLM, LLaVA, etc.)
            messages = [{"role": "user", "content": f"<image>\n{prompt}"}]

            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            except (ValueError, AttributeError):
                # Fallback for tokenizers without chat template
                rendered = f"User: {messages[0]['content']}\nAssistant:"

            pre, post = rendered.split("<image>", 1)
            pre_ids = self.tokenizer(
                pre, return_tensors="pt", add_special_tokens=False
            ).input_ids
            post_ids = self.tokenizer(
                post, return_tensors="pt", add_special_tokens=False
            ).input_ids

            # Insert image token
            IMAGE_TOKEN_INDEX = -200
            img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)
        elif self.model_type == "blip":
            # Handle BLIP models - only process text, not images
            text_inputs = self.tokenizer.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Process actual image data
        if im_data is not None:
            # Use the actual image data from the training loop
            # im_data is typically [batch_size, channels, height, width]
            if im_data.dim() == 4:
                # Take the first image in the batch
                image_tensor = im_data[0]  # [channels, height, width]
            else:
                image_tensor = im_data  # [channels, height, width]

            # Normalize image to [0, 1] range if needed
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0

            # Convert to PIL Image for proper processing
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        else:
            # Fallback: create a properly normalized dummy image for testing
            dummy_image = torch.rand(3, 224, 224).to(self.device)
            image_np = dummy_image.permute(1, 2, 0).cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        # Process image based on model type
        if self.model_type == "causal_lm":
            # Try to use vision tower for vision-capable models (like FastVLM, LLaVA)
            if hasattr(self.model, "get_vision_tower"):
                pixel_values = (
                    self.model.get_vision_tower()
                    .image_processor(images=image_pil, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .to(self.device, dtype=self.model.dtype)
                )
            else:
                # Fallback for models without vision tower
                print(
                    f"Warning: Model {self.model_name} doesn't have vision tower, using fallback image processing"
                )
                pixel_values = torch.randn(1, 3, 224, 224).to(
                    self.device, dtype=self.model.dtype
                )
        elif self.model_type == "blip":
            # BLIP models use the processor directly for images only
            pixel_values = self.tokenizer.image_processor(
                images=image_pil, return_tensors="pt"
            ).pixel_values.to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Generate with VLM based on model type
        with torch.no_grad():
            if self.model_type == "causal_lm":
                try:
                    # Try with images parameter for vision-capable models
                    outputs = self.model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        images=pixel_values,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                except ValueError as e:
                    # Fallback for models that don't accept images parameter
                    print(f"Warning: Model doesn't accept images parameter: {e}")
                    outputs = self.model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
            elif self.model_type == "blip":
                # BLIP models use different generation method
                outputs = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        # Decode response
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract relationships
        relationships = self._extract_relationships_from_text(response_text)

        # Convert to distributions
        distributions = self._convert_relationships_to_distributions(
            relationships, pair_idx, pred_labels
        )

        # Prepare output
        output = {
            "distribution": obj_distribution,
            "pred_labels": pred_labels,
            "attention_distribution": distributions["attention_distribution"],
            "spatial_distribution": distributions["spatial_distribution"],
            "contact_distribution": distributions["contact_distribution"],
            "attention_gt": entry.get("attention_gt", []),
            "spatial_gt": entry.get("spatial_gt", []),
            "contact_gt": entry.get("contact_gt", []),
            "labels": entry.get("labels", pred_labels),
            "pair_idx": pair_idx,  # Required for evaluation
            "im_idx": im_idx,  # Required for evaluation
            "boxes": entry.get(
                "boxes", torch.zeros((len(pred_labels), 5), device=self.device)
            ),  # Required for evaluation
            "scores": entry.get(
                "scores", torch.ones(len(pred_labels), device=self.device)
            ),  # Required for evaluation
            "pred_scores": torch.softmax(obj_distribution, dim=1).max(dim=1)[
                0
            ],  # Use actual confidence scores
            "vlm_response": response_text,  # For debugging
        }

        return output
