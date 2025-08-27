#!/usr/bin/env python3
"""
TEMPURA Hyperparameter Generator
Generates hyperparameter combinations for TEMPURA model training.
Supports both grid search and random search methods.
"""

import argparse
import json
import random
from itertools import product
from typing import Any, Dict, List, Union


class TEMPURAHyperparameterGenerator:
    """Generator for TEMPURA model hyperparameters"""

    def __init__(self, search_method: str = "grid", max_combinations: int = 20):
        self.search_method = search_method
        self.max_combinations = max_combinations

        # Define hyperparameter search spaces
        self.hyperparameter_spaces = {
            # Learning rate
            "lr": [1e-5, 5e-5, 1e-4, 5e-4],
            # Model architecture
            "enc_layer": [1, 2, 3],
            "dec_layer": [2, 3, 4],
            # TEMPURA-specific parameters
            "obj_head": ["gmm", "linear"],
            "rel_head": ["gmm", "linear"],
            "K": [2, 4, 6, 8],
            # Memory parameters
            "rel_mem_compute": [None, "separate", "joint"],
            "obj_mem_compute": [False, True],
            "take_obj_mem_feat": [False, True],
            "obj_mem_weight_type": ["simple", "both", "al", "ep"],
            "rel_mem_weight_type": ["simple", "both", "al", "ep"],
            "mem_feat_selection": ["manual", "automated"],
            "mem_fusion": ["early", "late"],
            "mem_feat_lambda": [None, 0.1, 0.5, 1.0],
            "pseudo_thresh": [5, 7, 10],
            # Uncertainty parameters
            "obj_unc": [False, True],
            "rel_unc": [False, True],
            # Loss parameters
            "obj_loss_weighting": [None, "ep", "al"],
            "rel_loss_weighting": [None, "ep", "al"],
            "mlm": [False, True],
            "eos_coef": [0.5, 1.0, 2.0],
            "obj_con_loss": [None, "euc_con", "info_nce"],
            "lambda_con": [0.5, 1.0, 2.0],
            # Training parameters
            "optimizer": ["adamw", "adam"],
            "bce_loss": [False, True],
        }

    def generate_grid_search_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible combinations for grid search"""
        # Get all parameter names and their possible values
        param_names = list(self.hyperparameter_spaces.keys())
        param_values = list(self.hyperparameter_spaces.values())

        # Generate all combinations
        all_combinations = []
        for combination in product(*param_values):
            hp_dict = dict(zip(param_names, combination))

            # Apply some logical constraints
            if self._is_valid_combination(hp_dict):
                all_combinations.append(hp_dict)

        # Limit the number of combinations if needed
        if len(all_combinations) > self.max_combinations:
            print(
                f"Warning: Grid search would generate {len(all_combinations)} combinations."
            )
            print(f"Limiting to {self.max_combinations} combinations.")
            all_combinations = all_combinations[: self.max_combinations]

        return all_combinations

    def generate_random_search_combinations(self) -> List[Dict[str, Any]]:
        """Generate random combinations for random search"""
        combinations = []
        attempts = 0
        max_attempts = (
            self.max_combinations * 10
        )  # Allow more attempts to find valid combinations

        while len(combinations) < self.max_combinations and attempts < max_attempts:
            attempts += 1

            # Generate random combination
            hp_dict = {}
            for param_name, param_values in self.hyperparameter_spaces.items():
                hp_dict[param_name] = random.choice(param_values)

            # Check if combination is valid
            if self._is_valid_combination(hp_dict):
                combinations.append(hp_dict)

        if len(combinations) < self.max_combinations:
            print(
                f"Warning: Could only generate {len(combinations)} valid combinations."
            )

        return combinations

    def _is_valid_combination(self, hp_dict: Dict[str, Any]) -> bool:
        """Check if a hyperparameter combination is valid"""

        # Memory-related constraints
        if hp_dict["obj_mem_compute"] and not hp_dict["take_obj_mem_feat"]:
            # If object memory computation is enabled, should take object memory features
            return False

        if hp_dict["rel_mem_compute"] is not None and hp_dict[
            "rel_mem_compute"
        ] not in ["separate", "joint"]:
            return False

        # Loss weighting constraints
        if hp_dict["obj_loss_weighting"] not in [None, "ep", "al"]:
            return False

        if hp_dict["rel_loss_weighting"] not in [None, "ep", "al"]:
            return False

        # Contrastive loss constraints
        if hp_dict["obj_con_loss"] is not None and hp_dict["obj_con_loss"] not in [
            "euc_con",
            "info_nce",
        ]:
            return False

        # Memory feature lambda constraints
        if hp_dict["mem_feat_lambda"] is not None and hp_dict["mem_feat_lambda"] <= 0:
            return False

        # K value constraints
        if hp_dict["K"] <= 0:
            return False

        # Learning rate constraints
        if hp_dict["lr"] <= 0:
            return False

        # Layer constraints
        if hp_dict["enc_layer"] <= 0 or hp_dict["dec_layer"] <= 0:
            return False

        return True

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate hyperparameter combinations based on search method"""
        if self.search_method == "grid":
            return self.generate_grid_search_combinations()
        elif self.search_method == "random":
            return self.generate_random_search_combinations()
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")

    def print_combination_summary(self, combinations: List[Dict[str, Any]]):
        """Print a summary of the generated combinations"""
        import sys

        print(
            f"Generated {len(combinations)} hyperparameter combinations using {self.search_method} search.",
            file=sys.stderr,
        )
        print("\nParameter ranges:", file=sys.stderr)
        for param_name, param_values in self.hyperparameter_spaces.items():
            if isinstance(param_values[0], bool):
                print(f"  {param_name}: {param_values}", file=sys.stderr)
            elif isinstance(param_values[0], (int, float)):
                print(
                    f"  {param_name}: {min(param_values)} to {max(param_values)}",
                    file=sys.stderr,
                )
            else:
                print(f"  {param_name}: {param_values}", file=sys.stderr)

        print(f"\nFirst few combinations:", file=sys.stderr)
        for i, combo in enumerate(combinations[:3]):
            print(f"  Combination {i+1}: {combo}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate TEMPURA hyperparameter combinations"
    )
    parser.add_argument(
        "--search_method",
        choices=["grid", "random"],
        default="random",
        help="Search method: grid or random",
    )
    parser.add_argument(
        "--max_combinations",
        type=int,
        default=10,
        help="Maximum number of combinations to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible results"
    )

    args = parser.parse_args()

    # Set random seed for reproducible results
    random.seed(args.seed)

    # Create generator
    generator = TEMPURAHyperparameterGenerator(
        search_method=args.search_method, max_combinations=args.max_combinations
    )

    # Generate combinations
    combinations = generator.generate_combinations()

    # Print summary to stderr so it doesn't interfere with JSON output
    import sys

    generator.print_combination_summary(combinations)

    # Output as JSON for PowerShell to consume (to stdout)
    print(json.dumps(combinations, indent=2), file=sys.stdout, flush=True)


if __name__ == "__main__":
    main()
