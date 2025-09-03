### In Progress
- [ ] **Documentation**: Update documentation
    - OED model
    - new config system
    - update README.md
- [ ] **Overhaul tests**: 
    - fixture
    - components
    - unit
- [ ] E2E features:
    - compute efficiency
    - model detector/smart checkpointing
    - nlp module 
- [ ] ActionGenome benchmark
    - 3/6 in train script definen
    - 6/6 done (7?)
- [ ] Features

### Remaining Tasks
- [ ] Improve CLI (scripts/core/training/train.py -> train)

### Solutions:
- `Store model name key`: Good practices if you decide to include "model_type"
Store versioned metadata (checkpoint["version"] = 2) so your loader can evolve safely.
Use a registry/dictionary of known models instead of eval() or arbitrary class names.
MODEL_REGISTRY = {
    "resnet18": ResNet18,
    "transformer": TransformerModel,
}
model = MODEL_REGISTRY[checkpoint["model_type"]](**checkpoint["config"])
Save config params alongside "model_type" (checkpoint["config"]), not just the name.
Keep loading flexible but explicit: fallback defaults, warnings if unknown keys, etc.
- ...
