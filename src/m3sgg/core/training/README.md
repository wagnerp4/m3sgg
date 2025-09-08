# Training Module

This module provides a modularized approach to training scene graph generation models, following the requirements outlined in TODO.md.

## Structure

```
src/m3sgg/core/training/
├── __init__.py              # Module exports
├── trainer.py               # Main Trainer class
├── evaluation.py            # Evaluator class
├── example_usage.py         # Usage examples
└── README.md               # This file
```

## Classes

### Trainer

The main `Trainer` class encapsulates the training loop and provides the following methods:

- `train_loop()`: Main training loop that orchestrates the entire training process
- `train_epoch(epoch)`: Train the model for one epoch
- `train_step(batch_idx, train_iter, unc_vals)`: Execute one training step
- `evaluate_epoch(epoch)`: Evaluate the model for one epoch
- `save_checkpoints(epoch, score, mrecall)`: Save model checkpoints

### Evaluator

The `Evaluator` class handles evaluation logic:

- `eval_loop()`: Run the complete evaluation loop
- `_evaluate_easg()`: EASG-specific evaluation
- `_evaluate_action_genome()`: Action Genome-specific evaluation

## Usage

### Basic Usage

```python
from m3sgg.core.training import Trainer, Evaluator

# Initialize trainer with all necessary components
trainer = Trainer(
    model=model,
    config=config,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    optimizer=optimizer,
    scheduler=scheduler,
    logger=logger,
    # ... other components
)

# Run training
trainer.train_loop()
```

### Integration with Existing Code

To integrate with the existing `train.py` script:

1. Replace the monolithic training loop with:
   ```python
   trainer = Trainer(
       model=model,
       config=conf,
       dataloader_train=dataloader_train,
       dataloader_test=dataloader_test,
       optimizer=optimizer,
       scheduler=scheduler,
       logger=logger,
       object_detector=object_detector,
       object_detector_EASG=object_detector_EASG,
       matcher=matcher,
       evaluator=evaluator,
       evaluator2=evaluator2,
       dataset_train=dataset_train,
       dataset_test=dataset_test,
   )
   
   trainer.train_loop()
   ```

2. The complex loss computation logic can be moved to separate methods in the `Trainer` class or to model-specific classes.

## Benefits

1. **Modularity**: Training logic is separated from the main script
2. **Reusability**: Trainer can be used with different models and configurations
3. **Testability**: Individual methods can be unit tested
4. **Maintainability**: Easier to modify and extend training behavior
5. **Readability**: Clear separation of concerns

## TODO Items

The current implementation provides the basic structure. The following items need to be completed:

1. **Loss Computation**: Move the complex loss computation logic from `train.py` to the `Trainer` class
2. **Model-Specific Logic**: Implement model-specific training and evaluation logic
3. **Error Handling**: Add comprehensive error handling and validation
4. **Documentation**: Add detailed docstrings and type hints
5. **Testing**: Add unit tests for the Trainer and Evaluator classes

## Migration Path

1. **Phase 1**: Create the basic Trainer structure (✅ Completed)
2. **Phase 2**: Move loss computation logic to Trainer methods
3. **Phase 3**: Move evaluation logic to Evaluator class
4. **Phase 4**: Update train.py to use the new Trainer class
5. **Phase 5**: Add comprehensive testing and documentation

## Design Decisions

### Location Choice

The training module is placed in `src/m3sgg/core/training/` because:

1. **Core Functionality**: Training is a core part of the framework
2. **Separation from Models**: Keeps training logic separate from model implementations
3. **Consistency**: Follows the existing `core/` structure pattern
4. **Accessibility**: Easy to import and use from scripts

### Class Design

The `Trainer` class is designed to be:

1. **Comprehensive**: Contains all training-related functionality
2. **Configurable**: Accepts all necessary components as parameters
3. **Extensible**: Easy to add new model types or training strategies
4. **Testable**: Methods are designed for easy unit testing

## Future Enhancements

1. **Callback System**: Add callback support for custom training hooks
2. **Distributed Training**: Add support for multi-GPU training
3. **Resume Training**: Add checkpoint resuming functionality
4. **Metrics Tracking**: Add comprehensive metrics tracking and logging
5. **Hyperparameter Tuning**: Add support for automated hyperparameter optimization
