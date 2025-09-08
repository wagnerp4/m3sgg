# TODOs

## Abstract Module View:

- [ ] Feature Module

    - [ ] Object Detector Features

        - [ ] Benchmark: Resnet sizes on d2, sgdet r@50.

        - [ ] Step 2: Include ViT <> DETR gradually.

    - [ ] Embedding Features

        - [ ] Research related embeddings ...

        - [ ] Trajectory Embeddings
        
    - [ ] Motion, Segmenting Features

        - [ ] Depth Estimation image: Feature image
            - i.e. Llava-SpaceSGG
        - [ ] Optical Flow Estimation: 
            - RAFT
        - [ ] 

- [ ] SGG Module

    - [✅] VLM

        - [✅] https://huggingface.co/apple/FastVLM-0.5B

        - [✅] BLIP (missing: BLIP-2)

        - [ ] Llava-1.5

        - [ ] LLaVA-SpaceSGG

- [ ] Language module

    - [✅] Eval summary

        - [ ] Data

        - [ ] Wrapper call replacements

---

## General Todos:

- [ ] Documentation: Update documentation

    - [ ] OED model

    - [ ] New config system

    - [ ] Update README.md

- [✅] Overhaul tests: 

    - [✅] Fixture

    - [✅] Components

    - [✅] Unit

- [ ] E2E:

    - [ ] Compute efficiency

- [ ] Small sgg benchmark on dataset: ag200.
    - models
        - plot R@20/50/100 and mR@20/50/100 vs. iteration on mode sgdet
            - use 3 models (STTran, DSG-DETR, Tempura) on 3 modes (3x3 checkpoints a 10 epochs in iterations)
            - log loss, metric per epoch
            - log computation-relevant metrics (GB storage, GFLOP/s, Time, Hardware-related)
            - queue all 9 trainings (a 10 epochs) to take place after eachother in one script call (.sh,...)
    - task: write one combined .ps1 script that runs all after another, saves 90 resulting outputs. Afterwards it should plot and store
    3 things: 
    1) combined plot of all 3 model's metrics, on all three models modes (predcls,sgcls,sgdet) over 10 epochs (x_tick=1).
    2) plot of train vs. eval loss values on sgdet mode, sttran over 10 epochs.
    3) plot of comparing run-time, memory, performance trade-off between the core 3 models

    Store the resulting plots and logfiles under data/benchmark/ag200. Use .pdf as output format
    Use high dpi settings, make sure the resolution and axis labeling is consistent to the code and description above
    and ready to be used in a research paper.

- [ ] Small detector benchmark on dataset: ag200.
    - models
        - plot R@20/50/100 and mR@20/50/100 vs. iteration on mode sgdet
            - use 1 model (STTran) on 3 modes (1x3 checkpoints a 10 epochs in iterations)
            - log metric per iter ticks -> to create a metrics vs. iter plot of the 
                - separate train loop in code: iter loop. epoch loop. inner loop. step. 
            - log computation-relevant metrics (GB storage, GFLOP/s, Time, Hardware-related)
            - also plot loss in the same
            - queue all 9 trainings (a 10 epochs) to take place after eachother in one script call (.sh,...)
    - task: write one combined .ps1 script that runs all after another

- [ ] Improve CLI (scripts/core/training/train.py -> train)

- [ ] Recompile fasterRCNN, avoid __init__.py dependency

- [ ] Checkpoint Manager

- [ ] Device Handling (batch tasks)