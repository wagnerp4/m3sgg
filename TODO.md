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
            - RAFT, Lucas Kanade Method (1981)
        - [ ] 

- [ ] SGG Module

    - [ ] VLM

        - [✅] https://huggingface.co/apple/FastVLM-0.5B

        - [✅] BLIP (missing: BLIP-2)

        - [ ] Llava-1.5

        - [ ] LLaVA-SpaceSGG

- [ ] Language module

    - [ ] Eval summary

        - [ ] Data

        - [ ] Wrapper call replacements

---

## General Necessary Updates

- [ ] Documentation: Update documentation

    - [ ] OED model

    - [ ] New config system

    - [ ] Update README.md

- [ ] Overhaul tests: 

    - [ ] Fixture

    - [ ] Components

    - [ ] Unit

- [ ] E2E:

    - [ ] Compute efficiency

- [ ] Small benchmark on ag200.
    - plot R@20/50 and mR@20/50 vs. iter (sgdet) vanilla
        - use 3 models (STTran, DSG-DETR, Tempura) on 3 modes 
        - log metric per iter ticks -> to create a metrics vs. iter plot of the 
            - separate train loop in code: iter loop. epoch loop. inner loop. step. 
        - log computation-relevant metrics (GB storage, GFLOP/s, Time, Hardware-related)
    - plot R@20/50 and mR@20/50 vs. iter (sgdet) different detectors

- [ ] Improve CLI (scripts/core/training/train.py -> train)

(...)