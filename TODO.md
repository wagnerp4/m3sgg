Datasets:
- Visual_genome

Models:
- OED
- SceneLLM replica
- SceneLLM adaption

Feature module:
- optical flow
- motion embeddings
- trajectory embeddings

NLP module:
- summarization, action anticipation
- gemma 3 270M, 
- other task?

UI/UX:
meta:
- discard entirely? -> web-app with hosting (compute requirements)
- if not: switch to better native python desktop app
logic:
- visualization modes
- sgdet performance
real-time:
- reduce runtime

Code:
- Cleanup
    - Comments
    - Formatter
- Documentation
    - Sphinx