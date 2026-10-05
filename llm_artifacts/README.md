# LLM + RAG artifacts (IFT6758 only)

Task 1 asks you to save, for reproducibility:

- `api_docs_snapshot/`: a snapshot of the NHL API documentation you gave the RAG pipeline
  (e.g. Zmalski's NHL API Reference, dword4's new-api.md, the game-ID section of the old doc)
- `retrieved_excerpts/`: the excerpts the retriever returned for each prompt
- `generated/`: the Python code the model wrote, one file per prompt iteration
- environment details (library versions, precision, generation hyperparameters)

Model: Qwen3-4B-Instruct-2507, loaded via `ift6758.llm.load_model()`.
