### What is this?
This program uses LangChain to fine-tune a large-language model with the documents provided in the `/documents` directory. LangChain generates text embeddings from the documents and uses Chroma (locally DuckDB) as the [vector store](https://js.langchain.com/docs/modules/indexes/vector_stores/).

### How to run?
1. Create a new [virtual environment](https://docs.python.org/3/library/venv.html#module-venv).
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the program with `python qna.py`.