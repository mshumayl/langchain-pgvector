### What is this?
This is a simple CLI Q&A tool that uses LangChain to generate document embeddings, store them in a vector store ([PGVector](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pgvector.html) hosted on Supabase), retrieve them based on input similarity, and augment the LLM prompt with the knowledge base context.

The knowledge base documents are stored in the `/documents` directory.

### How to run the program?
1. Create a new [virtual environment](https://docs.python.org/3/library/venv.html#module-venv) and launch it.
2. Install dependencies with `pip install -r requirements.txt`.
3. Create a vector database on Supabase by enabling the [PGVector extension](https://supabase.com/docs/guides/database/extensions/pgvector).
4. Add your [OpenAI API key](https://platform.openai.com/account/api-keys) and PGVector database information into an `.env` file (refer to `.env.example` as reference).
5. If running for the first time, set the `INITIALIZE` variable to `True`. 
6. Run the program with `python qna.py`.

### How to 'fine-tune' it on my own documents?
Simply replace the `.txt` files in the `/documents` directory with your own documents and run the program.

### I am encountering an error about token dimension mismatch (`1536` vs `768`)
Follow the recommendations from this [GitHub Issue thread](https://github.com/hwchase17/langchain/issues/2219).