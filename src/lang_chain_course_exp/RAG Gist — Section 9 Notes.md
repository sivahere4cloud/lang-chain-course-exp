# RAG Gist — Section 9 Notes

Oct 1, 2026 · @Siva

## Overview

Section 9 built a working 2-step RAG over one article: a Wikipedia "Vector database" text is stored as 11 chunks in Pinecone and answers questions from them only. The test question "Which conferences have hosted competitions on vector search?" returned SISAP and NeurIPS with retrieval, versus a generic 10-venue list without it.

&#91;embedded content: RAG pipeline · ingestion once, retrieval per question\]

The top row runs once to fill the index; the bottom row runs for every question and only embeds the question, never the article again.

- Branch: `project/rag-gist` in `lang-chain-course-exp`
- Data: `mediumblog1.txt` in the project root (Wikipedia, CC BY-SA 4.0)
- Index: `medium-analyzer` (1536 dimensions, cosine, AWS us-east-1, serverless)
- Models: `text-embedding-3-small` for embeddings, `gpt-5-mini` for answers

| File | Part | Job |
| --- | --- | --- |
| `rag0_create_index.py` | 0 | Create the Pinecone index if missing |
| `rag1_load.py` | 1 | Text file to a `Document` |
| `rag2_split.py` | 2 | `Document` to chunks |
| `rag3_embed.py` | 3 | Chunks and question to vectors, scored by cosine |
| `rag4_ingest.py` | 4 | Load, split, embed, store in Pinecone |
| `rag5_naive_retrieval.py` | 5 | Retrieve, prompt, answer as separate steps |
| `rag6a_lcel_basics.py` | 6a | LCEL building blocks, no API calls |
| `rag6_lcel_chain.py` | 6b | The same RAG as one LCEL chain, with streaming |

## What changed since the videos

The course code was recorded months before these versions, so five things differ; each was checked against the installed packages.

| Course | Now | What we did |
| --- | --- | --- |
| `TextLoader` from `langchain_community` | `langchain-community` is being sunset and prints a deprecation warning | Read the file with plain Python and build `Document` objects |
| `CharacterTextSplitter` | Docs use `RecursiveCharacterTextSplitter` (`langchain-text-splitters` 1.1.2) | Recursive splitter, 500 / 50 |
| Index created in the Pinecone website | Docs create it in code with `ServerlessSpec` | `rag0_create_index.py`, safe to re-run |
| `StrOutputParser` returns `str` | Returns `TextAccessor`, a `str` subclass kept for the `.text()` to `.text` change in v1.0 | Works as a string; use `str(answer)` when an exact `str` is needed |
| Old RAG tutorial URL | Now redirects to a Deep Agents version; LCEL details live in the API reference | Read [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval), [semantic search](https://docs.langchain.com/oss/python/langchain/knowledge-base) and the [RunnablePassthrough reference](https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough) |

Versions used: `langchain` 1.4.2, `langchain-openai` 1.6.6, `langchain-pinecone` 0.2.13, `pinecone` 7.3.0, `langchain-text-splitters` 1.1.2.

## Parts 0 to 6

Each part is one file, and later parts import earlier functions instead of copying them.

### Part 0: Create the index

`Pinecone()` reads `PINECONE_API_KEY`; `pc.has_index(name)` returns `bool`; `pc.create_index(name, dimension, metric, spec=ServerlessSpec(cloud, region))` runs only when the index is missing; `pc.describe_index(name)` returns details with `.status.ready`. The free Starter plan only allows AWS `us-east-1`.

### Part 1: Load

`Path.read_text(encoding="utf-8")` returns the file as `str`; `Document(page_content=text, metadata={"source": file_path.name})` wraps it. The run loaded 3,399 characters with `{'source': 'mediumblog1.txt'}`.

### Part 2: Split

`RecursiveCharacterTextSplitter(chunk_size, chunk_overlap).split_documents([doc])` returns `list[Document]`, each keeping the source metadata. It tries paragraphs, then lines, then spaces.

| `chunk_size` | Chunks | Effect |
| --- | --- | --- |
| 4000 | 1 | Whole article in one chunk, so no real retrieval |
| 500 | 11 | Lengths 75 to 497; short chunks are a lone source line, a sentence remainder and a lone heading |

### Part 3: Embed

`OpenAIEmbeddings(model=...)` only creates the client. `embed_query(text)` returns one `list[float]` of 1536 numbers; `embed_documents(texts)` returns one vector per text in a single API call. Cosine similarity = dot product divided by the two vector lengths.

### Part 4: Ingest

`pc.Index(name)` connects to the existing index; `PineconeVectorStore(index=..., embedding=...)` wraps it; `add_documents(documents=chunks, ids=ids)` embeds and upserts, returning the stored IDs. Fixed IDs `mediumblog1.txt-chunk-1` to `-11` mean a re-run overwrites instead of duplicating. The run went from 0 to 11 vectors.

### Part 5: Naive retrieval

`vector_store.as_retriever(search_kwargs={"k": 3})` returns a retriever; `retriever.invoke(question)` returns `list[Document]`; `format_docs` joins their text; `ChatPromptTemplate` fills `{context}` and `{question}`; `llm.invoke` returns an `AIMessage`. The top chunk was the one naming SISAP and NeurIPS.

### Part 6: LCEL chain

```python
retrieval_part = itemgetter("question") | retriever | RunnableLambda(format_docs)
rag_chain = (
    RunnablePassthrough.assign(context=retrieval_part)
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke({"question": QUESTION})
```

`RunnablePassthrough.assign` keeps `{"question"}` and adds `"context"`, so the prompt gets both keys. `.invoke` returns the whole answer; `.stream` yields it piece by piece. LangSmith shows one `RunnableSequence` trace with every step inside.

## Part 7: RAG architectures

What we built is 2-step RAG: it always retrieves, then generates, in a fixed order. The [Retrieval docs](https://docs.langchain.com/oss/python/langchain/retrieval) describe three architectures.

| Architecture | Who decides to retrieve | Strength | Cost | Good for |
| --- | --- | --- | --- | --- |
| 2-step RAG | Your code, every time | Simple and predictable; 1 LLM call | Retrieves even when not needed | FAQs, docs bots, a store or pharmacy chatbot |
| Agentic RAG | An agent with a retriever tool | Flexible; can search several times or not at all | More LLM calls, more latency and cost; harder to control | Research assistants |
| Hybrid RAG | Fixed steps plus checks (for example, judge relevance, rewrite the query, retry) | Predictable with quality checks | More steps to build | Domain-specific Q&A where wrong answers matter |

The course (lecture 50) warns that agentic RAG gives the LLM too much freedom for many production bots: it can answer off-topic questions and adds tool-call round trips. Section 3's job agent was agentic; this section's chain is 2-step.

## Production lessons

The AI calls are a few lines; most production work is making them safe to re-run, debuggable and consistent.

1. **Idempotent setup and ingestion.** Check `has_index` before creating, and use fixed record IDs so a re-run overwrites (11 stays 11) instead of duplicating (11 becomes 22).
2. **One embedding model everywhere.** Ingestion and querying must use the same model, and the index dimension must match it (1536 for `text-embedding-3-small`). A mismatch returns wrong chunks with no error.
3. **Chunk size suits the documents.** 4000 gave 1 chunk for a 3,399-character article; 500 gave 11. Tune it with evals, not by guessing.
4. **Clean text before embedding.** Licence lines, lone headings and citation markers like `[1]` become weak chunks; keep source details in metadata instead.
5. **Ground the prompt.** "Answer only from the context, otherwise say you don't know" is the cheapest guard against confident made-up answers.
6. **Keys from the environment.** `.env` and `load_dotenv()`, never `getpass` or keys in code.
7. **Fail early with clear errors.** Check the file exists and show its full path; always pass `encoding="utf-8"` (Windows defaults differ).
8. **One trace per request.** An LCEL chain shows retrieval, prompt and LLM in one LangSmith trace, so you can tell whether retrieval or generation failed.
9. **Build once, call many times.** Create clients and the chain at startup and reuse them per request; stream answers for chat UIs.
10. **Settings in one place.** Index name, models, chunk size and `k` are repeated across files today; the module project moves them to one `config.py`.

## How to read the docs

Start at the integration page for the one tool you need, not the overview, and drop a layer only when it leaves a question open.

| Layer | Where | Answers | Example from this section |
| --- | --- | --- | --- |
| L1 Overview | [Vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores) | Which tools exist; smallest code to connect | Pinecone snippet assumes the index already exists |
| L2 Integration page | [Pinecone integration](https://docs.langchain.com/oss/python/integrations/vectorstores/pinecone) | Full walkthrough: setup, credentials, initialization, add, query | `has_index` + `create_index`, `add_documents`, `as_retriever` |
| L3 Vendor docs | [Pinecone: create an index](https://docs.pinecone.io/guides/index-data/create-an-index) | Every option and plan limit | Starter plan only allows `us-east-1` |
| L4 Code itself | API reference, hover or Ctrl+click in Cursor | Exact parameters and return types | `describe_index`, `RunnablePassthrough.assign`, `TextAccessor` |

1. Open the L2 page for the tool.
2. Read the headings first; that is the order you write the code in.
3. For each code block, note what it needs (imports, keys) and what it gives (client, index, store).
4. Replace notebook habits: `getpass` becomes `.env`; hard-coded names become constants.
5. Unclear option? Go to L3.
6. Unclear signature or return type? Go to L4.

## Git workflow

Work on one branch per section, then merge it into `main`; GitHub only counts commits on the default branch toward the contribution graph.

```
# start a section (from the finished previous one)
git switch main
git merge project/<previous-section>
git push
git switch -c project/<new-section>

# while working
git add .
git status            # check .env is not listed
git commit -m "..."
git push              # first push of a branch: git push -u origin project/<new-section>

# section done
git switch main
git merge project/<new-section>
git push
```

- `git branch -m <new-name>` renames the current branch (used to fix `project/rag-gistm`).
- `git show <branch-or-commit>:<path>` reads an old file without switching branches.
- Deleting a section's files on a new branch never deletes them from older branches or `main`.

## Glossary

| Term | Meaning |
| --- | --- |
| Chunk | One piece of a document; one record in the vector store |
| `chunk_overlap` | Characters shared by neighbouring chunks |
| Context | Retrieved chunk text pasted into the prompt |
| Cosine similarity | How alike two vectors' directions are; closer to 1 is more similar |
| Dimension | Numbers per vector; must match the embedding model |
| `Document` | LangChain's text container: `page_content` + `metadata` |
| Embedding | A list of numbers representing a text's meaning |
| Grounding | Telling the LLM to answer only from the given context |
| Idempotent | Safe to run any number of times with the same result |
| Index (Pinecone) | A table of vectors with built-in similarity search |
| Ingestion | Getting data into the vector store: load, split, embed, store |
| `k` | How many chunks the retriever returns |
| LCEL | Joining Runnables with `\|`; output of one is input of the next |
| Retriever | Takes a question, returns relevant `Document`s |
| Runnable | Anything with `.invoke()`: prompts, LLMs, retrievers, parsers |
| `RunnablePassthrough.assign` | Keeps the input dict and adds a computed key |
| Serverless | Pinecone runs and scales the servers; pay per use |
| Upsert | Insert, or overwrite if the ID exists |
| Vector store | Database that stores embeddings and finds the nearest ones |
