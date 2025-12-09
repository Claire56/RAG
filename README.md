# Insurance RAG System - Knowledge Base Assistant

A comprehensive Retrieval-Augmented Generation (RAG) system designed for insurance companies in the insurance industry. This system builds an intelligent knowledge base from documentation, policy materials, and customer service guides, enabling accurate and up-to-date responses to insurance-related queries.

## 🎯 Learning Objectives

This project teaches you:

### RAG Pipeline Fundamentals
1. **Embeddings**: Converting text into numerical vectors for semantic search
2. **Vector Databases**: Storing and querying high-dimensional vectors (ChromaDB, Pinecone, Weaviate)
3. **Chunking Strategies**: Breaking documents into optimal-sized pieces
4. **Re-ranking**: Improving retrieval quality with cross-encoders
5. **Knowledge Freshness**: Detecting and flagging outdated content

### Cost & Latency Optimization
1. **Token Counting**: Understanding API usage and costs
2. **Caching**: Using Redis to cache embeddings and LLM responses
3. **Request Batching**: Efficiently processing multiple queries
4. **Model Comparison**: Evaluating trade-offs between GPT-4, GPT-3.5, and local models

## 📚 Core Concepts

### What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:
- **Retrieval**: Finding relevant documents from a knowledge base
- **Augmentation**: Adding retrieved context to prompts
- **Generation**: Using LLMs to generate answers based on retrieved context

**Why RAG?**
- Reduces hallucinations by grounding answers in source documents
- Enables up-to-date information beyond training cutoff
- Provides source citations for transparency
- More cost-effective than fine-tuning for domain knowledge

### Embeddings

**Embeddings** are numerical representations of text that capture semantic meaning:
- Similar texts have similar embedding vectors
- Enables semantic search (finding relevant content by meaning, not just keywords)
- Common models: OpenAI `text-embedding-ada-002`, `text-embedding-3-small/large`

**Example:**
```
"auto insurance claim" → [0.123, -0.456, 0.789, ...]
"car insurance filing" → [0.125, -0.454, 0.791, ...]  # Similar!
"restaurant menu"      → [-0.234, 0.567, -0.123, ...]  # Different!
```

### Vector Databases

**Vector databases** efficiently store and search millions of embeddings:
- **ChromaDB**: Open-source, lightweight, easy to use
- **Pinecone**: Managed, scalable, production-ready
- **Weaviate**: GraphQL-based, supports hybrid search

**Key Operations:**
- **Insert**: Add document embeddings with metadata
- **Query**: Find similar embeddings (cosine similarity, dot product)
- **Filter**: Combine vector search with metadata filters

### Chunking Strategies

**Why chunk documents?**
- LLM context windows are limited (GPT-4: 128k tokens, GPT-3.5: 16k tokens)
- Better retrieval when chunks match query granularity
- Balance: too small = lose context, too large = irrelevant content

**Strategies:**
1. **Fixed-size chunking**: Simple, predictable sizes (e.g., 500 tokens)
2. **Sentence-aware chunking**: Split at sentence boundaries
3. **Semantic chunking**: Group semantically related sentences
4. **Recursive chunking**: Hierarchical splitting (paragraph → sentence → word)

### Re-ranking

**Problem**: Vector search returns many candidates, but top-K might not be best
**Solution**: Re-rank results using cross-encoder models (BGE-reranker, Cohere)

**Two-stage approach:**
1. **Retrieval**: Fast vector search returns 50-100 candidates
2. **Re-ranking**: Slower but more accurate model ranks top 5-10

### Knowledge Freshness

**Challenge**: Documentation changes over time, cached embeddings become stale
**Solution**: Track source update dates and flag outdated content

**Implementation:**
- Store metadata: `last_updated`, `version`, `source_url`
- Compare against known update dates
- Flag or re-embed outdated chunks

## 🏗️ Project Structure

```
InsuranceRAG/
├── README.md                    # This comprehensive guide
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
│
├── scraper/                     # Web scraping module
│   ├── __init__.py
│   ├── web_scraper.py          # Scrape documentation sites
│   └── content_extractor.py    # Extract and clean text
│
├── chunking/                    # Text chunking strategies
│   ├── __init__.py
│   ├── chunker.py              # Main chunking logic
│   └── strategies.py           # Different chunking methods
│
├── embeddings/                  # Embedding generation
│   ├── __init__.py
│   ├── embedder.py             # Generate embeddings
│   └── providers.py            # Support multiple embedding models
│
├── vector_db/                   # Vector database integration
│   ├── __init__.py
│   ├── chromadb_client.py      # ChromaDB implementation
│   └── base.py                 # Abstract base class
│
├── reranking/                   # Re-ranking module
│   ├── __init__.py
│   └── reranker.py             # Re-rank retrieved results
│
├── query_interface/             # Query and generation
│   ├── __init__.py
│   ├── rag_pipeline.py         # Main RAG orchestration
│   ├── query_processor.py      # Process user queries
│   └── prompt_templates.py     # RAG prompt templates
│
├── cache/                       # Caching layer
│   ├── __init__.py
│   ├── redis_cache.py          # Redis caching for embeddings/responses
│   └── cache_manager.py        # Cache management utilities
│
├── optimization/                # Cost & latency optimization
│   ├── __init__.py
│   ├── token_counter.py        # Count tokens and estimate costs
│   ├── latency_tracker.py      # Track request latency
│   ├── cost_calculator.py      # Calculate API costs
│   └── model_comparison.py     # Compare different models
│
├── freshness/                   # Knowledge freshness checks
│   ├── __init__.py
│   └── freshness_checker.py    # Check and flag outdated content
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   └── logger.py               # Logging utilities
│
├── data/                        # Data storage
│   ├── raw/                    # Raw scraped content
│   ├── processed/              # Processed chunks
│   └── metadata.json           # Document metadata with update dates
│
├── logs/                        # Application logs
│
└── main.py                      # Main entry point
```

## 🚀 Quick Start

### 1. Installation

```bash
cd InsuranceRAG
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file from `.env.example`:

```bash
# OpenAI (for embeddings and GPT models)
OPENAI_API_KEY=your_openai_key_here

# Redis (for caching) - optional, will use in-memory cache if not set
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Ollama (for local Llama 3) - optional
OLLAMA_BASE_URL=http://localhost:11434

# Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
VECTOR_DB=chromadb
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 3. Build the Knowledge Base

```bash
# Scrape documentation (example: insurance policy documentation)
python main.py scrape --url https://example-insurance.com/documentation/policies/

# Or use local files
python main.py index --directory ./data/raw/policies/

# Process and embed documents
python main.py embed
```

### 4. Query the System

```bash
# Interactive query
python main.py query "What is the deductible for comprehensive coverage?"

# With model comparison
python main.py query "How do I file a claim?" --compare-models

# Check knowledge freshness
python main.py check-freshness
```

## 💡 Use Cases for Insurance Companies

### 1. Customer Service Chatbot
- Answer policy questions instantly
- Provide accurate claim filing procedures
- Explain coverage details

### 2. Agent Knowledge Base
- Help agents find answers quickly
- Ensure consistent information across team
- Reduce training time for new agents

### 3. Policy Documentation Search
- Search across thousands of policy documents
- Find relevant clauses and conditions
- Compare different policy types

### 4. Regulatory Compliance
- Track regulation updates
- Ensure documentation is current
- Flag outdated policies

## 🔧 Key Components Explained

### Web Scraper (`scraper/web_scraper.py`)

Scrapes documentation websites and extracts clean text:
- Handles JavaScript-rendered content
- Extracts metadata (title, date, URL)
- Cleans HTML and formatting

### Chunker (`chunking/chunker.py`)

Implements multiple chunking strategies:
- **Fixed-size**: Split by token count
- **Sentence-aware**: Respect sentence boundaries
- **Semantic**: Group related sentences
- **Recursive**: Hierarchical splitting

### Embedder (`embeddings/embedder.py`)

Generates embeddings for text chunks:
- Supports OpenAI, Cohere, HuggingFace models
- Batch processing for efficiency
- Handles rate limits and retries

### Vector DB (`vector_db/chromadb_client.py`)

Stores and queries embeddings:
- Create collections with metadata
- Similarity search with filters
- Update and delete operations

### RAG Pipeline (`query_interface/rag_pipeline.py`)

Main orchestration:
1. Embed user query
2. Retrieve similar chunks from vector DB
3. Re-rank results
4. Augment prompt with context
5. Generate answer with LLM

### Freshness Checker (`freshness/freshness_checker.py`)

Monitors content staleness:
- Compares chunk timestamps with source dates
- Flags outdated content
- Triggers re-embedding when needed

### Cache Manager (`cache/redis_cache.py`)

Caches expensive operations:
- Embeddings (same text = same embedding)
- LLM responses (frequent queries)
- Retrieval results

### Cost Calculator (`optimization/cost_calculator.py`)

Tracks API usage:
- Token counting (input + output)
- Cost estimation per request
- Cumulative usage tracking

## 📊 Cost & Latency Optimization

### Token Counting

**Why it matters:**
- OpenAI charges per token (input and output)
- GPT-4: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- GPT-3.5: ~$0.0015 per 1K input tokens, ~$0.002 per 1K output tokens

**Example:**
```python
# Query: "What is my deductible?"
# Input tokens: 50 (query + context)
# Output tokens: 100 (response)
# GPT-3.5 cost: (50/1000 * $0.0015) + (100/1000 * $0.002) = $0.000275
```

### Caching Strategy

**What to cache:**
1. **Embeddings**: Same text → same embedding (save API calls)
2. **LLM Responses**: Frequent queries → cache responses
3. **Retrieval Results**: Similar queries → reuse top chunks

**Cache TTL:**
- Embeddings: Never expire (immutable)
- LLM Responses: 24 hours (may need updates)
- Retrieval: 1 hour (documents may update)

### Model Comparison

**GPT-4 vs GPT-3.5 vs Llama 3:**

| Model      | Cost (per 1K tokens) | Latency | Quality | Best For              |
|------------|---------------------|---------|---------|-----------------------|
| GPT-4      | ~$0.03 input        | ~2-3s   | Highest | Complex reasoning     |
| GPT-3.5    | ~$0.0015 input      | ~0.5s   | Good    | Simple queries        |
| Llama 3    | Free (local)        | ~1-2s   | Good    | Cost-sensitive, privacy|

**When to use what:**
- **GPT-4**: Complex questions requiring reasoning
- **GPT-3.5**: Simple Q&A, high-volume queries
- **Llama 3**: Cost-sensitive, privacy-critical, offline use

## 🎓 Learning Path

### Week 1: Understanding RAG Basics
1. Learn about embeddings and vector similarity
2. Implement simple chunking
3. Build basic retrieval system

### Week 2: Vector Databases
1. Set up ChromaDB
2. Implement semantic search
3. Add metadata filtering

### Week 3: Advanced Retrieval
1. Implement re-ranking
2. Experiment with chunking strategies
3. Optimize retrieval quality

### Week 4: Optimization
1. Add caching layer
2. Implement token counting
3. Compare model performance

### Week 5: Production Features
1. Knowledge freshness checks
2. Error handling and retries
3. Monitoring and logging

## 📈 Example Workflow

### 1. User Query
```
"How do I file a comprehensive auto insurance claim?"
```

### 2. Embedding Generation
```
Query → Embedding Vector: [0.123, -0.456, ...]
```

### 3. Vector Search
```
Search vector DB → Find top 10 similar chunks
```

### 4. Re-ranking
```
Cross-encoder model → Rank and select top 3 chunks
```

### 5. Prompt Augmentation
```
System: "You are an insurance assistant..."
Context: [Retrieved chunks about claim filing]
User: "How do I file a comprehensive auto insurance claim?"
```

### 6. LLM Generation
```
Generate answer based on context
```

### 7. Response + Sources
```
Answer: "To file a comprehensive claim..."
Sources: [URL1, URL2, URL3]
```

## 🔍 Monitoring & Debugging

### Metrics to Track
- **Retrieval Quality**: Are retrieved chunks relevant?
- **Response Accuracy**: Are answers correct?
- **Latency**: Query → Response time
- **Cost**: Tokens and API costs per query
- **Cache Hit Rate**: Percentage of cached responses
- **Freshness**: Number of outdated chunks

### Logging
All operations are logged:
- Query and response
- Retrieved chunks
- Token usage
- Latency
- Cache hits/misses

## 🚨 Common Pitfalls & Solutions

### 1. Chunks Too Small/Large
**Problem**: Poor retrieval quality
**Solution**: Experiment with chunk sizes (200-1000 tokens)

### 2. Irrelevant Retrieval
**Problem**: Vector search returns wrong chunks
**Solution**: Use re-ranking, improve embeddings, add metadata filters

### 3. Outdated Information
**Problem**: Documents updated but embeddings stale
**Solution**: Implement freshness checks, re-embed periodically

### 4. High Costs
**Problem**: Too many API calls
**Solution**: Cache aggressively, use GPT-3.5 for simple queries

### 5. Slow Latency
**Problem**: Long response times
**Solution**: Cache responses, batch embeddings, use faster models

## 📚 Further Reading

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083)

## 🎯 Next Steps

1. **Experiment**: Try different chunking strategies
2. **Optimize**: Measure and improve latency/cost
3. **Scale**: Move to Pinecone for larger datasets
4. **Enhance**: Add hybrid search (vector + keyword)
5. **Deploy**: Build API endpoint or web interface

## 🤝 Contributing

This is a learning project. Feel free to:
- Add more chunking strategies
- Support additional vector databases
- Implement hybrid search
- Add evaluation metrics (retrieval accuracy, answer quality)

---

**Built for learning RAG pipelines and optimization techniques in a real-world insurance context.**
