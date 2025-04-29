Here’s the formal README you requested, crafted in the style of the Medium article but keeping it appropriate for a GitHub project:

⸻

Knowledge Graph Retrieval with Hierarchical Community Clustering

Author: Deniz Askin, Ph.D.
Edited by: ChatGPT o1-pro
License: © Deniz Askin

⸻

Overview

Retrieving knowledge from large datasets is challenging, especially when relevant concepts are scattered across multiple documents without explicit links. This project addresses the problem by building Knowledge Graphs enhanced through Hierarchical Community Detection and LLM-based Naming, significantly narrowing the search space for answering complex, cross-document questions.

We integrate:
	•	Web scraping (via Tavily),
	•	Knowledge Graph Construction (via Neo4j + SimpleKGPipeline),
	•	k-Nearest Neighbors and Leiden Community Detection (via scikit-learn and igraph),
	•	LLM summarization (via OpenAI GPT-4o),
	•	GraphRAG retrieval over clustered data.

The result is a system capable of scalable, semantically aware, and human-friendly information retrieval.

⸻

How It Works

1. Web Scraping and Data Collection

Using the Tavily API, the code retrieves multiple web documents based on user-defined queries.

2. Knowledge Graph Construction

Using Neo4j and the SimpleKGPipeline, scraped documents are chunked, embedded with sentence-transformers, and stored as nodes in a graph database, each labeled Chunk.

3. Vector Embedding and Indexing

Chunks are vectorized, and a vector index is built inside Neo4j, enabling similarity-based searches via embeddings.

4. Community Detection
	•	k-Nearest Neighbors is used to create a similarity graph of chunks.
	•	Leiden algorithm clusters chunks into communities based on similarity.

This organizes semantically similar chunks into dense groups.

5. Super-Community Formation

Communities are further clustered based on the similarity of their centroid embeddings, forming super-communities that represent broader topics.

This two-layered hierarchy reduces the search space significantly.

6. LLM-Based Naming

An LLM (GPT-4o) assigns descriptive names to:
	•	Each node (chunk),
	•	Each community (cluster),
	•	Each super-community (higher-level cluster).

This makes the graph navigable, interpretable, and ready for routing retrieval.

7. Retrieval and Question Answering (GraphRAG)

Upon receiving a user question, the system:
	•	Routes the query to the relevant community/super-community,
	•	Retrieves the most relevant nodes (chunks),
	•	Uses the GraphRAG framework to answer the question, citing the retrieved context.

⸻

Key Features
	•	🔍 Scalable Graph Retrieval: Using community-pruned retrieval for faster, focused querying.
	•	🧠 Semantic Clustering: Clusters nodes and communities based on meaning, not just surface similarity.
	•	✍️ Human-Friendly Naming: LLM-summarized labels make navigation intuitive.
	•	🛠 Automatic Pipeline: From web scraping to graph-based querying, everything is automated.
	•	🧩 Modular: Easily adaptable for other datasets, embedding models, or retrievers.

⸻

Example Use Case

Question: “What is the similarity between Leo Messi and System of a Down?”

Workflow:
	1.	Tavily scrapes information about Leo Messi, Till Lindemann, and System of a Down.
	2.	Documents are chunked and converted into graph nodes.
	3.	k-NN and Leiden detect communities: e.g., “Football Legends” and “Rock Bands”.
	4.	LLM names the communities accordingly.
	5.	GraphRAG restricts search to relevant communities.
	6.	The system outputs a precise comparison between Messi’s football achievements and System of a Down’s musical success.

Result:

“Lionel Messi and System of a Down both have significant achievements in their respective fields of entertainment…”

⸻

Why This Matters
	•	Efficiency: Queries are processed faster by searching only relevant graph regions.
	•	Accuracy: Thematic clustering increases the likelihood of finding the correct supporting context.
	•	Scalability: Works even for datasets spanning thousands of documents.
	•	Explainability: Hierarchically labeled nodes and communities make the system understandable to users and developers.

⸻

Technologies Used
	•	Neo4j: Knowledge graph database.
	•	Tavily: Web scraping API.
	•	OpenAI GPT-4o: LLM-based summarization and naming.
	•	Sentence-Transformers: Text embeddings.
	•	Scikit-learn: k-Nearest Neighbors.
	•	igraph + leidenalg: Graph-based community detection.
	•	GraphRAG: Contextual retriever over graphs.

⸻

Installation

pip install -r requirements.txt

Make sure to configure your .env file with:

NEO4J_URI=bolt://your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key



⸻

Running the Pipeline

python main.py

Modify the question, web_scraping_questions, and index_name in the if __name__ == "__main__": block to customize your run.

⸻

Citation

If you use this code or approach, please cite:

Askin, D. , Weiss, R. (2025). Optimizing Knowledge Retrieval with Hierarchical Clustering. Medium Article.
https://medium.com/@denizaskin/by-deniz-askin-and-rotem-weiss-27fdbdb75816
⸻

Final Thoughts

This two-step solution — combining first-level community detection and second-level super-community formation — provides a robust, scalable, and intelligent method for retrieving information from large, messy document collections. It mirrors how the human brain structures memory: clustering related pieces together to retrieve information quickly and meaningfully.

⸻
