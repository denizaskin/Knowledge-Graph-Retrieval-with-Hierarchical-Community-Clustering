
![1_EA4IAFp98ZUP_9SBO10dFA](https://github.com/user-attachments/assets/9bad53fc-2a40-4351-970d-03e774be0393)




📚 **Knowledge Graph Retrieval with Hierarchical Community Clustering**

Author: Deniz Askin, Ph.D.
License: © Deniz Askin

⸻

🧠 **Overview**

Retrieving knowledge from large datasets is challenging, especially when concepts are scattered across documents without explicit links.
This project addresses the problem by building Knowledge Graphs enhanced with Hierarchical Community Detection and LLM-based Naming, significantly narrowing the search space for answering complex, cross-document questions.

We integrate:
```bash
	•	🌐 Web scraping (via Tavily)
	•	📊 Knowledge Graph Construction (via Neo4j + SimpleKGPipeline)
	•	🤖 k-Nearest Neighbors and Leiden Community Detection (via scikit-learn and igraph)
	•	📝 LLM summarization (via OpenAI GPT-4o)
	•	🔎 GraphRAG retrieval over clustered data
```
Result: Scalable, Semantically Aware, and Human-Friendly Information Retrieval.

⸻

⚙️ **How It Works**
```bash
1. Web Scraping and Data Collection
	•	Retrieve multiple web documents using the Tavily API based on user queries.

2. Knowledge Graph Construction
	•	Chunk documents and store them as Chunk nodes in a Neo4j graph database using SimpleKGPipeline.

3. Vector Embedding and Indexing
	•	Vectorize chunks using sentence-transformers.
	•	Build a vector index in Neo4j to enable similarity-based search.

4. Community Detection
	•	Use k-Nearest Neighbors to construct a similarity graph.
	•	Cluster nodes into communities using the Leiden algorithm.

5. Super-Community Formation
	•	Cluster communities into super-communities by comparing their centroid embeddings.

6. LLM-Based Naming
	•	Use GPT-4o to assign descriptive names to:
	•	Each node (chunk)
	•	Each community (cluster)
	•	Each super-community

7. Retrieval and Question Answering
	•	Route user queries to relevant communities/super-communities.
	•	Retrieve top-k relevant nodes using GraphRAG.
	•	Generate concise answers using context.
```
⸻
```bash
✨ Key Features
	•	🔍 Scalable Graph Retrieval: Faster, focused querying using pruned graph sections.
	•	🧠 Semantic Clustering: Meaning-based organization of data, not just keywords.
	•	✍️ Human-Friendly Naming: Easy graph navigation with LLM-generated labels.
	•	🛠 Fully Automated Pipeline: From scraping to graph-based answering.
	•	🧩 Highly Modular: Plug-and-play different models or retrievers.
```
⸻

🧪 **Example Use Case**
```bash
Question: “What is the similarity between Leo Messi and System of a Down?”
```
```bash
Workflow:
	1.	Tavily scrapes data about Leo Messi, Till Lindemann, and System of a Down.
	2.	Documents are chunked and stored as graph nodes.
	3.	k-NN and Leiden detect communities like “Football Legends” and “Rock Bands”.
	4.	LLM assigns descriptive names to communities.
	5.	GraphRAG focuses search on these communities.
	6.	System outputs an insightful answer comparing Messi’s and System of a Down’s achievements.
```
**Sample Output:**

“Lionel Messi and System of a Down both have significant achievements in their respective fields of entertainment…”

⸻

🚀 **Why This Matters**
```bash
	•	⚡ Efficiency: Pruned retrieval = faster response times.
	•	🎯 Accuracy: Thematic clustering improves context retrieval.
	•	🌍 Scalability: Handles datasets with thousands of documents.
	•	🧩 Explainability: Clear, labeled graph hierarchy aids users and developers.
```
⸻

🛠 **Technologies Used**
```bash
	•	Neo4j – Knowledge graph database
	•	Tavily – Web scraping API
	•	OpenAI GPT-4o – LLM for summarization
	•	Sentence-Transformers – Text embeddings
	•	Scikit-learn – k-Nearest Neighbors
	•	igraph + leidenalg – Graph-based community detection
	•	GraphRAG – Contextual retrieval over graphs
```
⸻

📦 **Installation**
```bash
pip install -r requirements.txt
```
Configure your .env file:
```bash
NEO4J_URI=bolt://your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```


⸻

▶️ **Running the Pipeline**
```bash
python main.py
```
Modify:
```bash
	•	question
	•	web_scraping_questions
	•	index_name
```
inside the **if __name__ == "__main__"** block to customize.

⸻

📚 **Citation**

If you use this work, please cite:

Askin, D., Weiss, R. (2025). Optimizing Knowledge Retrieval with Hierarchical Clustering. Medium Article.

⸻

💬 Final Thoughts

By combining first-level community detection and second-level super-community formation, this project creates an intelligent, scalable, and human-readable knowledge retrieval system — mirroring the brain’s efficiency in memory organization.
