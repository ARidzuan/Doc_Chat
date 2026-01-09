#Mod 1.
V1 Tracker implemented using Local Tracking
    Pipeline_chatbot_v1\web\metrics.html
    Pipeline_chatbot_v1\tracking

Tracks [Model name,Prompt and response text,Token counts (estimated),Latency (ms),Timestamps,Errors,Temperature]

Data is stored in `data/llm_tracking.db`

#Mod 2. 
Added segmentation of documentation based on thematics (defined in config.py)
When adding new doc, run test_segmentation.py build in order to build Chroma DB files based on topics specified in 

Mod3.
Adding cache based on previous queries. Allows to reaccess previously used items.
Limitation, limit is based on text, so same text with relaxation on cap locks.
To integrate similar but differnet queries, LLM systems can be used but is long

For instance, queries like What is the capital of France?, Tell me the name of the capital of France?, and What The capital of France is? all convey the same intent and should be identified as the same question.

Mod4
Upgraded to newer reranker model for sentence-transformer

"cross-encoder/BAAI/bge-reranker-large"
prev : "cross-encoder/ms-marco-MiniLM-L-6-v2"

Mod5
Added Chainlit Simple implementation

Mod6
Added Chainlit implementation with PERSISTENCE [FIXED - Jan 7, 2026]
- Multi-user support with authentication
- Chat history persistence using SQLAlchemy
- Resume previous conversations
- Create new chats without deleting old history
- Backend REST API for chat management
- Export and search functionality
- Scalable architecture (SQLite → PostgreSQL)


Mod 7
Added User Sign up for simple password managing

