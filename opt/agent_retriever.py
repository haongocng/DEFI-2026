# Bạn cần cài đặt: pip install chromadb
import chromadb
from chromadb.utils import embedding_functions

class AgentRetriever:
    def __init__(self, gemini_client, knowledge_base):
        self.client = gemini_client
        self.kb = knowledge_base
        
        self.chroma_client = chromadb.PersistentClient(path="memory/chroma_db")
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="system_rules",
            metadata={"hnsw:space": "cosine"}
        )

    def sync_kb_to_vector_db(self):
        rules = self.kb.get_all_rules()
        if not rules: return

        ids = [str(i) for i in range(len(rules))]
        documents = [r['condition_description'] for r in rules]
        metadatas = [{"instruction": r['instruction']} for r in rules]
        
        embeddings = [self.client.get_embedding(doc) for doc in documents]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def retrieve_rules(self, user_context, top_k=3):
        user_vec = self.client.get_embedding(user_context)
        
        results = self.collection.query(
            query_embeddings=[user_vec],
            n_results=top_k
        )
        
        if results['metadatas']:
            return [m['instruction'] for m in results['metadatas'][0]]
        return []