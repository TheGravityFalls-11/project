import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class VectorDB:
    def __init__(self, api_key=None, index_name="developer-quickstart-py"):
        if api_key is None:
            api_key = os.getenv("PINECONE_API_KEY")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = None
    
    def create_index(self, cloud="aws", region="us-east-1", model="llama-text-embed-v2", field_map=None):
        if field_map is None:
            field_map = {"text": "chunk_text"}
        
        if not self.pc.has_index(self.index_name):
            self.pc.create_index_for_model(
                name=self.index_name,
                cloud=cloud,
                region=region,
                embed={
                    "model": model,
                    "field_map": field_map
                }
            )
        
        self.index = self.pc.Index(self.index_name)
        return self.index
    
    def get_index(self):
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        return self.index
    
    def upsert_chunks(self, chunks, namespace="__default__", batch_size=90):
        index = self.get_index()
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            records = []
            for j, chunk in enumerate(batch_chunks):
                records.append({
                    "_id": f"chunk_{i + j}",
                    "chunk_text": str(chunk),
                    "chunk_id": i + j
                })
            
            index.upsert_records(namespace, records)
    
    def query(self, query_text, namespace="__default__", top_k=3):
        index = self.get_index()
        return index.search(
            namespace=namespace,
            query={
                "inputs": {"text": query_text},
                "top_k": top_k
            },
            fields=["chunk_text", "chunk_id"]
        )
    
    def delete_index(self):
        if self.pc.has_index(self.index_name):
            self.pc.delete_index(self.index_name)