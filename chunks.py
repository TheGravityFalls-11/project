import os
from dotenv import load_dotenv
from chonkie import SentenceChunker, OverlapRefinery

load_dotenv()

class Chunking:
    def __init__(self):
        self.chunker = SentenceChunker(
            tokenizer_or_token_counter="character",  # Or "gpt2" etc.
            chunk_size=2048,
            chunk_overlap=128,
            min_sentences_per_chunk=2
        )

        self.refinery = OverlapRefinery(
            tokenizer_or_token_counter="character",
            context_size=0.25,
            method="prefix",
            merge=True
        )

    def from_text(self, text):
        chunks = self.chunker.chunk(text) 
        refined_chunks = self.refinery.refine(chunks)  
        return refined_chunks  
