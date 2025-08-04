from fastapi import FastAPI, Request
from data_extraction import DataExtractor
from chunks import Chunking
from vectordb import VectorDB
from groq import Groq
import time
import json
import os
import tempfile
import requests
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

app = FastAPI()

@app.post("/hackrx/run")
async def hackrx_run(request: Request):
    start_time = time.time()
    try:
        body = await request.json()

        pdf_path = body.get("pdf_path")
        questions = body.get("questions", [])

        if not pdf_path:
            return {"error": "pdf_path is required in the JSON body"}

        if not questions:
            return {"error": "questions list is required in the JSON body"}

        # Check if pdf_path is a URL
        if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
            response = requests.get(pdf_path)
            if response.status_code != 200:
                return {"error": f"Failed to download PDF: HTTP {response.status_code}"}
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_file.write(response.content)
            tmp_file.flush()
            tmp_pdf_path = tmp_file.name
        else:
            tmp_pdf_path = pdf_path  # Assume it's a local path

        # Extract text from PDF
        extractor = DataExtractor()
        documents = extractor.from_pdf(tmp_pdf_path)

        full_text = "\n".join([doc.text for doc in documents])

        # Chunk the text
        chunker = Chunking()
        refined_chunks = chunker.from_text(full_text)

        # Create vector index and upsert chunks
        vector_db = VectorDB()
        vector_db.create_index()
        vector_db.upsert_chunks(refined_chunks)

        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return {"error": "GROQ_API_KEY not found in environment variables"}

        client = Groq(api_key=groq_api_key)

        instruction = (
            "Give ONLY the direct factual answer from the provided content. NO introductions like 'Based on' or 'The query asks'. "
            "NO question repetition. NO explanations unless specifically asked. Extract the exact information requested. "
            "If information is not found in the content, say 'Not available'. "
            "Be precise and use the exact wording from the source when possible.\n\n"
        )

        answers_list = []

        for query_text in questions:
            pinecone_results = vector_db.query(query_text, top_k=3)

            if (
                "result" in pinecone_results and
                "hits" in pinecone_results["result"] and
                pinecone_results["result"]["hits"]
            ):
                combined_content = []

                for hit in pinecone_results["result"]["hits"]:
                    score = hit.get("_score", 0)
                    chunk_text = hit.get("fields", {}).get("chunk_text", "No text available")
                    combined_content.append(f"Content (Score: {score:.4f}): {chunk_text}")

                search_result = "Search Results:\n" + "\n\n".join(combined_content)

                llm_input = instruction + f"Query: {query_text}\n\n{search_result}"

                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "user", "content": llm_input}
                    ],
                    temperature=0.0,
                    max_completion_tokens=512,
                    top_p=0.9,
                    stream=False,
                    stop=None,
                )

                answer = (
                    completion.choices[0]
                    .message.content.strip()
                    .replace("\\n", " ")
                    .replace("\n", " ")
                )
            else:
                answer = "Not available"

            answers_list.append(answer)

        elapsed = time.time() - start_time
        return {
            "answers": answers_list,
            "processing_time_seconds": round(elapsed, 2)
        }

    except Exception as e:
        return {"error": str(e)}
