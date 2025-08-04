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

        # Accepts 'documents' key
        pdf_path = body.get("documents")
        questions = body.get("questions", [])

        if not pdf_path:
            return {"error": "documents field is required in the JSON body"}

        if not questions:
            return {"error": "questions list is required in the JSON body"}

        # Download the PDF if URL
        if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
            response = requests.get(pdf_path)
            if response.status_code != 200:
                return {"error": f"Failed to download PDF: HTTP {response.status_code}"}
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_file.write(response.content)
            tmp_file.flush()
            tmp_pdf_path = tmp_file.name
        else:
            tmp_pdf_path = pdf_path  # Assume local file path

        # Extract text from PDF
        extractor = DataExtractor()
        documents = extractor.from_pdf(tmp_pdf_path)

        full_text = "\n".join([doc.text for doc in documents])

        # Chunk the text
        chunker = Chunking()
        refined_chunks = chunker.from_text(full_text)

        # Create vector index and upsert
        vector_db = VectorDB()
        vector_db.create_index()
        vector_db.upsert_chunks(refined_chunks)

        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return {"error": "GROQ_API_KEY not found in environment variables"}

        client = Groq(api_key=groq_api_key)

        instruction = (
            "Give ONLY the direct factual answer from the provided content. "
            "NO introductions or explanations unless specifically asked. "
            "If information is not found, say 'Not available'. "
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
                    chunk_text = hit.get("fields", {}).get("chunk_text", "No text available")
                    combined_content.append(chunk_text)

                search_result = "\n\n".join(combined_content)

                llm_input = instruction + f"\n\nQuery: {query_text}\n\n{search_result}"

                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "user", "content": llm_input}
                    ],
                    temperature=0.0,
                    max_completion_tokens=512
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
