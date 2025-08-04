import os
import requests
import tempfile
import time
import datetime
from flask import Flask, request, jsonify
from data_extraction import DataExtractor
from chunks import Chunking
from vectordb import VectorDB
from groq import Groq

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query_from_pdf_url():
    data = request.get_json()
    if not data or 'pdf_url' not in data or 'questions' not in data:
        return jsonify({"error": "Missing 'pdf_url' or 'questions'"}), 400

    pdf_url = data['pdf_url']
    questions = data['questions']

    try:
        # Download PDF
        response = requests.get(pdf_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download PDF"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_pdf_path = tmp_file.name

        # Step 1: Extract
        extractor = DataExtractor()
        documents = extractor.from_pdf(tmp_pdf_path)
        full_text = "\n".join([doc.text for doc in documents])

        # Step 2: Chunk
        chunker = Chunking()
        refined_chunks = chunker.from_text(full_text)

        # Step 3: Vector DB
        vector_db = VectorDB()
        vector_db.create_index()
        vector_db.upsert_chunks(refined_chunks)

        # Step 4: Answer questions
        client = Groq(api_key=os.getenv("GROQ_API_KEY", "your_key_here"))

        instruction = (
            'You are a Retrieval Augmentation Expert professional human. You are given a user query and the top 1 search result. '
            'Read both, then return a precise, single-line, factual answer. '
            'If the result is irrelevant, respond: "Not relevant to the document."'
        )

        answers = []
        for q in questions:
            pinecone_results = vector_db.query(q, top_k=1)

            chunk_text = "No relevant text found"
            if 'result' in pinecone_results and 'hits' in pinecone_results['result']:
                hits = pinecone_results['result']['hits']
                if hits:
                    chunk_text = hits[0]['fields'].get('chunk_text', chunk_text)

            # Call LLM
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": f"{instruction}\n\nQuery: {q}\n\nContext:\n{chunk_text}"
                }],
                temperature=0.5,
                max_completion_tokens=512,
            )
            answer = completion.choices[0].message.content.strip()
            answers.append(answer)

        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
