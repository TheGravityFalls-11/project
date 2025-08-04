from data_extraction import DataExtractor
from chunks import Chunking
from vectordb import VectorDB
import datetime
from groq import Groq
import time
import json
import os
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    start_time = time.time()
    
    pdf_path = input("Enter PDF file path: ").strip()
    
    try:
        extractor = DataExtractor()
        documents = extractor.from_pdf(pdf_path)
        print(f"[Time] After PDF extraction: {time.time() - start_time:.2f} seconds")
        
        full_text = "\n".join([doc.text for doc in documents])
        
        chunker = Chunking()
        refined_chunks = chunker.from_text(full_text)
        print(f"[Time] After chunking: {time.time() - start_time:.2f} seconds")
        
        vector_db = VectorDB()
        vector_db.create_index()
        print(f"[Time] After index creation: {time.time() - start_time:.2f} seconds")
        vector_db.upsert_chunks(refined_chunks)
        print(f"[Time] After upserting chunks: {time.time() - start_time:.2f} seconds")
        
        print(f"Successfully indexed {len(refined_chunks)} chunks to Pinecone")
        
        # Get API key from environment variable
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")
        
        client = Groq(api_key=groq_api_key)
        
        instruction = (
            'Give ONLY the direct factual answer from the provided content. NO introductions like "Based on" or "The query asks". '
            'NO question repetition. NO explanations unless specifically asked. Extract the exact information requested. '
            'If information is not found in the content, say "Not available". '
            'Be precise and use the exact wording from the source when possible.\n\n'
        )
        
        json_input = input("Enter JSON: ").strip()
        
        while json_input.startswith("Enter JSON:"):
            json_input = json_input[11:].strip()
        
        try:
            input_data = json.loads(json_input)
            questions = input_data.get("questions", [])
            
            if not questions:
                print("No questions found in JSON input")
                return
                
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format: {e}")
            return
        
        answers_list = []
        
        for idx, query_text in enumerate(questions, 1):
            q_start = time.time()
            
            # Retrieve top 3 chunks for better context coverage
            pinecone_results = vector_db.query(query_text, top_k=3)
            
            if 'result' in pinecone_results and 'hits' in pinecone_results['result'] and pinecone_results['result']['hits']:
                # Combine multiple chunks for better accuracy
                combined_content = []
                total_score = 0
                
                for hit in pinecone_results['result']['hits']:
                    score = hit.get('_score', 0)
                    chunk_text = hit.get('fields', {}).get('chunk_text', 'No text available')
                    combined_content.append(f"Content (Score: {score:.4f}): {chunk_text}")
                    total_score += score
                
                search_result = f"Search Results:\n" + "\n\n".join(combined_content)
                
                llm_input = instruction + f"Query: {query_text}\n\n{search_result}"
                
                llm_start = time.time()
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": llm_input
                        }
                    ],
                    temperature=0.0,  # Lower temperature for more consistent/accurate responses
                    max_completion_tokens=512,
                    top_p=0.9,  # Slightly lower top_p for more focused responses
                    stream=False,
                    stop=None,
                )
                
                answer = completion.choices[0].message.content.strip().replace('\\n', ' ').replace('\n', ' ')
                
            else:
                answer = "Not available"
            
            answers_list.append(answer)
        
        result_json = {"answers": answers_list}
        print(json.dumps(result_json, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()