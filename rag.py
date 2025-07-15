from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from tenacity import retry, wait_exponential, stop_after_attempt
from tenacity.retry import retry_if_exception_type 
from google.api_core.exceptions import ResourceExhausted 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time
import os
import dotenv
import json
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
import uvicorn


# Load environment variables from .env file
dotenv.load_dotenv()

app = FastAPI()

video_dict: Dict[str, dict] = {}
faiss_dbs: Dict[str, any] = {} 

class QARequest(BaseModel):
    video_id: str
    question: str

def getVideoData(video_id):
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    # print(f"Using YouTube API Key: {API_KEY}")
    print(video_id)
    if not API_KEY:
        raise ValueError("YOUTUBE_API_KEY environment variable not set.")
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    try:
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        # print(transcript_list)
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print(f"Transcript for video {video_id} retrieved successfully.")
        request = youtube.videos().list(
            part='snippet',
            id=video_id
        )
        response = request.execute()

        if not response["items"]:
            print("Video not found.")
            return None
        
        snippet = response["items"][0]["snippet"]
        metadata = {
            "title": snippet.get("title", "N/A"),
            "publishedAt": snippet.get("publishedAt", "N/A"),
            "channelTitle": snippet.get("channelTitle", "N/A")
        }

        metadata_str = json.dumps(metadata, indent=2)

      
        video_data = {
            "video_id": video_id,
            "metadata": metadata_str,
            "transcript": transcript
        }
        # print(video_data)
        return video_data
        # return transcript

    except TranscriptsDisabled:
      print("No captions available for this video.")
    except NoTranscriptFound:
      print("Transcript not found for the given language/video.")
    except Exception as e:
      print(f"An error occurred: {e}")
  


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    reraise=True,
    retry=retry_if_exception_type(ResourceExhausted) # Correct way to specify the exception for retry
)
def get_embeddings_with_retry(texts: list[str]):
    """
    Helper function to get embeddings with retry logic for quota exhaustion.
    Uses embed_documents for batching.
    
    """
    
    embedding_model_name = "models/embedding-001" 
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings_instance = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=api_key) 
    return embeddings_instance.embed_documents(texts) 




def vectordb_loading(transcript):
    batch_size = 10  # Adjust this based on your API's rate limits and testing
    faiss_index = None # Initialize FAISS index as None
    long_text = transcript
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    dummy_chunks_text = text_splitter.split_text(long_text)
# Convert text chunks to LangChain Document objects if your original 'chunks' are Documents
    chunks = [Document(page_content=chunk_text) for chunk_text in dummy_chunks_text]

    print(f"Total chunks to process: {len(chunks)}")

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = [doc.page_content for doc in batch_chunks] # Extract text content for embedding
    
        print(f"Processing batch {int(i/batch_size) + 1}/{(len(chunks) + batch_size - 1) // batch_size} (chunks {i} to {min(i + batch_size, len(chunks)) - 1})...")
    
        try:
            # Get embeddings for the current batch using the retry wrapper
            batch_embeddings = get_embeddings_with_retry(batch_texts)
            embedding_model_name = "models/embedding-001" 
            api_key = os.getenv("GOOGLE_API_KEY")
            embeddings_instance = GoogleGenerativeAIEmbeddings(model=embedding_model_name, google_api_key=api_key) 
            # Create or add to the FAISS index
            if faiss_index is None:
                # First batch: create the index
                # IMPORTANT: Pass the 'embeddings_instance' itself, not the function
                faiss_index = FAISS.from_embeddings(
                    text_embeddings=list(zip(batch_texts, batch_embeddings)),
                    embedding=embeddings_instance 
                )
                print("FAISS index created with the first batch.")
            else:
                # Subsequent batches: add to the existing index
                # FAISS.add_texts expects the raw texts and pre-computed embeddings
                faiss_index.add_texts(
                    texts=batch_texts,
                    embeddings=batch_embeddings
                )
                print(f"Added {len(batch_chunks)} chunks to FAISS index.")
    
        except ResourceExhausted as e:
            print(f"Quota exhausted for batch {int(i/batch_size) + 1} after all retries. Consider increasing quota or reducing batch size/delay. Error: {e}")
            # The @retry decorator will handle the waiting and retrying.
            # If it exhausts all retries, the exception will be re-raised and caught here.
            break # Or handle further based on your application's needs
        except Exception as e:
            print(f"An unexpected error occurred during embedding or indexing for batch {int(i/batch_size) + 1}: {e}")
            break # Stop processing on other errors
    
        # Introduce a small delay between batches to further mitigate rate limits
        time.sleep(2) # Wait for 2 seconds before processing the next batch
    return faiss_index


def giveAnswers(question, video_data,retrieved_docs):
    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context with detailed explanation.
      If the context is insufficient, just say you don't know.
      
      metadata: {metadata}

      {context}
      Question: {question}
    """,
    input_variables = ['context','metadata','question']
)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    metadata = video_data["metadata"]
    prompt_text = prompt.format(context=context_text, metadata=metadata, question=question)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: 
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    llm=ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=api_key)
    llm_response = llm.invoke(prompt_text)
    answer = llm_response.content
    return answer


video_dict = {}
faiss_dbs = {}

@app.post("/ask")
async def ask_question(payload: QARequest):
    video_id = payload.video_id
    question = payload.question

    if video_id in video_dict:
        video_data = video_dict[video_id]
        vectordb = faiss_dbs[video_id]
        print(f"[Using Cached Video Data] {video_id}")
    else:
        video_data = getVideoData(video_id)
        if not video_data:
            return {"answer": "Video data not found or transcript unavailable."}
        
        vectordb = vectordb_loading(video_data["transcript"])
        video_dict[video_id] = video_data
        faiss_dbs[video_id] = vectordb

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})  
    retrieved_docs = retriever.invoke(question)
    print(f"[Retrieved Docs] {retrieved_docs}")

    answer = giveAnswers(question, video_data, retrieved_docs)
    return {"answer": answer}


