import os
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embeddings_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

#load vector store if it exists
def load_vector_store():
    try:
        embedding_model = get_embeddings_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading existing vector store...")
            vector_store = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
            return vector_store
        else:
            logger.warning("Vector store not found, creating a new one...")
    except Exception as e:
        error_message = CustomException("Failed to load an existing vector store", e)
        logger.error(str(error_message))


#function to create a new vector store
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks found")
        
        logger.info("Creating a new vector store...")

        embedding_model = get_embeddings_model()
        vector_store = FAISS.from_documents(text_chunks, embedding_model)
        vector_store.save_local(DB_FAISS_PATH)
        logger.info("Vector store created successfully")
        return vector_store
    except Exception as e:
        error_message = CustomException("Failed to create a new vector store", e)
        logger.error(str(error_message))