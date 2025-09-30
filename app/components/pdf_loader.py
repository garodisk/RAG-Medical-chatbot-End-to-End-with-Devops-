import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files(data_path: str):
    try:

        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data path {DATA_PATH} does not exist")

        logger.info(f"Loading PDF files from {DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)

        documents = loader.load()

        if not documents:
            raise CustomException(f"No PDF files found in {DATA_PATH}")
        else:
            logger.info(f"Loaded {len(documents)} PDF files")

        return documents

    except Exception as e:
        error_message =  CustomException("Failed to load PDF files", e)
        logger.error(str(error_message))
        return []


def create_text_chunks(documents: list):
    try:
        if not documents:
            raise CustomException("No documents found")
        
        logger.info(f"Creating text chunks from {len(documents)} documents")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        text_chunks = text_splitter.split_documents(documents)

        if not text_chunks:
            raise CustomException("No text chunks found")
        else:
            logger.info(f"Created {len(text_chunks)} text chunks")

        return text_chunks

    except Exception as e:
        error_message = CustomException("Failed to create text chunks", e)
        logger.error(str(error_message))
        return []