from langchain_groq import ChatGroq
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import os

logger = get_logger(__name__)

def get_llm_model(model_name: str = "llama-3.1-8b-instant"):
    try:
        logger.info("Loading LLM from Groq using LLaMA3 model...")

        llm = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.3,
            max_tokens=256,
        )

        logger.info("LLM loaded successfully from Groq.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load an LLM from Groq", e)
        logger.error(str(error_message))
        return None