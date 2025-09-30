import os

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import get_llm_model
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID, HF_TOKEN

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = (
    "You are a helpful medical expert that can answer medical questions\n"
    "in 2-3 lines using only the following context:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:\n"
)

def set_custom_prompt():
    try:
        logger.info("Setting custom prompt")
        prompt_template = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
        logger.info("Custom prompt set successfully")
        return prompt_template
    except Exception as e:
        error_message = CustomException("Failed to set custom prompt", e)
        logger.error(str(error_message))

def create_qa_chain():        
    try:
        logger.info("Loading vector store for context retrieval")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not found")

        logger.info("Vector store loaded successfully")

        llm = get_llm_model()

        if llm is None:
            raise CustomException("LLM model not found")
        
        prompt_template = set_custom_prompt()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template}
        )
        logger.info("QA chain created successfully")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to create QA chain", e)
        logger.error(str(error_message))