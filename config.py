import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt2-medium")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "200"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_K = int(os.getenv("TOP_K", "50"))
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "2"))
    CONSTRAINED_PHRASES = [" Facebook", " Twitter"]
  