from dotenv import load_dotenv
from os import getenv

load_dotenv()

OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = getenv("OPENROUTER_BASE_URL")