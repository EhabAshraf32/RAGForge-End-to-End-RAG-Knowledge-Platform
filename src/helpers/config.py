from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):

    APP_NAME: str
    APP_VERSION: str

    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int

    MONGODB_URL: str
    MONGODB_DATABASE: str

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    # Optional[...] is required: pydantic-settings validates defaults, so `str = None`
    # made every one of these keys mandatory in .env (e.g. COHERE_API_KEY)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    # OPENROTER_API_URL: str = None

    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_SIZE: Optional[int] = None
    # max characters of a single text sent to the EMBEDDING model (keep >= chunk_size)
    INPUT_DAFAULT_MAX_CHARACTERS: Optional[int] = None
    # max characters of a single message (system / context+query / history turn) sent to the GENERATION model
    GENERATION_DAFAULT_MAX_INPUT_CHARACTERS: int = 20000
    GENERATION_DAFAULT_MAX_TOKENS: Optional[int] = None
    GENERATION_DAFAULT_TEMPERATURE: Optional[float] = None
    # seconds before an OpenAI-compatible (e.g. OpenRouter) request is abandoned
    OPENAI_API_TIMEOUT: float = 120

    # client-side conversation history sent with /answer (stateless backend)
    RAG_HISTORY_MAX_MESSAGES: int = 6
    RAG_HISTORY_MAX_CHARACTERS: int = 3000

    # ---------------- security ----------------
    # HS256 signing key for access tokens (>= 32 chars). Generate: python -m helpers.security secret
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    # TA / admin account. Hash generated with: python -m helpers.security hash
    ADMIN_USERNAME: Optional[str] = None
    ADMIN_PASSWORD_HASH: Optional[str] = None
    ADMIN_TOKEN_EXPIRE_MINUTES: int = 60
    # students exchange the course access code (shared by the TA) for a student token
    STUDENT_ACCESS_CODE_HASH: Optional[str] = None
    # the only project (course) students may ask questions about
    STUDENT_PROJECT_ID: Optional[str] = None
    STUDENT_TOKEN_EXPIRE_MINUTES: int = 480
    # rate limits (0 disables a limit)
    LOGIN_RATE_LIMIT: int = 10                 # login attempts per client IP ...
    LOGIN_RATE_LIMIT_WINDOW_SECONDS: int = 900  # ... per 15 minutes
    STUDENT_ANSWER_RATE_LIMIT_PER_MINUTE: int = 5
    STUDENT_ANSWER_RATE_LIMIT_PER_HOUR: int = 60

    # frontends (e.g. Streamlit Cloud) that call the API server-side send the end user's IP in X-Client-IP
    # together with this secret; the IP is then used ONLY as a rate-limit key (never for authorization)
    TRUSTED_FRONTEND_SECRET: Optional[str] = None

    # ---------------- hosted deployment ----------------
    # Cloudflare Workers AI (EMBEDDING_BACKEND="CLOUDFLARE", EMBEDDING_MODEL_ID="@cf/baai/bge-m3")
    CLOUDFLARE_ACCOUNT_ID: Optional[str] = None
    CLOUDFLARE_API_TOKEN: Optional[str] = None
    # keep uploaded PDFs in MongoDB GridFS (needed when the server disk is ephemeral, e.g. Render free)
    STORE_FILES_IN_DB: bool = False

    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None
    # Qdrant server / Qdrant Cloud. If QDRANT_URL is set it is used instead of the local VECTOR_DB_PATH
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

    PRIMARY_LANG : str = "en"
    DEFAULT_LANG : str = "en"

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
