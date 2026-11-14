from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    HUGGINGFACE = "HUGGINGFACE"
    # Cloudflare Workers AI through its OpenAI-compatible endpoint (e.g. @cf/baai/bge-m3 embeddings)
    CLOUDFLARE = "CLOUDFLARE"
    # Groq through its OpenAI-compatible endpoint
    GROQ = "GROQ"
    # OPENROUTER="OPENAI-OPENROUTER"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CoHereEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"

    DOCUMENT = "search_document"
    QUERY = "search_query"

class ChatRoleEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"
    