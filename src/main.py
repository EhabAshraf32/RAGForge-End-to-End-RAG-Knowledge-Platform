from fastapi import FastAPI
from routes import base, data, nlp, auth
from controllers import AuthController
import logging
from helpers.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templatess.template_parser import TemplateParser


app = FastAPI()

@app.on_event("startup")
async def startup_span():
    
    settings=get_settings()

    # security: fail closed - never serve the API without a token signing key
    auth_controller = AuthController()
    if not auth_controller.is_token_signing_configured():
        raise RuntimeError("JWT_SECRET_KEY is missing or shorter than 32 characters. "
                           "Generate one with: python -m helpers.security secret")
    if not auth_controller.is_admin_configured():
        logging.getLogger('uvicorn.error').warning("Admin login disabled: set ADMIN_USERNAME and ADMIN_PASSWORD_HASH")
    if not auth_controller.is_student_configured():
        logging.getLogger('uvicorn.error').warning("Student login disabled: set STUDENT_ACCESS_CODE_HASH and STUDENT_PROJECT_ID")

    app.mongo_conn=AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client= app.mongo_conn[settings.MONGODB_DATABASE]
    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings)


    # generation client
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    if app.generation_client is None:
        raise RuntimeError(f"Generation backend {settings.GENERATION_BACKEND!r} is unknown or not configured")
    app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    if app.embedding_client is None:
        raise RuntimeError(f"Embedding backend {settings.EMBEDDING_BACKEND!r} is unknown or not configured")
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)
    
    # vector_db client
    app.vectordb_client=vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )
    app.vectordb_client.connect()

    # template parser
    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )

@app.on_event("shutdown")
async def shutdown_span():
    app.mongo_conn.close()
    app.vectordb_client.disconnect()

# app.on_event("startup")(startup_span)
# app.on_event("shutdown")(shutdown_span)

# app.router.lifespan.on_startup.append(startup_db_client)
# app.router.lifespan.on_shutdown.append(shutdown_db_client)

app.include_router(base.base_router)
app.include_router(auth.auth_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)