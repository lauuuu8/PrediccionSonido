from typing import List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    sm_endpoint_name: str
    aws_region: str = "us-east-1"
    
    # Credenciales (Access y Secret)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    # Agregamos el Token de Sesión (Opcional)
    aws_session_token: str | None = None
    
    target_sr: int
    n_mels: int
    allowed_origins: str = "*"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
