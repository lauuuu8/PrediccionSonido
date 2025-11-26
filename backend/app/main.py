from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.config import settings
from app.schemas import PredictionResponse
from app.services.audio_preprocessing import wav_bytes_to_cnn_lstm_input
from app.services.sagemaker_inference import call_sagemaker_cnn_lstm

app = FastAPI(title="API de predicción de ruidos")

# Procesar los orígenes permitidos (separar por comas si es una lista en string)
origins_list = [origin.strip() for origin in settings.allowed_origins.split(",")]

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predecir", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    Recibe un archivo .wav, lo preprocesa y llama al endpoint de SageMaker.
    Devuelve la clase predicha y las probabilidades.
    """
    try:
        if not file.filename.lower().endswith(".wav"):
            return JSONResponse(
                status_code=400,
                content={"error": "Solo se aceptan archivos .wav"},
            )

        # 1. Leer bytes del archivo
        wav_bytes = await file.read()

        # 2. Preprocesar → (1, T, 128)
        x = wav_bytes_to_cnn_lstm_input(wav_bytes)

        # 3. Llamar a SageMaker
        result = call_sagemaker_cnn_lstm(x)

        # 4. Construir respuesta tipada
        return PredictionResponse(
            filename=file.filename,
            class_id=result["class_id"],
            class_name=result["class_name"],
            probs=result["probs"],
        )

    except Exception as e:
        # Es buena práctica imprimir el error en los logs del servidor también
        print(f"Error en predicción: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
