# app/services/sagemaker_inference.py
import json
import numpy as np
import boto3

from app.config import settings

# Inicializamos el cliente de SageMaker Runtime.
runtime = boto3.client(
    "sagemaker-runtime",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    aws_session_token=settings.aws_session_token,
)

CLASS_NAMES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]


def call_sagemaker_cnn_lstm(x) -> dict:
    # Convertimos el array de numpy a lista para poder serializarlo a JSON
    payload = x.tolist()

    # Invocamos el endpoint de SageMaker
    response = runtime.invoke_endpoint(
        EndpointName=settings.sm_endpoint_name,
        Body=json.dumps(payload),
        ContentType="application/json",
    )

    # Leemos la respuesta
    result_str = response["Body"].read().decode("utf-8")
    result = json.loads(result_str)

    # Procesamos la respuesta para obtener las probabilidades
    # Algunos modelos devuelven {"predictions": [[...]]} y otros directamenet [[...]]
    if isinstance(result, dict) and "predictions" in result:
        probs = np.array(result["predictions"][0], dtype="float32")
    else:
        probs = np.array(result[0], dtype="float32")

    # Obtenemos la clase con mayor probabilidad
    class_id = int(np.argmax(probs))
    class_name = CLASS_NAMES[class_id]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "probs": probs.tolist(),
    }

