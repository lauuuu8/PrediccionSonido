# app/services/audio_preprocessing.py
import io
import numpy as np
import librosa

from app.config import settings

N_MFCC = 40
MAX_LEN = 174
BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 6

def wav_bytes_to_cnn_lstm_input(wav_bytes: bytes) -> np.ndarray:

    # 1. Load audio
    audio_buffer = io.BytesIO(wav_bytes)
    y, sr = librosa.load(audio_buffer, sr=settings.target_sr)

    # 2. MFCC extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # 3. Padding / trimming
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, MAX_LEN - mfcc.shape[1])),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :MAX_LEN]

    # 4. Transpose to (time, mfcc)
    mfcc = mfcc.T   # shape (174, 40)

    # 5. Add batch & channel dimension
    x = np.expand_dims(mfcc, axis=(0, -1)).astype("float32")
    # shape: (1, 174, 40, 1)

    return x