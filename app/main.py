import asyncio
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STORES ---
video_out_clients = set()
tts_clients = set()
# Almacenará el último frame recibido como BYTES (binario)
latest_video_frame: Optional[bytes] = None

# --- CONSTANTES FAKER ---
MOCK_OBJECTS = ["Silla", "Mesa", "Computadora", "Persona", "Escalera", "Puerta"]
# No necesitamos FAKE_IMAGE ya que el video es echo, pero lo dejamos por si acaso
# FAKE_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

# --- TAREA DE FONDO (GENERADOR TTS) ---
async def fake_stream_generator():
    """Solo se encarga de simular el TTS (Sigue siendo texto)."""
    counter = 0
    while True:
        # Lógica de TTS (Cada 3 segundos)
        if counter % 30 == 0 and tts_clients:
            obj = random.choice(MOCK_OBJECTS)
            dist = round(random.uniform(1.0, 5.0), 1)
            text_msg = f"{obj} a {dist} metros"
            
            for ws in list(tts_clients):
                try:
                    # TTS usa send_text()
                    await ws.send_text(text_msg)
                except:
                    tts_clients.remove(ws)

        counter += 1
        await asyncio.sleep(0.1) # 100ms

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fake_stream_generator())

# --------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------

@app.websocket("/video/in")
async def video_in(websocket: WebSocket):
    """
    Recibe el stream del cliente como BYTES, lo guarda y lo reenvía (ECHO).
    """
    await websocket.accept()
    global latest_video_frame
    
    try:
        while True:
            # *** RECIBIR BINARIO ***
            frame_data_bytes = await websocket.receive_bytes()
            
            # 1. ACTUALIZAR BUFFER
            latest_video_frame = frame_data_bytes
            processed_frame = latest_video_frame 

            # 2. BROADCAST a todos los clientes de /video/out
            if video_out_clients and processed_frame is not None:
                to_remove = []
                for ws in list(video_out_clients):
                    try:
                        # *** ENVIAR BINARIO ***
                        await ws.send_bytes(processed_frame)
                    except:
                        to_remove.append(ws)
                for ws in to_remove:
                    video_out_clients.remove(ws)
                    
    except WebSocketDisconnect:
        pass

@app.websocket("/video/out")
async def video_out(websocket: WebSocket):
    """El cliente escucha aquí para ver el video 'procesado' (binario)."""
    await websocket.accept()
    video_out_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1) # Keep alive
    except WebSocketDisconnect:
        video_out_clients.remove(websocket)

@app.websocket("/tts")
async def tts_socket(websocket: WebSocket):
    """El cliente escucha aquí para recibir instrucciones de voz (texto)."""
    await websocket.accept()
    tts_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        tts_clients.remove(websocket)