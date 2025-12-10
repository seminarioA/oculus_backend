import asyncio
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from vision_processor import VisionProcessor

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Global variables
vision_processor: VisionProcessor = None
video_out_clients: Set[WebSocket] = set()
tts_clients: Set[WebSocket] = set()
executor: ThreadPoolExecutor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    global vision_processor, executor

    # Startup
    LOGGER.info("Starting up FastAPI server...")
    try:
        # Initialize thread pool executor
        executor = ThreadPoolExecutor(max_workers=4)
        LOGGER.info("Thread pool executor initialized")

        # Load VisionProcessor
        LOGGER.info("Loading VisionProcessor...")
        vision_processor = VisionProcessor()
        LOGGER.info("VisionProcessor loaded successfully")

    except Exception as e:
        LOGGER.error(f"Failed to initialize application: {e}")
        raise

    yield

    # Shutdown
    LOGGER.info("Shutting down FastAPI server...")
    if executor:
        executor.shutdown(wait=True)
        LOGGER.info("Thread pool executor shut down")


# Create FastAPI app
app = FastAPI(
    title="Vision Processing API",
    description="FastAPI server for YOLOv8 object detection with Depth Pro depth estimation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware - configurable for production
# For production, set ALLOWED_ORIGINS environment variable with comma-separated domains
# Default allows all origins for development, but should be restricted in production
allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
if not allowed_origins:
    LOGGER.warning(
        "ALLOWED_ORIGINS is empty, defaulting to allow all origins (*). "
        "Set ALLOWED_ORIGINS for production deployment."
    )
    allowed_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True if allowed_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_frame_sync(frame_bytes: bytes) -> Tuple[bytes, Optional[str]]:
    """Process frame synchronously in thread pool.

    Args:
    ----
        frame_bytes: JPEG encoded frame bytes.

    Returns:
    -------
        processed_frame_bytes: JPEG encoded processed frame.
        tts_alert: Optional TTS alert message.

    """
    try:
        # Decode JPEG frame
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            LOGGER.warning("Failed to decode frame")
            # Return original bytes so caller doesn't crash
            return frame_bytes, None

        # Process frame with vision processor
        annotated_frame, tts_alert = vision_processor.process_frame(frame)

        # Encode processed frame as JPEG
        success, buffer = cv2.imencode(".jpg", annotated_frame)
        if not success:
            LOGGER.warning("Failed to encode processed frame")
            # Return original bytes so caller doesn't crash
            return frame_bytes, tts_alert

        processed_frame_bytes = buffer.tobytes()

        return processed_frame_bytes, tts_alert

    except Exception as e:
        LOGGER.error(f"Error in process_frame_sync: {e}")
        # Return original frame bytes on error to maintain consistent return type
        return frame_bytes, None


@app.websocket("/video/in")
async def video_in(websocket: WebSocket):
    """WebSocket endpoint for ingesting video frames.

    Accepts binary JPEG frames, processes them, and broadcasts results.
    """
    await websocket.accept()
    LOGGER.info("Client connected to /video/in")

    try:
        while True:
            # Receive binary frame data
            try:
                frame_data = await websocket.receive_bytes()
            except RuntimeError as e:
                LOGGER.error(f"Error receiving data: {e}")
                break

            if not frame_data:
                LOGGER.warning("Received empty frame data")
                continue

            # Process frame in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            try:
                processed_frame_bytes, tts_alert = await loop.run_in_executor(
                    executor, process_frame_sync, frame_data
                )
            except Exception as e:
                LOGGER.error(f"Error processing frame: {e}")
                continue

            # Broadcast processed frame to /video/out clients
            if processed_frame_bytes:
                disconnected_clients = set()
                for client in video_out_clients:
                    try:
                        await client.send_bytes(processed_frame_bytes)
                    except Exception as e:
                        LOGGER.error(f"Error sending to video/out client: {e}")
                        disconnected_clients.add(client)

                # Remove disconnected clients
                video_out_clients.difference_update(disconnected_clients)

            # Broadcast TTS alert to /tts clients
            if tts_alert:
                disconnected_clients = set()
                for client in tts_clients:
                    try:
                        await client.send_text(tts_alert)
                    except Exception as e:
                        LOGGER.error(f"Error sending to tts client: {e}")
                        disconnected_clients.add(client)

                # Remove disconnected clients
                tts_clients.difference_update(disconnected_clients)

    except WebSocketDisconnect:
        LOGGER.info("Client disconnected from /video/in")
    except Exception as e:
        LOGGER.error(f"Error in /video/in handler: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/video/out")
async def video_out(websocket: WebSocket):
    """WebSocket endpoint for receiving processed video frames.

    Clients connect here to receive processed frames.
    """
    await websocket.accept()
    video_out_clients.add(websocket)
    LOGGER.info(f"Client connected to /video/out (total: {len(video_out_clients)})")

    try:
        # Keep connection alive
        while True:
            # Wait for client to disconnect or send data
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as e:
                LOGGER.error(f"Error in /video/out: {e}")
                break

    except WebSocketDisconnect:
        LOGGER.info("Client disconnected from /video/out")
    except Exception as e:
        LOGGER.error(f"Error in /video/out handler: {e}")
    finally:
        video_out_clients.discard(websocket)
        LOGGER.info(f"Client removed from /video/out (remaining: {len(video_out_clients)})")
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/tts")
async def tts(websocket: WebSocket):
    """WebSocket endpoint for receiving TTS alerts.

    Clients connect here to receive text-to-speech alerts.
    """
    await websocket.accept()
    tts_clients.add(websocket)
    LOGGER.info(f"Client connected to /tts (total: {len(tts_clients)})")

    try:
        # Keep connection alive
        while True:
            # Wait for client to disconnect or send data
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as e:
                LOGGER.error(f"Error in /tts: {e}")
                break

    except WebSocketDisconnect:
        LOGGER.info("Client disconnected from /tts")
    except Exception as e:
        LOGGER.error(f"Error in /tts handler: {e}")
    finally:
        tts_clients.discard(websocket)
        LOGGER.info(f"Client removed from /tts (remaining: {len(tts_clients)})")
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/")
async def root():
    """Root endpoint returning server status."""
    return {
        "status": "running",
        "endpoints": {
            "video_in": "/video/in",
            "video_out": "/video/out",
            "tts": "/tts",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)