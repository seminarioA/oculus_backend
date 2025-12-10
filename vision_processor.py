import logging
from typing import Dict, List, Optional, Tuple
import time
import math

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from matplotlib import cm
import timm
import sys

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------
# Alturas reales (SOLO donde tiene sentido usar geometría)
# ------------------------------------------------
REAL_HEIGHTS_M: Dict[str, float] = {
    "person": 1.70,     # razonable
    "bicycle": 1.0,
    "motorcycle": 1.1,
    "car": 1.5
    # NO incluimos chair o laptop: inducen errores graves
}

# ------------------------------------------------
# Cargar MiDaS Small
# ------------------------------------------------
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
except Exception as e:
    logging.error(f"Error al cargar MiDaS Small: {e}")
    midas = None


class VisionProcessor:
    def __init__(self):

        # -------- Clases a detectar --------
        self.TARGET_CLASSES: Dict[str, str] = {
            "person": "persona",
            "bicycle": "bicicleta",
            "motorcycle": "moto",
            "car": "auto",
            "chair": "silla",
            "tv": "computadora",
            "laptop": "laptop"
        }

        self.URGENT_THRESH = 1.2
        self.WARN_THRESH   = 2.0

        self.device = torch.device("cpu")

        if midas is None:
            raise RuntimeError("MiDaS no disponible.")

        # -------- MI DAS --------
        LOGGER.info("Cargando MiDaS_small...")
        self.midas = midas.to(self.device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        # -------- YOLO --------
        LOGGER.info("Cargando YOLOv8n...")
        self.yolo = YOLO("yolov8n.pt")
        self.class_to_id = {name: idx for idx, name in self.yolo.model.names.items()}

        self.target_ids = [
            self.class_to_id[name]
            for name in self.TARGET_CLASSES.keys()
            if name in self.class_to_id
        ]

        self.turbo = cm.get_cmap("turbo")
        self.REAL_HEIGHTS_M = REAL_HEIGHTS_M

        # -------- Parámetros cámara Samsung A20 --------
        # FOV horizontal típico ~61 grados
        self.FOV_HORIZONTAL_DEG = 61.0
        self.focal_real_px = None

        # -------- Anti spam TTS --------
        self.last_tts_time = 0.0
        self.TTS_INTERVAL = 3.5

        # -------- Suavizado --------
        self.dist_smooth = {}
        self.ALPHA = 0.30

    # --------------------------------------------------------------------
    def _compute_focal_px(self, width_px: int) -> float:
        fov_rad = math.radians(self.FOV_HORIZONTAL_DEG)
        return (width_px / 2) / math.tan(fov_rad / 2)

    # --------------------------------------------------------------------
    @torch.no_grad()
    def _infer_midas(self, frame_rgb: np.ndarray) -> np.ndarray:
        img = self.transform(frame_rgb).to(self.device)
        prediction = self.midas(img).squeeze().cpu().numpy()

        H, W = frame_rgb.shape[:2]
        depth_rel = cv2.resize(prediction, (W, H), interpolation=cv2.INTER_CUBIC)

        # MiDaS da valores relativos → invertimos
        depth_m = 1.0 / (depth_rel + 1e-6)

        # Reescalado heurístico para cámara móvil:
        depth_m *= 0.85

        return depth_m

    # --------------------------------------------------------------------
    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = frame_rgb.shape[:2]

        if self.focal_real_px is None:
            self.focal_real_px = self._compute_focal_px(W)
            LOGGER.info(f"Focal estimada: {self.focal_real_px:.1f} px")

        # ---------- Profundidad ----------
        depth_m = self._infer_midas(frame_rgb)

        # ---------- YOLO ----------
        results = self.yolo(frame_rgb, conf=0.25, classes=self.target_ids, verbose=False)
        boxes = results[0].boxes

        nearest_by_class = {}

        # ---------- Procesamiento ----------
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            cls_id = int(b.cls[0].item())
            cls_name = self.yolo.model.names[cls_id]
            label_tts = self.TARGET_CLASSES.get(cls_name, cls_name)
            h_pixels = max(1, y2 - y1)

            # ==========================================================
            #    1) Distancia vía MiDaS → media dentro de la caja
            # ==========================================================
            crop = depth_m[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            d_midas = float(np.median(crop))

            # Ajuste para rangos cortos (importante):
            if d_midas < 0.8:
                d_midas *= 0.70
            elif d_midas < 1.2:
                d_midas *= 0.80

            # Límite razonable
            d_midas = max(0.20, min(d_midas, 8.0))

            # ==========================================================
            #    2) Distancia geométrica SOLO si clase tiene altura real
            # ==========================================================
            if cls_name in self.REAL_HEIGHTS_M:
                H_real = self.REAL_HEIGHTS_M[cls_name]
                Z_geom = (self.focal_real_px * H_real) / h_pixels

                # Corrección rangos cortos
                if Z_geom < 1.0:
                    Z_geom *= 0.60

                Z_geom = max(0.30, min(Z_geom, 10.0))

                # Mezcla: dar prioridad a MiDaS < 1.2m
                if d_midas < 1.2:
                    Z_final = 0.75 * d_midas + 0.25 * Z_geom
                else:
                    Z_final = 0.50 * d_midas + 0.50 * Z_geom
            else:
                # Para sillas y computadoras: solo MiDaS (más estable)
                Z_final = d_midas

            # ==========================================================
            #            Suavizado EMA por clase
            # ==========================================================
            if cls_name not in self.dist_smooth:
                self.dist_smooth[cls_name] = Z_final
            else:
                self.dist_smooth[cls_name] = (
                    self.ALPHA * Z_final +
                    (1 - self.ALPHA) * self.dist_smooth[cls_name]
                )

            Zs = self.dist_smooth[cls_name]

            if cls_name not in nearest_by_class or Zs < nearest_by_class[cls_name][4]:
                nearest_by_class[cls_name] = (x1, y1, x2, y2, Zs, label_tts)

        # ==========================================================
        #                         TTS
        # ==========================================================
        tts_alert = None
        now = time.time()

        if now - self.last_tts_time > self.TTS_INTERVAL:
            msgs = []
            for (_, _, _, _, d, label) in nearest_by_class.values():
                if d < self.URGENT_THRESH:
                    msgs.append(f"Cuidado {label} a {d:.1f} metros")
                elif d < self.WARN_THRESH:
                    msgs.append(f"Atención {label} a {d:.1f} metros")

            if msgs:
                tts_alert = "; ".join(msgs)
                self.last_tts_time = now

        # ==========================================================
        #                      ANOTACIÓN VISUAL
        # ==========================================================
        annotated = frame_bgr.copy()

        for (x1, y1, x2, y2, d, label) in nearest_by_class.values():
            color = (0, 255, 0)
            if d < self.URGENT_THRESH:
                color = (0, 0, 255)
            elif d < self.WARN_THRESH:
                color = (0, 255, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label}: {d:.2f}m",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated, tts_alert