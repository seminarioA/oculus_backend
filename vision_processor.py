import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import depth_pro
from ultralytics import YOLO
from matplotlib import cm

LOGGER = logging.getLogger(__name__)


class VisionProcessor:
    def __init__(self):
        # ---------------- Configuración de clases objetivo ----------------
        # Clave: nombre de clase YOLO; valor: etiqueta a usar en TTS
        self.TARGET_CLASSES: Dict[str, str] = {
            "person": "persona",
            "bicycle": "bicicleta",
            "motorcycle": "moto",
            "car": "auto",
            "bus": "bus",
            "truck": "camión",
            "dog": "perro",
            "traffic light": "semáforo",
            "stop sign": "señal de stop",
            "fire hydrant": "hidrante",
            "bench": "banco",
            "chair": "silla",
            "couch": "sofá",
            "dining table": "mesa",
            "potted plant": "planta",
            "refrigerator": "refrigerador",
        }

        # Umbrales de alerta (metros)
        self.URGENT_THRESH = 1.5
        self.WARN_THRESH = 3.0

        # ---------------- Dispositivos y modelos ----------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device, precision=self.precision
        )
        self.model.eval()

        # YOLO liviano; cambia a yolov8s/8m si necesitas más precisión
        self.yolo = YOLO("yolov8n.pt")
        self.class_to_id = {name: idx for idx, name in self.yolo.model.names.items()}
        self.target_ids = [
            self.class_to_id[name]
            for name in self.TARGET_CLASSES.keys()
            if name in self.class_to_id
        ]

        # Colormap opcional (para depurar profundidad en overlay, no usado aquí)
        self.turbo = cm.get_cmap("turbo")

    def _compute_focal_px(self, width_px: int) -> float:
        """
        Focal en píxeles aproximada si no hay EXIF:
        27mm equiv, sensor 3.58mm => f_px ≈ 0.626 * width_px
        """
        f_mm_real = 27 * (3.58 / 43.27)  # ≈ 2.24 mm
        sensor_width_mm = 3.58
        return f_mm_real / sensor_width_mm * width_px  # ≈ 0.626 * W

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        Args:
            frame_bgr: np.ndarray BGR (cv2)

        Returns:
            annotated_bgr: np.ndarray BGR con cajas y texto
            tts_alert: str con alertas concatenadas o None
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, None

        # Convertir a RGB para DepthPro/YOLO
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = frame_rgb.shape[:2]

        # Focal (sin EXIF)
        f_px = self._compute_focal_px(W)

        # DepthPro
        image_t = self.transform(frame_rgb)
        pred = self.model.infer(image_t, f_px=torch.tensor(f_px, device=self.device))
        depth_m = pred["depth"].cpu().numpy()  # (H, W) en metros

        # YOLO (en RGB)
        results = self.yolo(frame_rgb, conf=0.25, classes=self.target_ids, verbose=False)
        boxes = results[0].boxes

        # Más cercano por clase
        nearest_by_class: Dict[str, Tuple[int, int, int, int, float, float, str]] = {}
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            cls_id = int(b.cls[0].item())
            cls_name = self.yolo.model.names[cls_id]
            label_tts = self.TARGET_CLASSES.get(cls_name, cls_name)

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, W), min(y2, H)

            crop = depth_m[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            med_depth = float(np.median(crop))  # metros

            if cls_name not in nearest_by_class or med_depth < nearest_by_class[cls_name][4]:
                nearest_by_class[cls_name] = (x1, y1, x2, y2, med_depth, float(b.conf), label_tts)

        detections = list(nearest_by_class.values())

        # Filtrar por umbrales para TTS
        tts_lines: List[str] = []
        for (_x1, _y1, _x2, _y2, d, _conf, label_tts) in detections:
            if d < self.URGENT_THRESH:
                tts_lines.append(f"Cuidado {label_tts} a {d:.1f} metros")
            elif d < self.WARN_THRESH:
                tts_lines.append(f"Atención {label_tts} a {d:.1f} metros")
            else:
                continue

        # Anotar sobre el frame BGR original
        annotated = frame_bgr.copy()
        for (x1, y1, x2, y2, d, _conf, label_tts) in detections:
            color = (0, 0, 255) if d < self.URGENT_THRESH else ((0, 255, 255) if d < self.WARN_THRESH else (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{label_tts}: {d:.2f}m",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        tts_alert = "; ".join(tts_lines) if tts_lines else None
        return annotated, tts_alert