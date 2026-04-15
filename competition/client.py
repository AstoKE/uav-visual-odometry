"""
client.py — Yarışma sunucusu HTTP istemcisi

Sunucu API'si (şablon — gerçek endpoint'ler yarışma dokümanından alınacak):
  GET  /session/start       → session_id, camera_params
  GET  /frame/next          → frame_id, image_bytes, pos_x, pos_y, pos_z, health
  POST /frame/result        → {"frame_id": ..., "x": ..., "y": ..., "z": ...}
  GET  /session/end         → final score

Kullanım:
    from competition.client import CompetitionClient
    client = CompetitionClient(base_url="http://...", token="...")
    frame_id, img, pos, health = client.get_next_frame()
    client.submit_result(frame_id, x=1.2, y=0.5, z=0.0)

Gerçek API endpoint'leri ve auth şeması belli olduktan sonra
aşağıdaki _API_* sabitlerini güncelle.
"""

import requests
import numpy as np
import cv2
import json
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ── Sunucu ayarları (yarışma dökümanından güncellenecek) ─────────────────────
_DEFAULT_BASE_URL = "http://localhost:8080"   # ← gerçek URL buraya
_TIMEOUT = 10  # saniye

@dataclass
class FrameData:
    frame_id: str
    image: np.ndarray        # BGR, (H, W, 3)
    ref_x: float             # metre — health=0'da güvensiz
    ref_y: float
    ref_z: float
    health: int              # 1 = güvenilir, 0 = güvensiz


@dataclass
class CameraParams:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class CompetitionClient:
    """
    Yarışma sunucusu ile HTTP üzerinden iletişim kurar.

    Parameters
    ----------
    base_url : str
        Sunucu base URL (örn. "http://192.168.1.100:8080")
    token : str | None
        Bearer token veya API key (varsa)
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.session  = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.session.headers["Content-Type"] = "application/json"
        self._session_id: str | None = None
        self._camera_params: CameraParams | None = None

    # ── Oturum yönetimi ───────────────────────────────────────────────────────

    def start_session(self) -> CameraParams:
        """
        Yeni bir yarışma oturumu başlat, kamera parametrelerini al.
        Returns: CameraParams
        """
        resp = self._get("/session/start")
        data = resp.json()
        self._session_id = data.get("session_id", "")
        cam = data.get("camera", {})
        self._camera_params = CameraParams(
            fx=float(cam.get("fx", 457)),
            fy=float(cam.get("fy", 457)),
            cx=float(cam.get("cx", 320)),
            cy=float(cam.get("cy", 240)),
            width=int(cam.get("width",  640)),
            height=int(cam.get("height", 480)),
        )
        log.info(f"Oturum başladı: {self._session_id}  cam={self._camera_params}")
        return self._camera_params

    def end_session(self) -> dict:
        """Oturumu bitir, sonuçları al."""
        resp = self._get("/session/end")
        return resp.json()

    # ── Frame akışı ───────────────────────────────────────────────────────────

    def get_next_frame(self) -> FrameData | None:
        """
        Sunucudan bir sonraki frame'i al.
        Oturum bittiyse None döner.
        """
        try:
            resp = self._get("/frame/next")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.info("Frame kalmadı — oturum bitti.")
                return None
            raise

        data = resp.json()

        if data.get("done", False):
            return None

        # Görüntüyü decode et
        img_bytes = bytes.fromhex(data["image_hex"]) if "image_hex" in data \
            else self._fetch_image(data.get("image_url", ""))

        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return FrameData(
            frame_id = str(data["frame_id"]),
            image    = image,
            ref_x    = float(data.get("pos_x", 0.0)),
            ref_y    = float(data.get("pos_y", 0.0)),
            ref_z    = float(data.get("pos_z", 0.0)),
            health   = int(data.get("health", 1)),
        )

    def submit_result(self, frame_id: str, x: float, y: float, z: float) -> bool:
        """
        Pozisyon tahminini sunucuya gönder.
        Returns: True başarılıysa
        """
        payload = {
            "frame_id": frame_id,
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "z": round(float(z), 4),
        }
        if self._session_id:
            payload["session_id"] = self._session_id

        resp = self._post("/frame/result", payload)
        ok = resp.status_code == 200
        if not ok:
            log.warning(f"submit_result başarısız: {resp.status_code} {resp.text}")
        return ok

    # ── Yardımcılar ───────────────────────────────────────────────────────────

    def _get(self, path: str) -> requests.Response:
        url = self.base_url + path
        resp = self.session.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp

    def _post(self, path: str, payload: dict) -> requests.Response:
        url = self.base_url + path
        resp = self.session.post(url, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp

    def _fetch_image(self, url: str) -> bytes:
        resp = self.session.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.content

    @property
    def camera_params(self) -> CameraParams | None:
        return self._camera_params
