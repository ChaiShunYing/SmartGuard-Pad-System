"""
SmartGuard Pad - Cloud face recognition API (FastAPI version)
When ESP32 detects a weight drop, it takes a photo and POSTs it to /recognize,
which runs InsightFace comparison, returns the result immediately, and logs
the event to Firestore + Storage in the background (so ESP32 doesn't wait
for the Firebase round trip).

This version adds per-stage timing logs so we can see exactly where the
~7 seconds is being spent (network read, decode, face detection, etc).
"""
import os
import json
import uuid
import datetime
import pickle
import traceback
import time
import numpy as np
import cv2
from fastapi import FastAPI, Request, BackgroundTasks
from insightface.app import FaceAnalysis
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.oauth2 import service_account
from google.cloud import firestore as gcf

# ============ Config ============
THRESHOLD = 0.5

# ============ Initialize Firebase Admin (loaded once at startup) ============
firebase_creds = json.loads(os.environ["FIREBASE_CREDS_JSON"])
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred, {
    "storageBucket": "smartguard-pad-system.firebasestorage.app"
})

google_creds = service_account.Credentials.from_service_account_info(firebase_creds)
db = gcf.Client(project=firebase_creds["project_id"], credentials=google_creds, database="smartguard-db")
bucket = storage.bucket()
print(f"[DEBUG] Firestore client initialized for project: {firebase_creds['project_id']}")

# ============ Initialize insightface (loaded once at startup) ============
app = FastAPI()
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320, 320))

with open("owner_profile_l.pkl", "rb") as f:
    owner_embeddings = pickle.load(f)
print(f"Loaded {len(owner_embeddings)} owner embeddings")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def check_is_owner(face_embedding):
    best_score = max(cosine_similarity(face_embedding, e) for e in owner_embeddings)
    return best_score >= THRESHOLD, best_score


def log_event(device_id: str, image_bytes: bytes, result: str, similarity: float, reason: str, event_id: str):
    """
    Upload the photo to Storage and write an event record to Firestore.
    Runs as a BackgroundTask AFTER the response is already sent to the
    ESP32, so it never adds latency to the /recognize response.
    """
    try:
        blob = bucket.blob(f"events/{device_id}/{event_id}.jpg")
        blob.upload_from_string(image_bytes, content_type="image/jpeg")
        blob.make_public()
        photo_url = blob.public_url
    except Exception as e:
        print("=== STORAGE UPLOAD FAILED - FULL DEBUG INFO ===")
        print(f"Exception type: {type(e)}")
        print(f"Exception repr: {repr(e)}")
        traceback.print_exc()
        print("=================================================")
        photo_url = None

    try:
        event_ref = (
            db.collection("devices").document(device_id)
              .collection("events").document(event_id)
        )
        event_ref.set({
            "timestamp": datetime.datetime.utcnow(),
            "result": result,
            "similarity": similarity,
            "reason": reason,
            "photo_url": photo_url,
        })
    except Exception as e:
        print("=== FIRESTORE WRITE FAILED - FULL DEBUG INFO ===")
        print(f"Exception type: {type(e)}")
        print(f"Exception repr: {repr(e)}")
        traceback.print_exc()
        print("=================================================")


@app.post("/recognize")
async def recognize(request: Request, background_tasks: BackgroundTasks):
    t0 = time.time()
    device_id = request.query_params.get("device_id", "unknown-device")
    timestamp_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    event_id = f"{timestamp_str}_{uuid.uuid4().hex[:6]}"

    # ESP32 sends raw JPEG bytes directly (not multipart/form-data),
    # so read the raw request body instead of using UploadFile
    contents = await request.body()
    t1 = time.time()
    print(f"[TIMING] Read body: {t1 - t0:.2f}s, size: {len(contents)} bytes")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    t2 = time.time()
    print(f"[TIMING] Decode image: {t2 - t1:.2f}s")

    if frame is None:
        print("Recognition result: not_owner, reason: invalid_image")
        print(f"[TIMING] TOTAL server-side time: {t2 - t0:.2f}s")
        background_tasks.add_task(log_event, device_id, contents, "not_owner", 0.0, "invalid_image", event_id)
        return {"result": "not_owner", "similarity": 0.0, "reason": "invalid_image", "event_id": event_id}

    faces = face_app.get(frame)
    t3 = time.time()
    print(f"[TIMING] Face detection + embedding: {t3 - t2:.2f}s")

    if len(faces) == 0:
        print("Recognition result: not_owner, reason: no_face_detected")
        print(f"[TIMING] TOTAL server-side time: {t3 - t0:.2f}s")
        background_tasks.add_task(log_event, device_id, contents, "not_owner", 0.0, "no_face_detected", event_id)
        return {"result": "not_owner", "similarity": 0.0, "reason": "no_face_detected", "event_id": event_id}

    is_owner, score = check_is_owner(faces[0].embedding)
    t4 = time.time()
    print(f"[TIMING] Compare with owner embeddings: {t4 - t3:.2f}s")

    result = "owner" if is_owner else "not_owner"
    print(f"Recognition result: {result}, similarity: {score:.4f}")
    print(f"[TIMING] TOTAL server-side time: {t4 - t0:.2f}s")

    background_tasks.add_task(log_event, device_id, contents, result, round(float(score), 4), "ok", event_id)
    return {"result": result, "similarity": round(float(score), 4), "reason": "ok", "event_id": event_id}


@app.get("/health")
async def health():
    return {"status": "ok"}
