"""
MeshAnythingV2 FastAPI Service

Single-file API following ai-services-unified conventions.
Converts 3D meshes / point clouds into artist-quality low-poly meshes.
"""

import os
import shutil
import uuid
import gc
import asyncio
import datetime
import logging
import logging.handlers

import torch
import trimesh
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from typing import Optional, Literal
from pydantic import BaseModel, Field

# =============================================================================
# Logging
# =============================================================================

os.makedirs("/app/logs", exist_ok=True)


class TaiwanFormatter(logging.Formatter):
    _TZ = datetime.timezone(datetime.timedelta(hours=8))

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=self._TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{record.msecs:03.0f}"


class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return "GET /health" not in record.getMessage()


def _rotating_file_handler(filename: str, formatter: logging.Formatter) -> logging.Handler:
    handler = logging.handlers.TimedRotatingFileHandler(
        f"/app/logs/{filename}",
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


_fmt = TaiwanFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_access_fmt = TaiwanFormatter("%(asctime)s %(message)s")

logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.addHandler(_rotating_file_handler("app.log", _fmt))
logger.addHandler(logging.StreamHandler())

_uvicorn_access = logging.getLogger("uvicorn.access")
_uvicorn_access.addFilter(HealthCheckFilter())
_uvicorn_access.addHandler(_rotating_file_handler("access.log", _access_fmt))

_uvicorn = logging.getLogger("uvicorn")
_uvicorn.addHandler(_rotating_file_handler("uvicorn.log", _fmt))

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="MeshAnythingV2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory: {OUTPUT_DIR}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_MESH_EXTENSIONS = {".obj", ".ply", ".stl", ".glb", ".gltf", ".off"}
SUPPORTED_PC_EXTENSIONS = {".npy"}

# -- Global state --------------------------------------------------------------
model = None
gpu_lock = asyncio.Lock()


# =============================================================================
# GPU Memory Tracking
# =============================================================================

def log_gpu_memory(label: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(
            f"GPU memory [{label}]: allocated={allocated:.2f} GB  reserved={reserved:.2f} GB"
        )


def flush_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("after flush")


# =============================================================================
# Error Classification
# =============================================================================

GPU_ERROR_RESPONSES = {
    503: {
        "description": "GPU OOM or model loading failed (retry after 30s)",
        "content": {
            "application/json": {
                "examples": {
                    "GPU_OOM": {
                        "value": {"detail": {"error_code": "GPU_OOM", "message": "GPU out of memory"}}
                    }
                }
            }
        },
    },
    507: {"description": "Disk full"},
    500: {"description": "Inference error"},
}


def classify_exception(e: Exception) -> tuple:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return 503, "GPU_OOM", "GPU out of memory. Retry later."
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return 503, "GPU_OOM", "GPU out of memory. Retry later."
    if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
        return 507, "DISK_FULL", "Disk full. Contact administrator."
    return 500, "INFERENCE_ERROR", str(e)


# =============================================================================
# Model Loading (Global Lazy Load)
# =============================================================================

def ensure_model_loaded():
    global model
    if model is None:
        logger.info("[Lazy Load] Loading MeshAnythingV2 model...")
        log_gpu_memory("before model load")
        try:
            from MeshAnything.models.meshanything_v2 import MeshAnythingV2
            loaded_model = MeshAnythingV2.from_pretrained("Yiwen-ntu/meshanythingv2")
            loaded_model = loaded_model.to(DEVICE)
            loaded_model.eval()
            model = loaded_model
            log_gpu_memory("model loaded")
            logger.info("[Lazy Load] Model loaded successfully!")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Model loading failed: {e}")


# =============================================================================
# Preprocessing Helpers
# =============================================================================

def load_and_process_mesh(mesh_path: str, mc: bool = False, mc_level: int = 7) -> np.ndarray:
    """Load a mesh file and convert to point cloud with normals (8192 x 6)."""
    from mesh_to_pc import process_mesh_to_pc

    mesh = trimesh.load(mesh_path)
    if mc:
        logger.info(f"Running Marching Cubes preprocessing (level={mc_level})...")
    pc_list, _ = process_mesh_to_pc([mesh], marching_cubes=mc, mc_level=mc_level)
    return pc_list[0]


def load_point_cloud(npy_path: str) -> np.ndarray:
    """Load a .npy point cloud file (N x 6: xyz + normals), sample 8192 points."""
    cur_data = np.load(npy_path)
    if cur_data.shape[0] < 8192:
        raise ValueError(f"Point cloud has {cur_data.shape[0]} points, need at least 8192")
    idx = np.random.choice(cur_data.shape[0], 8192, replace=False)
    return cur_data[idx]


def normalize_pc(pc_normal: np.ndarray) -> np.ndarray:
    """Normalize point cloud coordinates to [-0.9995, 0.9995] range."""
    pc_coor = pc_normal[:, :3]
    normals = pc_normal[:, 3:]
    bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
    pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
    pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
    return np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)


def postprocess_mesh(output_tensor: torch.Tensor) -> trimesh.Trimesh:
    """Convert model output tensor to a clean trimesh object."""
    recon_mesh = output_tensor
    valid_mask = torch.all(~torch.isnan(recon_mesh.reshape((-1, 9))), dim=1)
    recon_mesh = recon_mesh[valid_mask]

    vertices = recon_mesh.reshape(-1, 3).cpu().numpy()
    vertices_index = np.arange(len(vertices))
    triangles = vertices_index.reshape(-1, 3)

    scene_mesh = trimesh.Trimesh(
        vertices=vertices, faces=triangles,
        force="mesh", merge_primitives=True
    )
    scene_mesh.merge_vertices()
    scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
    scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.remove_unreferenced_vertices()
    scene_mesh.fix_normals()
    return scene_mesh


# =============================================================================
# Response Schemas
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_busy: bool
    device: str = Field(description="'cuda' or 'cpu'")


class GenerateResponse(BaseModel):
    status: str
    request_id: str
    output_url: str
    num_faces: int


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "gpu_busy": gpu_lock.locked(),
        "device": DEVICE,
    }


@app.post("/generate", response_model=GenerateResponse, responses=GPU_ERROR_RESPONSES)
async def generate(
    file: UploadFile = File(...),
    input_type: Literal["mesh", "pc_normal"] = Form("mesh"),
    mc: bool = Form(False, description="Apply Marching Cubes preprocessing (mesh input only)"),
    mc_level: int = Form(7, ge=1, le=10, description="Marching Cubes octree depth"),
    sampling: bool = Form(False, description="Enable stochastic sampling"),
    seed: int = Form(0, ge=0, description="Random seed (used when sampling=true)"),
):
    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if input_type == "mesh" and ext not in SUPPORTED_MESH_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported mesh format '{ext}'. Must be one of: {sorted(SUPPORTED_MESH_EXTENSIONS)}",
        )
    if input_type == "pc_normal" and ext not in SUPPORTED_PC_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported point cloud format '{ext}'. Must be .npy",
        )

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    # Save uploaded file
    input_filename = file.filename or f"input{ext}"
    input_path = os.path.join(req_dir, input_filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"[{request_id}] Received {input_type} file: {input_filename}")

    async with gpu_lock:
        await run_in_threadpool(ensure_model_loaded)

        def run_inference():
            try:
                # Set seed
                if sampling:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                # Load and preprocess input
                logger.info(f"[{request_id}] Preprocessing...")
                if input_type == "mesh":
                    pc_normal = load_and_process_mesh(input_path, mc=mc, mc_level=mc_level)
                else:
                    pc_normal = load_point_cloud(input_path)

                pc_normal = normalize_pc(pc_normal)
                pc_tensor = torch.from_numpy(pc_normal).unsqueeze(0).to(DEVICE)

                # Run inference
                logger.info(f"[{request_id}] Running inference (sampling={sampling})...")
                log_gpu_memory("before inference")
                with torch.no_grad():
                    outputs = model(pc_tensor, sampling=sampling)

                log_gpu_memory("after inference")

                # Post-process output mesh
                scene_mesh = postprocess_mesh(outputs[0])
                num_faces = len(scene_mesh.faces)

                # Add face colors (orange, matching original)
                brown_color = np.array([255, 165, 0, 255], dtype=np.uint8)
                scene_mesh.visual.face_colors = np.tile(brown_color, (num_faces, 1))

                # Export
                output_path = os.path.join(req_dir, "output.obj")
                scene_mesh.export(output_path)
                logger.info(f"[{request_id}] Done! {num_faces} faces")

                return num_faces

            finally:
                flush_gpu()

        try:
            num_faces = await run_in_threadpool(run_inference)
            return {
                "status": "ok",
                "request_id": request_id,
                "output_url": f"/download/{request_id}/output.obj",
                "num_faces": num_faces,
            }
        except Exception as e:
            logger.error(f"[{request_id}] Error: {e}")
            status_code, error_code, message = classify_exception(e)
            raise HTTPException(
                status_code=status_code,
                detail={"error_code": error_code, "message": message},
            )


@app.get("/download/{request_id}/{file_name}")
async def download_file(request_id: str, file_name: str):
    file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    media_type = "application/octet-stream"
    if file_name.endswith(".obj"):
        media_type = "model/obj"
    elif file_name.endswith(".ply"):
        media_type = "application/x-ply"

    return FileResponse(file_path, media_type=media_type, filename=file_name)


@app.on_event("shutdown")
async def cleanup():
    logger.info("Server shutting down (output files preserved in %s)", OUTPUT_DIR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8192)
