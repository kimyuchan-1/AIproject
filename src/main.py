import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import TMS_TARGETS
from .schemas import FlowPredictIn, TMSPredictIn, PredictOut
from .loader import (
    tms_models,
    tms_y_scalers,
    tms_features,
    flow_model,
    flow_y_scaler,
    FLOW_N_FEATURES,
)
from .preprocess import (
    extract_input_lists,
    validate_input_lengths,
    build_hourly_predictions,
    merge_input_data,
    make_tms_tensor,
    preprocess_flow_input,
)
from .predict import autoregressive_predict, find_target_lag_idx

app = FastAPI(title="WWTP Prediction API", version="0.3.0")

# ====== CORS 설정 ======
origins = [
    "http://www.projectwwtp.kro.kr:8081",
    "http://localhost:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== API 엔드포인트 ======
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ready")
def ready():
    return {
        "ok": True,
        "model_version": "0.3.0",
        "models_loaded": {
            "flow": {
                "n_features": FLOW_N_FEATURES,
            },
            "tms": {
                target_name: {
                    "n_features": len(tms_features[target_name]),
                    "use_attention": TMS_TARGETS[target_name]["use_attention"],
                }
                for target_name in TMS_TARGETS
            },
        },
        "window_size": 48,
        "horizon_unit": "30min",
    }


@app.post("/debug/flow")
async def debug_flow(request: Request):
    raw = await request.body()
    headers = dict(request.headers)

    print("=== HEADERS ===")
    for k in ["content-type", "content-length", "transfer-encoding", "connection", "upgrade", "host"]:
        print(f"{k}: {headers.get(k)}")

    print(f"=== RAW BODY LENGTH: {len(raw)} bytes ===")
    return {
        "content_type": headers.get("content-type"),
        "content_length": headers.get("content-length"),
        "transfer_encoding": headers.get("transfer-encoding"),
        "connection": headers.get("connection"),
        "upgrade": headers.get("upgrade"),
        "body_length": len(raw),
        "preview": raw[:500].decode("utf-8", errors="replace"),
    }


@app.post("/predict/tms", response_model=PredictOut)
def predict_tms(x: TMSPredictIn):
    try:
        t0 = time.perf_counter()
        rid = x.request_id or str(uuid.uuid4())

        tms_list, aws368, aws541, aws569 = extract_input_lists(x)
        validate_input_lengths(tms_list, aws368, aws541, aws569)

        # 공통: 데이터 병합 + 30분 리샘플링 (1회)
        df_resampled = merge_input_data(tms_list, aws368, aws541, aws569)

        all_predictions = {}
        all_trajectories = {}

        for target_name, cfg in TMS_TARGETS.items():
            # 타겟별 전처리 → 텐서
            input_tensor = make_tms_tensor(df_resampled, target_name)

            mdl = tms_models[target_name]
            y_sc = tms_y_scalers[target_name]
            feature_names = tms_features[target_name]
            target_feat_idx = find_target_lag_idx(feature_names, cfg["target_col"])

            # 12시간 = 24 steps (30분 × 24) 한 번에 예측
            pred_12h = autoregressive_predict(input_tensor, 24, mdl, y_sc, target_feat_idx)

            # 시간별 예측값 추출 (매 2 steps = 1시간)
            hourly_preds = build_hourly_predictions(pred_12h)

            all_predictions[target_name] = hourly_preds
            all_trajectories[target_name] = {"12h": pred_12h}

        latency = int((time.perf_counter() - t0) * 1000)

        return PredictOut(
            request_id=rid,
            ok=True,
            output={
                "predictions": all_predictions,
                "trajectories": all_trajectories,
                "metadata": {
                    "window_size": 48,
                    "targets": list(TMS_TARGETS.keys()),
                    "input_records": len(tms_list),
                    "resampled_steps": df_resampled.shape[0],
                }
            },
            latency_ms=latency,
            error=None
        )
    except Exception as e:
        import traceback
        error_detail = {
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=400, detail=error_detail)


@app.post("/predict/flow", response_model=PredictOut)
def predict_flow(x: FlowPredictIn):
    try:
        t0 = time.perf_counter()
        rid = x.request_id or str(uuid.uuid4())

        flow_list, aws368, aws541, aws569 = extract_input_lists(x)
        validate_input_lengths(flow_list, aws368, aws541, aws569)

        input_tensor = preprocess_flow_input(flow_list, aws368, aws541, aws569)

        # Flow는 flow_TankA/B lag 사용 (Q_in lag 없음)
        # Q_in을 flow_TankA/B로 역분해할 수 없으므로 lag 업데이트 안 함
        flow_target_idx = None

        # 12시간 = 24 steps 한 번에 예측
        pred_12h = autoregressive_predict(input_tensor, 24, flow_model, flow_y_scaler, flow_target_idx)

        # 시간별 예측값 추출
        hourly_preds = build_hourly_predictions(pred_12h)

        latency = int((time.perf_counter() - t0) * 1000)

        return PredictOut(
            request_id=rid,
            ok=True,
            output={
                "predictions": hourly_preds,
                "trajectories": {
                    "12h": pred_12h,
                },
                "metadata": {
                    "window_size": 48,
                    "n_features": FLOW_N_FEATURES,
                    "input_records": len(flow_list),
                    "resampled_steps": input_tensor.shape[1],
                }
            },
            latency_ms=latency,
            error=None
        )
    except Exception as e:
        import traceback
        error_detail = {
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=400, detail=error_detail)
