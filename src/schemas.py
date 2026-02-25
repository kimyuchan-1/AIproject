from pydantic import BaseModel, Field


class FlowInput(BaseModel):
    SYS_TIME: str
    flow_TankA: float
    flow_TankB: float
    level_TankA: float
    level_TankB: float
    Q_in: float


class TMSInput(BaseModel):
    SYS_TIME: str
    TOC_VU: float
    PH_VU: float
    SS_VU: float
    FLUX_VU: float
    TN_VU: float
    TP_VU: float


class AWSInput(BaseModel):
    SYS_TIME: str
    TA: float
    RN_15m: float
    RN_60m: float
    RN_12H: float
    RN_DAY: float
    HM: float
    TD: float
    distance: float


class FlowPredictInput(BaseModel):
    dataList: list[FlowInput]
    awsList: dict[str, list[AWSInput]]


class TMSPredictInput(BaseModel):
    dataList: list[TMSInput]
    awsList: dict[str, list[AWSInput]]


class FlowPredictIn(BaseModel):
    model_config = {"populate_by_name": True}
    request_id: str | None = None
    input: FlowPredictInput = Field(alias="in")


class TMSPredictIn(BaseModel):
    model_config = {"populate_by_name": True}
    request_id: str | None = None
    input: TMSPredictInput = Field(alias="in")


class PredictOut(BaseModel):
    request_id: str
    ok: bool
    output: dict | None = None
    latency_ms: int
    error: dict | None = None
