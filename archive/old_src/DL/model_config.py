"""
모델별 설정
4개의 모델에 대한 타겟 및 특성 설정
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ModelSpec:
    """모델 사양"""
    name: str
    mode: str  # flow, modela, modelb, modelc
    target_cols: List[str]
    description: str
    
    def __post_init__(self):
        """검증"""
        if not self.target_cols:
            raise ValueError(f"{self.name}: target_cols가 비어있습니다")


# 4개 모델 정의
MODEL_SPECS = {
    "flow": ModelSpec(
        name="FLOW",
        mode="flow",
        target_cols=["Q_in"],
        description="유입 유량(Q_in) 예측 - TMS 데이터 사용 안 함"
    ),
    
    "modelA": ModelSpec(
        name="ModelA",
        mode="modela",
        target_cols=["TOC_VU", "SS_VU"],
        description="TOC와 SS 예측 - FLUX를 입력으로 사용"
    ),
    
    "modelB": ModelSpec(
        name="ModelB",
        mode="modelb",
        target_cols=["TN_VU", "TP_VU"],
        description="TN과 TP 예측 - FLUX를 입력으로 사용"
    ),
    
    "modelC": ModelSpec(
        name="ModelC",
        mode="modelc",
        target_cols=["FLUX_VU", "PH_VU"],
        description="FLUX와 PH 예측 - FLUX 제외 (예측 대상)"
    )
}


def get_model_spec(model_name: str) -> ModelSpec:
    """
    모델 사양 가져오기
    
    Parameters:
    -----------
    model_name : str
        모델 이름 (flow, modelA, modelB, modelC)
        
    Returns:
    --------
    ModelSpec : 모델 사양
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_SPECS:
        available = ", ".join(MODEL_SPECS.keys())
        raise ValueError(f"알 수 없는 모델: {model_name}. 사용 가능: {available}")
    
    return MODEL_SPECS[model_name_lower]


def get_available_models() -> List[str]:
    """사용 가능한 모델 목록 반환"""
    return list(MODEL_SPECS.keys())


def print_model_info():
    """모델 정보 출력"""
    print("\n" + "="*60)
    print("사용 가능한 모델")
    print("="*60)
    
    for model_name, spec in MODEL_SPECS.items():
        print(f"\n{spec.name} ({model_name}):")
        print(f"  타겟: {', '.join(spec.target_cols)}")
        print(f"  설명: {spec.description}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print_model_info()
