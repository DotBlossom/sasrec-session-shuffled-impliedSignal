# dependencies.py
from typing import Optional

import torch
from database import SessionLocal
#from inference import RecommendationService
from item_tower import HybridItemTower, OptimizedItemTower
from utils.vocab import get_std_vocab_size, get_std_field_keys

# 1. 모델 인스턴스를 저장할 전역 변수
global_encoder: Optional[HybridItemTower] = None
global_projector: Optional[OptimizedItemTower] = None
#global_gnn_model = Optional[SimGCL] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


global_batch_size: Optional[int] = None
#rec_service: RecommendationService = None

'''
def initialize_rec_service():
    global rec_service
    
    # DB 세션 열기: 모델 로딩에 필요한 아이템 벡터, ID 맵 등을 DB에서 가져오기 위해 필요
    db = SessionLocal()
    try:
        # RecommendationService 초기화
        # model_path는 'models/user_tower_latest.pth'와 같이 상대 경로를 권장합니다.
        model_path = "models/user_tower_symmetric_final.pth" 
        rec_service = RecommendationService(db_session=db, model_path=model_path)
    except Exception as e:
        print(f"❌ Recommendation Service 초기화 중 오류 발생: {e}")
        # 오류 발생 시 앱 시작을 중단하거나, rec_service를 None으로 유지하여 503 오류를 반환하도록 처리
        rec_service = None
    finally:
        # 모델 로딩 후 DB 세션을 즉시 닫아줍니다.
        db.close()
'''     


# 2. 모델 로딩 함수 (main.py의 startup 이벤트에서 호출됨)
def initialize_global_models():
    """
    모델 인스턴스를 로드하고 전역 변수에 저장합니다.
    FastAPI의 startup 이벤트 핸들러에서 호출
    """
    global global_encoder
    global global_projector
    global global_gnn_model
    global std_size 
    global num_std
    global global_batch_size
    std_size = get_std_vocab_size()
    num_std = len(get_std_field_keys())
    

    
    print("🚀 앱 시작: CoarseToFineItemTower 로딩 중...")
    global_encoder = HybridItemTower(std_size, num_std, embed_dim=128)
    print("✅ ItemTower 로드 완료.")

    print("🚀 앱 시작: OptimizedItemTower 로딩 중...")
    global_projector = OptimizedItemTower(input_dim=128, output_dim=128)
    print("✅ OptimizedItemTower 로드 완료.")
    
    
    print("🚀 앱 시작: Gnn 로딩 중...")
 #   global_gnn_model = SimGCL(in_feats=128, hidden_feats=64, out_feats=128, num_layers=2, dropout=0.3, alpha=0.2)
    print("✅ Gnn Model params 로드 완료.")

    global_batch_size = 192
    print(f"✅ Global Batch Size set to: {global_batch_size}")
    

# 3. 의존성 주입(DI) 제공자 함수
def get_global_encoder() -> HybridItemTower:
    """저장된 CoarseToFineItemTower 인스턴스를 반환하는 의존성 주입 함수."""
    if global_encoder is None:
        # 이 예외는 startup 이벤트가 실행되지 않았을 때만 발생
        raise Exception("Encoder model has not been loaded yet. Check application startup events.")
    return global_encoder

def get_global_projector() -> OptimizedItemTower:
    """저장된 OptimizedItemTower 인스턴스를 반환하는 의존성 주입 함수."""
    if global_projector is None:
        raise Exception("Projector model has not been loaded yet. Check application startup events.")
    return global_projector


def get_global_batch_size() -> int:
    
    if global_batch_size is None:
        raise Exception("global batch size has not been defined")
    return global_batch_size
