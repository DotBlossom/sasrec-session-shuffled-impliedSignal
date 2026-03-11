            
import os
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
import pickle
import sys


from tower_code.params_config import PipelineConfig


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from preprocessor.preprocessor_v2 import FeatureProcessor_v3, dataset_peek_v3
from  tower_code.v3_model_usertower import SASRecDataset_v3,SASRecUserTower_v3


def setup_environment(seed: int = 42):
    """난수 고정 및 디바이스 설정 (Airflow Task 독립성 보장)"""
    print("\n⚙️ [Phase 1] Setting up environment...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device set to: {device}")
    return device


# =====================================================================
# Phase 2: Data Preparation
# =====================================================================
def prepare_features(cfg: PipelineConfig):
    """FeatureProcessor 초기화 및 메타데이터 업데이트 (로컬 캐싱 적용)"""
    print("\n📊 [Phase 2] Loading Processors...")
    
    # 1. 캐시 파일 경로 설정
    cache_path = os.path.join(cfg.base_dir, "processor_cache.pkl")
    
    # 2. 캐시 존재 여부 확인 및 로드
    if os.path.exists(cache_path):
        print(f"   ✅ [Cache Hit] Found cached processors at {cache_path}")
        print("   ⏳ Loading from local storage...")
        with open(cache_path, 'rb') as f:
            train_proc, val_proc = pickle.load(f)
            
    # 3. 캐시가 없을 경우: 원본 파라켓 로드 및 생성
    else:
        print("   ⚠️ [Cache Miss] Cache not found. Processing from Parquet files...")
        
        # 경로 설정
        user_path = os.path.join(cfg.base_dir, "features_user_w_meta_nonleak.parquet") 
        item_path = os.path.join(cfg.base_dir, "features_item.parquet")
        #seq_path = os.path.join(cfg.base_dir, "features_sequence_cleaned.parquet")
        
        TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
        USER_VAL_FEAT_PATH = os.path.join(cfg.base_dir, "features_user_w_meta_nonleak_val.parquet")
        #SEQ_VAL_DATA_PATH = os.path.join(cfg.base_dir, "features_sequence_val.parquet")
        
        # Processor 초기화 
        train_proc = FeatureProcessor_v3(user_path, item_path)
        val_proc = FeatureProcessor_v3(USER_VAL_FEAT_PATH, item_path, base_processor=train_proc)
        
        # [신규] 생성된 Processor 객체를 로컬 파일로 저장 (HIGHEST_PROTOCOL로 속도/용량 최적화)
        print("   💾 Saving processors to local cache for future use...")
        with open(cache_path, 'wb') as f:
            pickle.dump((train_proc, val_proc), f, protocol=pickle.HIGHEST_PROTOCOL)

    # 4. Config 업데이트 (캐시에서 불러왔든 새로 만들었든 동일하게 적용)
    cfg.num_items = train_proc.num_items
    
    ####### 실제 item metadata id랑 묶인상태로 가져와야하고 연결 필요 #######
    cfg.num_prod_types = int(train_proc.i_side_arr[:, 0].max()) + 1
    cfg.num_colors = int(train_proc.i_side_arr[:, 1].max()) + 1
    cfg.num_graphics = int(train_proc.i_side_arr[:, 2].max()) + 1
    cfg.num_sections = int(train_proc.i_side_arr[:, 3].max()) + 1
    
    print(f"✅ Max hash needed: {cfg.num_prod_types+cfg.num_colors + cfg.num_graphics +  cfg.num_sections}")
    print(f"✅ Features Loaded. Total Items: {cfg.num_items}")
    
    return train_proc, val_proc, cfg
# =====================================================================
# Phase 3: Embedding Alignment & DataLoader
# =====================================================================
def load_aligned_pretrained_embeddings(processor, model_dir, pretrained_dim):
    """Dataset에서 사용할 수 있도록 정렬된 사전학습 벡터(N+1, Dim) 생성"""
    print(f"\n🔄 [Phase 3-1] Aligning Pretrained Item Embeddings...")
    emb_path = os.path.join(model_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(model_dir, "item_ids.pt")

    num_embeddings = processor.num_items + 1 
    aligned_weight = torch.randn(num_embeddings, pretrained_dim) * 0.01 
    aligned_weight[0] = 0.0 # Padding
    
    try:
        pretrained_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(pretrained_emb, dict):
            pretrained_emb = pretrained_emb.get('weight', pretrained_emb.get('item_content_emb.weight'))
        pretrained_ids = torch.load(ids_path, map_location='cpu')
        
        pretrained_map = {str(iid.item()) if isinstance(iid, torch.Tensor) else str(iid): pretrained_emb[idx] 
                          for idx, iid in enumerate(pretrained_ids)}
        
        matched = 0
        for i, current_id_str in enumerate(processor.item_ids):
            if current_id_str in pretrained_map:
                aligned_weight[i + 1] = pretrained_map[current_id_str]
                matched += 1
                
        print(f"✅ Matched: {matched}/{len(processor.item_ids)}")
    except Exception as e:
        print(f"⚠️ [Warning] Failed to load Pretrained files: {e}. Using random init.")
        
    return aligned_weight

def create_dataloaders(processor, cfg: PipelineConfig, global_now_str, aligned_pretrained_vecs=None, is_train=True):
    """Dataset 및 DataLoader 인스턴스화"""
    mode_str = "Train" if is_train else "Validation"
    print(f"\n📦 [Phase 3-2] Creating {mode_str} DataLoaders...")
    
    # 💡 1. is_train 파라미터 전달
    dataset = SASRecDataset_v3(processor, global_now_str = global_now_str, max_len=cfg.max_len, is_train=is_train)
    
    # Dataset 인스턴스에 정렬된 pretrained vector 룩업 테이블 주입
    dataset.pretrained_lookup = aligned_pretrained_vecs 
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        # 💡 2. 검증 시에는 셔플을 끄고, 자투리 데이터(마지막 배치)도 버리지 않고 모두 평가
        shuffle=is_train, 
        num_workers=0, 
        pin_memory=True,
        drop_last=is_train 
    )
    
    print(f"✅ {mode_str} Loader Ready: {len(loader)} batches/epoch")
    return loader

def load_item_tower_state_dict(model_dir: str, pth_filename: str, device):
    """
    [Data/IO] 물리적 파일(.pth)을 읽어 메모리(state_dict)로 올리는 순수 IO 역할.
    모델 구조나 학습 상태(Freeze 여부)에는 절대 관여하지 않음.
    """
    file_path = os.path.join(model_dir, pth_filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️ [IO Warning] Item Tower file not found: {file_path}")
        print("   -> Random initialization will be used.")
        return None
        
    print(f"📥 [IO] Loading Item Tower weights from {pth_filename}...")
    
    try:
        # map_location을 통해 CPU/GPU 메모리 매핑 최적화
        state_dict = torch.load(file_path, map_location=device)
        return state_dict
    except Exception as e:
        print(f"❌ [IO Error] Failed to load .pth file: {e}")
        return None
    
import hashlib
import json

def get_hash_id(text, hash_size):
    """문자열을 일관된 정수 ID(1 ~ hash_size)로 해싱 (0은 Padding)"""
    if not text or str(text).lower() in ['unknown', 'nan', 'none']:
        return 0
    # MD5를 사용하여 파이썬 세션이 바뀌어도 항상 동일한 해시값 보장
    hash_obj = hashlib.md5(str(text).strip().lower().encode('utf-8'))
    # 16진수를 정수로 변환 후 hash_size로 나눈 나머지 + 1
    return (int(hash_obj.hexdigest(), 16) % hash_size) + 1

def load_item_metadata_hashed(processor, base_dir, hash_size=1000):
    """JSON 파일을 읽어 정렬된 메타데이터 해시 텐서(N+1, 4)를 생성"""
    print("\n🏷️ [Phase 3-2] Loading and Hashing Item Metadata...")
    json_path = os.path.join(base_dir, "filtered_data_reinforced.json")
    
    num_items = processor.num_items + 1
    # 0번 인덱스는 패딩을 위해 0으로 유지 (N+1, 4차원 배열)
    item_side_arr = np.zeros((num_items, 4), dtype=np.int64)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            item_data = json.load(f)
    except Exception as e:
        print(f"❌ [Error] Failed to load JSON: {e}")
        return torch.tensor(item_side_arr, dtype=torch.long)
    
    # 빠른 검색을 위해 O(1) Lookup Dictionary 생성
    # int형 article_id를 string으로 변환하여 매핑
    metadata_dict = {str(item.get('article_id', '')): item for item in item_data}
    
    matched = 0
    for i, current_id_str in enumerate(processor.item_ids):
        idx = i + 1 # 1-based indexing
        
        if current_id_str in metadata_dict:
            meta = metadata_dict[current_id_str]
            
            # 카테고리 매핑 및 해싱 (해당 키가 없으면 빈 문자열 반환)
            type_val = meta.get("product_type_name", "")
            color_val = meta.get("colour_group_name", "")
            graphic_val = meta.get("graphical_appearance_name", "")
            section_val = meta.get("section_name", "")
            
            item_side_arr[idx, 0] = get_hash_id(type_val, hash_size)
            item_side_arr[idx, 1] = get_hash_id(color_val, hash_size)
            item_side_arr[idx, 2] = get_hash_id(graphic_val, hash_size)
            item_side_arr[idx, 3] = get_hash_id(section_val, hash_size)
            
            matched += 1

    print(f"✅ Metadata Matched & Hashed: {matched}/{len(processor.item_ids)} (Hash Size: {hash_size})")
    
    del item_data
    del metadata_dict
    import gc
    gc.collect()
    
    return torch.tensor(item_side_arr, dtype=torch.long)
# =====================================================================
# Phase 4: Model Setup
# =====================================================================
class SASRecItemTower(nn.Module):
    def __init__(self, num_items, d_model, log_q_tensor=None):
        super().__init__()
        
        # 💡 단순히 임베딩이라기보다 '미세조정 가능한 아이템 벡터 행렬'임을 명시
        self.item_matrix = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        
        if log_q_tensor is not None:
            self.register_buffer('log_q', log_q_tensor)
        else:
            self.register_buffer('log_q', torch.zeros(num_items + 1))

    def get_all_embeddings(self):
        return self.item_matrix.weight

    def get_log_q(self):
        return self.log_q
        
    def set_freeze_state(self, freeze: bool):
        for param in self.parameters():
            param.requires_grad = not freeze
        # SASRecItemTower 클래스 내부에 추가
    def get_embeddings(self, item_ids):
        """
        특정 아이템 ID들에 대해서만 임베딩을 추출 (VRAM 절약용)
        item_ids: [num_unique_ids] 형태의 텐서
        """
        # item_matrix 레이어를 직접 호출하여 인덱싱된 임베딩을 가져옵니다.
        return self.item_matrix(item_ids)
    # 💡 [핵심] 밖에서 억지로 쑤셔넣지 않고, 클래스 스스로 추론 벡터를 받아 초기화하는 메서드
    def init_from_pretrained(self, pretrained_vecs):
        """추론된 사전학습 벡터를 미세조정 가능한 파라미터(Weight)로 초기화"""
        with torch.no_grad():
            self.item_matrix.weight.copy_(pretrained_vecs)
        print("✅ Pretrained item vectors successfully loaded into learnable matrix!")
    
def setup_models(cfg: PipelineConfig, device, item_state_dict=None, log_q_tensor=None):
    print(f"\n🧠 [Phase 4] Initializing Models...")
    
    # 1. User Tower 생성
    user_tower = SASRecUserTower_v3(cfg).to(device)
    
    # 2. Item Tower 뼈대 생성
    item_tower = SASRecItemTower(
        num_items=cfg.num_items, 
        d_model=cfg.d_model, 
        log_q_tensor=log_q_tensor
    ).to(device)
    
    # 3. Data 주입 (IO 데이터 -> Architecture)
    if item_state_dict is not None:
        try:
            # strict=False 옵션: 저장된 모델과 현재 구조의 키 이름이 조금 달라도 유연하게 로드
            missing, unexpected = item_tower.load_state_dict(item_state_dict, strict=False)
            print(f"✅ Item Tower weights successfully loaded!")
            if unexpected:
                print(f"   - Ignored extra keys from .pth: {unexpected[:3]}...")
            if missing:
                print(f"   ⚠️ [CRITICAL WARNING] Missing keys: {missing}")
        except Exception as e:
            print(f"❌ [Error] Weight injection failed: {e}")

    # 4. 학습 상태(Freeze/Unfreeze) 통제 적용
    item_tower.set_freeze_state(cfg.freeze_item_tower)
    
    # 직관적인 로깅
    mode_str = "FROZEN ❄️ (Speed Optimized)" if cfg.freeze_item_tower else "UNFROZEN 🔥 (Joint Fine-tuning)"
    print(f"✅ Item Tower State: {mode_str}")
    
    return user_tower, item_tower



def create_category_mapping_tensor(json_path, processor, device):
    """
    JSON에서 product_type_name을 추출하여 아이템 모델 인덱스(1~N)에 매핑되는
    1D 카테고리 텐서를 생성합니다. (0번 인덱스는 패딩용으로 0값 유지)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 💡 수정 1: FeatureProcessor_v3는 1-based index(0은 패딩)를 사용하므로 +1 크기로 생성
    num_items_total = processor.num_items + 1
    category_tensor = torch.zeros(num_items_total, dtype=torch.long)
    
    # 카테고리 ID도 0을 패딩(Unknown)으로 취급하고 1부터 시작하도록 구성
    cat_str_to_id = {}
    
    for item in data:
        # 💡 수정 2: processor 내부에서 str 타입으로 인덱싱되어 있으므로 형변환 필수
        pid = str(item['article_id'])
        
        # 💡 수정 3: processor.item_id_map 대신 processor.item2id 사용
        if pid in processor.item2id:
            idx = processor.item2id[pid]
            cat_name = item['product_type_name']
            
            if cat_name not in cat_str_to_id:
                # 카테고리 ID를 1부터 순차적으로 부여
                cat_str_to_id[cat_name] = len(cat_str_to_id) + 1
                
            category_tensor[idx] = cat_str_to_id[cat_name]
            
    print(f"📦 Category mapping created. Unique categories: {len(cat_str_to_id)}")
    return category_tensor.to(device)
def mine_category_constrained_hard_negatives(item_embs, category_tensor, k=5, device='cuda'):
    """
    에포크 시작 시 호출. 동일 카테고리 내에서 Top-K 하드 네거티브를 추출합니다.
    """
    num_items = item_embs.size(0)
    # 초기화: 기본값은 자기 자신(또는 0)으로 채우되, 이후 로직에서 덮어씌움
    hard_neg_pool = torch.zeros((num_items, k), dtype=torch.long, device=device)
    
    unique_cats = torch.unique(category_tensor)
    
    for cat in unique_cats:
        # 1. 해당 카테고리에 속한 아이템 인덱스 추출
        cat_mask = (category_tensor == cat)
        cat_indices = torch.nonzero(cat_mask).squeeze(1) # [N_cat]
        
        # 카테고리 내 아이템이 너무 적으면 마이닝 생략 (자기 자신 복제)
        if len(cat_indices) <= 1:
            hard_neg_pool[cat_indices] = cat_indices.unsqueeze(1).expand(-1, k)
            continue
            
        # 2. 카테고리 내 아이템 임베딩 추출 및 유사도 계산
        cat_embs = item_embs[cat_indices] # [N_cat, dim]
        sim_matrix = torch.matmul(cat_embs, cat_embs.T) # [N_cat, N_cat]
        
        # 3. 마스킹: 자기 자신 및 너무 똑같은 복제품(유사도 0.99 이상) 제외
        sim_matrix.fill_diagonal_(-float('inf'))
        sim_matrix.masked_fill_(sim_matrix >= 0.99, -float('inf'))
        
        # 4. Top-K 추출
        actual_k = min(k, len(cat_indices) - 1)
        _, topk_idx_local = torch.topk(sim_matrix, actual_k, dim=1)
        
        # 5. 로컬 인덱스를 전체 글로벌 아이템 인덱스로 복원
        topk_idx_global = cat_indices[topk_idx_local]
        
        # 만약 카테고리 내 아이템 수가 K개보다 적다면 부족한 만큼 패딩(첫 번째 아이템 복제)
        if actual_k < k:
            pad_idx = topk_idx_global[:, [0]].expand(-1, k - actual_k)
            topk_idx_global = torch.cat([topk_idx_global, pad_idx], dim=1)
            
        # 최종 풀에 업데이트
        hard_neg_pool[cat_indices] = topk_idx_global
        
    return hard_neg_pool        

def mine_global_hard_negatives(item_embs, exclusion_top_k=50, mine_k=5, batch_size=2048, device='cuda'):
    """
    에포크 시작 시 호출. 카테고리 제약 없이 전체 아이템 풀에서 하드 네거티브를 추출합니다.
    진짜 정답일 확률이 높은 Top-K(exclusion_top_k)는 배제하고, 그 다음 순위에서 mine_k개를 뽑습니다.
    
    Args:
        item_embs (Tensor): 전체 아이템 임베딩 [num_items, dim]
        exclusion_top_k (int): 배제할 최상위 유사도 아이템 수 (False Negative 방어용 안전지대)
        mine_k (int): 최종적으로 추출할 하드 네거티브 개수
        batch_size (int): GPU OOM 방지를 위한 연산 청크 크기
    """
    
    num_items = item_embs.size(0)
    hard_neg_pool = torch.zeros((num_items, mine_k), dtype=torch.long, device=device)
    
    print(f"🔍 Mining Global Hard Negatives: Skipping Top {exclusion_top_k}, Mining Next {mine_k}...")

    # GPU OOM 방지를 위해 쿼리(Query) 아이템을 배치 단위로 분할하여 처리
    for i in range(0, num_items, batch_size):
        end_i = min(i + batch_size, num_items)
        batch_embs = item_embs[i:end_i] # [Batch, dim]
        
        # 1. 글로벌 유사도 계산 (현재 배치 아이템 vs 전체 아이템)
        sim_matrix = torch.matmul(batch_embs, item_embs.T) # [Batch, num_items]
        
        # 2. 마스킹: 너무 똑같은 복제품(유사도 0.99 이상) 제외
        sim_matrix.masked_fill_(sim_matrix >= 0.99, -float('inf'))
        
        # 마스킹: 자기 자신을 확실히 제외 (대각 원소 처리)
        batch_indices = torch.arange(i, end_i, device=device)
        sim_matrix[torch.arange(end_i - i, device=device), batch_indices] = -float('inf')
        
        # 3. Top-(exclusion_top_k + mine_k) 추출
        # 예: exclusion_top_k=50, mine_k=5 이면 상위 55개를 뽑음
        total_k = min(exclusion_top_k + mine_k, num_items - 1)
        _, topk_indices_global = torch.topk(sim_matrix, total_k, dim=1)
        
        # 4. 안전지대(False Negative) 건너뛰고 진짜 하드 네거티브만 슬라이싱
        # 인덱스 0~49 (Top 50)은 버리고, 50~54 (Next 5)만 가져옴
        hard_negs = topk_indices_global[:, exclusion_top_k : exclusion_top_k + mine_k]
        
        # 예외 처리: 전체 아이템 수가 너무 적어 mine_k개를 못 채운 경우 첫 번째 값으로 패딩
        if hard_negs.size(1) < mine_k:
            pad_size = mine_k - hard_negs.size(1)
            pad_idx = hard_negs[:, [0]].expand(-1, pad_size)
            hard_negs = torch.cat([hard_negs, pad_idx], dim=1)
            
        hard_neg_pool[i:end_i] = hard_negs
        
    return hard_neg_pool