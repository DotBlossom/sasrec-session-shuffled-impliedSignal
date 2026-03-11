import optuna
import wandb
import os
import torch
import torch.nn.functional as F
from optuna.trial import TrialState
from dataclasses import dataclass
import gc
    
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import pickle
import math
import os
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Garbage Collection
with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()
print(f"🧹 Memory Cleared. Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

@dataclass
class PipelineConfig:
# =====================================================================
    # Paths
    base_dir: str = "/kaggle/input/datasets/c24tw1f20/sasrec-usertower-session-shuffle"
    model_dir: str = "/kaggle/working"
    cache_dir: str = "/kaggle/working/cache"
    # Hyperparameters
    batch_size: int = 768
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    
    # Model Args (SASRecUserTower용)
    d_model: int = 128
    max_len: int = 50
    dropout: float = 0.3
    pretrained_dim: int = 128 # 사전학습 아이템 벡터 차원 
    nhead: int = 4
    num_layers: int = 2
    
    # Loss Penalties
    lambda_logq: float = 1.0
    lambda_sup: float = 0.1
    lambda_cl: float = 0.2
   
    # [신규] HNM 제어 파라미터
    top_k_percent: float = 0.01 # 상위 15% 하드 네거티브 사용 (10~20% 사이 권장)
    hnm_threshold: float = 0.90
    hard_margin: float = 0.01

    # model 관리
    freeze_item_tower: bool = True
    item_tower_pth_name: str = "encoder_ep03_loss0.8129.pth"
    # 자동 할당될 메타데이터 크기
    num_items: int = 0
    num_prod_types: int = 0
    num_colors: int = 0
    num_graphics: int = 0
    num_sections: int = 0
    num_age_groups: int = 10

    max_target_len: int = 10

    HN_K = 100  # 20개 추출 후 내부 0.95 제약으로 필터링
    EX_TOP_K = 25
    soft_penalty_weigh = 3
    hn_scheduled = False
    
    

if not os.path.exists(PipelineConfig.cache_dir): os.makedirs(PipelineConfig.cache_dir)

stage_status = [0,0,0,0,0,0,0]
stage_pointer = 0


def stage_container():
    global stage_status
    global stage_pointer
        
    stage_status[stage_pointer] = 1
    stage_pointer += 1

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


class FeatureProcessor_v3:
    def __init__(self, user_path, item_path, base_processor=None):
        print("🚀 Loading preprocessed features...")
        self.users = pd.read_parquet(user_path).drop_duplicates(subset=['customer_id']).set_index('customer_id')
        self.items = pd.read_parquet(item_path).drop_duplicates(subset=['article_id']).set_index('article_id')
        self.seqs = self.users[['sequence_ids', 'sequence_deltas']].copy() # 만약 deltas도 같이 묶으셨다면 포함

        # 인덱스 타입 강제 (String)
        self.users.index = self.users.index.astype(str)
        self.items.index = self.items.index.astype(str)
        self.seqs.index = self.seqs.index.astype(str)
        # 💡 [핵심 추가] 학습/평가에 의미가 없는 유저(데이터 1개 이하) 사전 필터링
        initial_count = len(self.seqs)
        self.seqs = self.seqs[self.seqs['sequence_ids'].apply(len) >= 2]
        filtered_count = len(self.seqs)
        
        print(f"✂️ [Filter] Removed {initial_count - filtered_count:,} users with less than 2 items.")
        # =================================================================
        # 1. ID Mappings (1-based, 0 is Padding)
        # =================================================================
        self.user_ids = self.seqs.index.tolist() # 시퀀스가 존재하는 유저만 대상
        self.user2id = {uid: i + 1 for i, uid in enumerate(self.users.index)}
        
        
        
        
        if base_processor is None:
            # Train일 때: 새롭게 아이템 번호표 생성
            self.item_ids = self.items.index.tolist()
            self.item2id = {iid: i + 1 for i, iid in enumerate(self.item_ids)}
            self.num_items = len(self.item_ids)
        else:
            # Validation일 때: Train의 번호표를 그대로 물려받음
            self.item_ids = base_processor.item_ids
            self.item2id = base_processor.item2id
            self.num_items = base_processor.num_items
        
        
        self.num_items = len(self.item_ids)
        
        
        # =================================================================
        # 2. Fast Lookup Arrays for Dataset (__getitem__ 속도 최적화)
        # =================================================================
        print("⚡ Building fast lookup tables...")
        num_users_total = len(self.users) + 1
        
        # 💡 [신규] 동적(Dynamic) 시퀀스 피처를 저장할 딕셔너리 
        # Numpy 배열로 변환하여 Dataset의 input_indices 슬라이싱에 완벽히 호환되도록 만듦
        self.u_dyn_buckets = {}  # Shape: (seq_len, 3)
        self.u_dyn_conts = {}    # Shape: (seq_len, 4)
        self.u_dyn_cats = {}     # Shape: (seq_len, 1) - preferred_channel
        
        # 💡 [변경] 유저 피처를 Static(1D)과 Dynamic(2D 시퀀스)으로 분리하여 저장
        self.u_static_buckets = np.zeros((num_users_total, 1), dtype=np.int64) # age
        self.u_static_cats = np.zeros((num_users_total, 4), dtype=np.int64) # club, news, fn, active
        
        # 동적 피처를 담을 딕셔너리 (유저별 시퀀스 길이가 다르므로 딕셔너리 + Numpy 배열 사용)
        self.u_dyn_buckets = {}
        self.u_dyn_conts = {}
        self.u_dyn_cats = {}
        self.u_dyn_time = {}
        self.u_seqs = {}
        self.u_deltas = {}
        for uid, row in self.users.iterrows():
            if uid not in self.user2id: continue
            uidx = self.user2id[uid]
            
            # 💡 [신규] 시퀀스 데이터를 딕셔너리에 저장
            self.u_seqs[uidx] = row['sequence_ids']
            self.u_deltas[uidx] = row['sequence_deltas']
            
            self.u_static_buckets[uidx] = [row['age_bucket']]
            self.u_static_cats[uidx] = [
                row['club_member_status_idx'], row['fashion_news_frequency_idx'], 
                row['FN'], row['Active']
            ]
            
            # 💡 [최적화] dtype을 int8, int32, float16으로 낮춰서 RAM 폭파 방지
            self.u_dyn_buckets[uidx] = np.column_stack([
                row['asof_avg_price_bucket'], row['asof_total_cnt_bucket'], row['asof_recency_bucket']
            ]).astype(np.int8)
            
            self.u_dyn_conts[uidx] = np.column_stack([
                row['asof_price_std_scaled'], row['asof_last_price_diff_scaled'],
                row['asof_repurchase_ratio_scaled'], row['asof_weekend_ratio_scaled']
            ]).astype(np.float16)
            
            self.u_dyn_cats[uidx] = np.array(row['asof_preferred_channel'], dtype=np.int8).reshape(-1, 1)
            
            self.u_dyn_time[uidx] = np.column_stack([
                row['asof_t_dat_ordinal'], row['asof_current_week']
            ]).astype(np.int32)
            
        self.i_side_arr = np.zeros((self.num_items + 1, 4), dtype=np.int16)
        for iid, row in self.items.iterrows():
            if iid not in self.item2id: continue
            idx = self.item2id[iid]
            self.i_side_arr[idx] = [
                row.get('type_id', 0), row.get('color_id', 0), 
                row.get('graphic_id', 0), row.get('section_id', 0)
            ]
        # 💡 [핵심 추가] 데이터프레임을 삭제하기 전에 확률 값만 미리 추출해서 저장합니다.
        print("📊 Extracting item probabilities for Log-Q correction...")
        # reindex를 통해 item2id 순서와 동일하게 정렬된 넘파이 배열을 만듭니다.
        self.item_raw_probs = self.items['raw_probability'].reindex(self.item_ids).values

        # -----------------------------------------------------------
        # 🧹 이제 안심하고 RAM을 잡아먹는 주범들을 삭제합니다.
        # -----------------------------------------------------------
        del self.users
        del self.items
        del self.seqs
        import gc
        gc.collect()
        print("   🧹 Memory cleaned: All heavy DataFrames deleted!")

    def get_logq_probs(self, device):
        """Negative Sampling이나 Loss 보정을 위한 아이템 등장 확률 Log 반환"""
        raw_probs = self.item_raw_probs 
        
        eps = 1e-6
        # 넘파이 배열이므로 그대로 연산 가능
        sorted_probs = np.nan_to_num(raw_probs, nan=0.0) + eps
        sorted_probs /= sorted_probs.sum()
        
        log_q_values = np.log(sorted_probs).astype(np.float32)
        
        full_log_q = np.zeros(self.num_items + 1, dtype=np.float32)
        full_log_q[1:] = log_q_values 
        full_log_q[0] = -20.0 # Padding Index
        stage_container()    
        return torch.tensor(full_log_q, dtype=torch.float32).to(device)

def monitor_asof_logic(df, sample_user_id):
    """특정 유저의 원본 거래와 계산된 AS-OF 피처를 비교하여 누수 여부 확인"""
    user_sample = df[df['customer_id'] == sample_user_id].sort_values('t_dat')
    
    print(f"\n📊 [Monitoring] User: {sample_user_id}")
    cols_to_show = ['t_dat', 'article_id', 'price', 'cum_cnt', 'asof_avg_price', 'asof_recency_days']
    
    # 상위 10개 행 출력: 첫 세션의 통계량이 0(또는 기본값)인지 확인
    print(user_sample[cols_to_show].head(10).to_string())
    
    # 검증 포인트 1: 첫 번째 행의 cum_cnt는 무조건 0이어야 함 (AS-OF Shift 확인)
    first_cum_cnt = user_sample.iloc[0]['cum_cnt']
    assert first_cum_cnt == 0, f"❌ Leakage Detected! First record should have 0 cumulative count, but got {first_cum_cnt}"
    
    # 검증 포인트 2: 시퀀스 길이 일치 확인
    # 리스트로 묶인 후 시퀀스 길이가 아이템 수와 같은지 체크 (aggregation 직후 실행)
    print(f"✅ AS-OF Shift Logic Passed: First session stats are properly masked.")

# make_user_features 함수 마지막 부분에 추가
def validate_final_lists(final_df):
    print("\n🔍 [Check] Sequence Length Consistency:")
    # 랜덤 유저 1명의 리스트 길이들 비교
    sample = final_df.sample(1).iloc[0]
    lengths = {col: len(sample[col]) for col in final_df.columns if isinstance(sample[col], list)}
    
    # 모든 리스트 피처의 길이가 동일한지 확인
    unique_lengths = set(lengths.values())
    if len(unique_lengths) == 1:
        print(f"✅ All feature sequences have consistent length: {list(unique_lengths)[0]}")
    else:
        print(f"❌ Consistency Error! Found varying lengths: {lengths}")
        
def monitor_processor_storage(processor):
    print("\n⚡ [Processor Monitoring] Fast Lookup Table Sanity Check")
    
    # 1. 메모리 사용량 대략적 파악
    num_users = len(processor.u_dyn_buckets)
    print(f" -> Total Users in Dynamic Storage: {num_users:,}")
    
    # 2. 랜덤 샘플 유저 정합성 체크
    if num_users > 0:
        sample_uid = list(processor.u_dyn_buckets.keys())[0]
        
        dyn_b = processor.u_dyn_buckets[sample_uid]
        dyn_c = processor.u_dyn_conts[sample_uid]
        dyn_t = processor.u_dyn_time[sample_uid]
        
        print(f" -> Sample User Map ID: {sample_uid}")
        print(f" -> Dynamic Buckets Shape: {dyn_b.shape} (Expect: [Seq_len, 3])")
        print(f" -> Dynamic Conts Shape: {dyn_c.shape} (Expect: [Seq_len, 4])")
        print(f" -> Dynamic Time Shape: {dyn_t.shape} (Expect: [Seq_len, 2])")
        
        # 3. 값의 범위 체크 (Scaling/Bucketing 확인)
        print(f" -> Conts Mean (Scaled): {dyn_c.mean():.4f} (Expect: Near 0 if well scaled)")
        print(f" -> Buckets Max: {dyn_b.max()} (Expect: <= 10 or 11)")





# =====================================================================
# Phase 2: Model Definitions
# =====================================================================




import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random

class SASRecDataset_v3_obsolete(Dataset):
    def __init__(self, processor, global_now_str="2020-09-22", max_len=30, is_train=True):
        self.processor = processor
        self.max_len = max_len
        self.is_train = is_train
        self.user_ids = processor.user_ids
        
        global_now_dt = pd.to_datetime(global_now_str)
        self.now_ordinal = global_now_dt.toordinal()
        self.now_week = global_now_dt.isocalendar().week

        if self.is_train: # 훈련용 데이터셋일 때만 딱 한 번 로그를 보고 싶다면!
            self.verify_session_logic()
            
    def verify_session_logic(self):
        """디버깅용: 세션 분리 및 셔플링이 정상적으로 작동하는지 1명의 유저를 뽑아 콘솔에 출력합니다."""
        print("\n" + "="*75)
        print("🕵️‍♂️ [Dataset Monitor] Session Grouping & Shuffling Verification")
        print("="*75)
        
        # 시퀀스 길이가 5 이상인 유저 1명 찾기
        sample_user = None
        for uid in self.user_ids:
            u_mapped_id = self.processor.user2id.get(uid, 0)
            if len(self.processor.u_seqs[u_mapped_id]) > 5:
                sample_user = uid
                break
                
        if not sample_user:
            print("충분한 시퀀스를 가진 유저가 없습니다.")
            return
            
        u_mapped_id = self.processor.user2id[sample_user]
        seq_raw = self.processor.u_seqs[u_mapped_id]
        time_deltas_raw = self.processor.u_deltas[u_mapped_id]
        
        # 1. 세션 ID 생성 (수정된 로직)
        session_ids_raw = [1]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == time_deltas_raw[i-1]:
                session_ids_raw.append(session_ids_raw[-1])
            else:
                session_ids_raw.append(session_ids_raw[-1] + 1)
                
        # 2. 셔플링 시뮬레이션 (수정된 로직 적용, 무조건 섞이도록 강제)
        input_indices = list(range(len(seq_raw)))
        grouped_indices = []
        current_group = [input_indices[0]]
        for i in range(1, len(input_indices)):
            if time_deltas_raw[input_indices[i]] == time_deltas_raw[input_indices[i-1]]:
                current_group.append(input_indices[i])
            else:
                grouped_indices.append(current_group)
                current_group = [input_indices[i]]
        if current_group: grouped_indices.append(current_group)
        
        shuffled_indices = []
        for group in grouped_indices:
            group_copy = group.copy()
            if len(group_copy) > 1:
                random.shuffle(group_copy) # 테스트용이므로 100% 섞음
            shuffled_indices.extend(group_copy)
            
        print(f"👤 Sample User ID: {sample_user}")
        print(f"{'Orig_Idx':<9} | {'Item_ID':<11} | {'Delta (Days)':<13} | {'Session_ID':<11} | {'Shuffled_Idx':<13}")
        print("-" * 75)
        
        for i in range(len(seq_raw)):
            orig_idx = i
            shuff_idx = shuffled_indices[i] # 셔플된 결과가 위치할 인덱스
            item_id = seq_raw[shuff_idx]
            delta = time_deltas_raw[shuff_idx]
            sess_id = session_ids_raw[shuff_idx]
            
            # 같은 세션(Session ID)끼리 묶여있는지, Orig_Idx가 섞였는지 확인
            print(f"{orig_idx:<9} | {item_id:<11} | {delta:<13} | {sess_id:<11} | {shuff_idx:<13}")
        print("="*75 + "\n")
    def _shuffle_indices_within_session(self, indices, time_deltas_raw):
        if len(indices) <= 1: return indices
        grouped_indices = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            idx = indices[i]
            prev_idx = indices[i-1] # 💡 추가
            
            # 💡 [핵심 버그 수정] 0인지 묻는 게 아니라, 이전 아이템과 남은 일수(Delta)가 같은지 확인!
            if time_deltas_raw[idx] == time_deltas_raw[prev_idx]:
                current_group.append(idx)
            else:
                grouped_indices.append(current_group)
                current_group = [idx]
                
        if current_group: grouped_indices.append(current_group)
        
        shuffled_indices = []
        for group in grouped_indices:
            if len(group) > 1 and random.random() < 0.5:
                random.shuffle(group)
            shuffled_indices.extend(group)
        return shuffled_indices

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
        
        seq_raw = self.processor.u_seqs[u_mapped_id]
        time_deltas_raw = self.processor.u_deltas[u_mapped_id]
        # 1. 전처리: ID 매핑 및 타임 델타 버킷화

        session_ids_raw = [1]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == time_deltas_raw[i-1]:
                session_ids_raw.append(session_ids_raw[-1]) # 델타가 같으면 같은 당일(세션)!
            else:
                session_ids_raw.append(session_ids_raw[-1] + 1) # 델타가 달라지면 새 세션 시작
        
        seq_mapped = [self.processor.item2id.get(iid, 0) for iid in seq_raw]
        bins = np.array([0, 3, 7, 14, 30, 60, 180, 330, 395])
        time_buckets = np.digitize(time_deltas_raw, bins, right=False).tolist()
        
        # 2. 인덱스 설정 및 셔플링
        if self.is_train:
            # All-time 학습: 마지막 아이템(정답)을 제외한 인덱스들
            input_indices = list(range(len(seq_raw) - 1))
            shuffled_indices = self._shuffle_indices_within_session(input_indices, time_deltas_raw)
        else:
            shuffled_indices = list(range(len(seq_raw)))

        # 3. max_len 슬라이싱 및 패딩 계산
        shuffled_indices = shuffled_indices[-self.max_len:]
        pad_len = self.max_len - len(shuffled_indices)
        
        # 4. 시퀀스 데이터 추출 (Input, Target, Time Interval)
        input_seq = [seq_mapped[i] for i in shuffled_indices]
        input_time = [time_buckets[i] for i in shuffled_indices]
        # 💡 [적용] 셔플된 순서에 맞춰 session_id도 재배열 (당일 아이템은 섞여도 같은 번호!)
        input_session = [session_ids_raw[i] for i in shuffled_indices]
        
        if self.is_train:
            target_seq = [seq_mapped[i + 1] for i in shuffled_indices]
        else:
            target_seq = []
            

        # 5. 동적 피처 행렬 추출 및 Target_Now 계산
        d_buckets = self.processor.u_dyn_buckets[u_mapped_id][shuffled_indices] # [len, 3]
        d_conts = self.processor.u_dyn_conts[u_mapped_id][shuffled_indices]     # [len, 4]
        d_cats = self.processor.u_dyn_cats[u_mapped_id][shuffled_indices]       # [len, 1]
        d_time = self.processor.u_dyn_time[u_mapped_id][shuffled_indices]       # [len, 2]

        input_dates = d_time[:, 0].tolist()
        
        if self.is_train:
            target_indices = [idx + 1 for idx in shuffled_indices]
            step_target_times = self.processor.u_dyn_time[u_mapped_id][target_indices, 0]
            # 💡 [신규] 바로 다음 타겟(정답)의 실제 주차(Week) 추출
            step_target_weeks = self.processor.u_dyn_time[u_mapped_id][target_indices, 1] 
            
            dynamic_offsets = np.clip(step_target_times - d_time[:, 0], 0, 365).astype(np.int64)
        else:
            dynamic_offsets = np.clip(self.now_ordinal - d_time[:, 0], 0, 365).astype(np.int64)
            # 💡 [신규] 추론 시에는 글로벌 타겟 주차(now_week)로 통일하여 예측 방향 강제
            step_target_weeks = np.array([self.now_week] * len(shuffled_indices))
        current_weeks = d_time[:, 1]

        # 6. 패딩 처리 (Left Padding)
        input_padded = [0] * pad_len + input_seq
        time_padded = [0] * pad_len + input_time
        target_padded = [0] * pad_len + target_seq if self.is_train else [0] * self.max_len
        padding_mask = [True] * pad_len + [False] * len(input_seq)
        session_padded = [0] * pad_len + input_session
        target_week_padded = [0] * pad_len + step_target_weeks.tolist()
        dates_padded = [0] * pad_len + input_dates
        # 7. 아이템 사이드 정보 추출 (Input Padded 기반)
        item_side_info = self.processor.i_side_arr[input_padded]
        type_ids = item_side_info[:, 0]
        color_ids = item_side_info[:, 1]
        graphic_ids = item_side_info[:, 2]
        section_ids = item_side_info[:, 3]
        
        # 8. 동적 피처 패딩 처리
        # 0으로 채워진 패딩 행렬 생성
        pad_b = np.zeros((pad_len, 3), dtype=np.int64)
        pad_c = np.zeros((pad_len, 4), dtype=np.float32)
        pad_cat = np.zeros((pad_len, 1), dtype=np.int64)
        pad_1d = np.zeros(pad_len, dtype=np.int64)

        d_buckets_p = np.vstack([pad_b, d_buckets]) if len(shuffled_indices) > 0 else pad_b
        d_conts_p = np.vstack([pad_c, d_conts]) if len(shuffled_indices) > 0 else pad_c
        d_cats_p = np.vstack([pad_cat, d_cats]) if len(shuffled_indices) > 0 else pad_cat
        
        offset_p = np.concatenate([pad_1d, dynamic_offsets]) if len(shuffled_indices) > 0 else pad_1d
        week_p = np.concatenate([pad_1d, current_weeks]) if len(shuffled_indices) > 0 else pad_1d

        # 9. 정적 피처 추출
        s_buckets = self.processor.u_static_buckets[u_mapped_id]
        s_cats = self.processor.u_static_cats[u_mapped_id]

        return {
            'user_ids': user_id,
            # 시퀀스 텐서
            'item_ids': torch.tensor(input_padded, dtype=torch.long),
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'time_bucket_ids': torch.tensor(time_padded, dtype=torch.long),
            'session_ids': torch.tensor(session_padded, dtype=torch.long),
            # 아이템 사이드 정보
            'type_ids': torch.tensor(type_ids, dtype=torch.long),
            'color_ids': torch.tensor(color_ids, dtype=torch.long),
            'graphic_ids': torch.tensor(graphic_ids, dtype=torch.long),
            'section_ids': torch.tensor(section_ids, dtype=torch.long),
            
            # 동적 세션 피처
            'price_bucket': torch.tensor(d_buckets_p[:, 0], dtype=torch.long),
            'cnt_bucket': torch.tensor(d_buckets_p[:, 1], dtype=torch.long),
            'recency_bucket': torch.tensor(d_buckets_p[:, 2], dtype=torch.long),
            'cont_feats': torch.tensor(d_conts_p, dtype=torch.float32),
            'channel_ids': torch.tensor(d_cats_p[:, 0], dtype=torch.long),
            
            # 글로벌 컨텍스트 (시간 거리 및 계절)
            'recency_offset': torch.tensor(offset_p, dtype=torch.long),
            'current_week': torch.tensor(week_p, dtype=torch.long),
            'target_week': torch.tensor(target_week_padded, dtype=torch.long),
            
            # 정적 유저 피처
            'age_bucket': torch.tensor(s_buckets[0], dtype=torch.long),
            'club_status_ids': torch.tensor(s_cats[0], dtype=torch.long),
            'news_freq_ids': torch.tensor(s_cats[1], dtype=torch.long),
            'fn_ids': torch.tensor(s_cats[2], dtype=torch.long),
            'active_ids': torch.tensor(s_cats[3], dtype=torch.long),

            'interaction_dates': torch.tensor(dates_padded, dtype=torch.long),
        }

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

class SASRecDataset_v3(Dataset):
    def __init__(self, processor, global_now_str="2020-09-22", max_len=30, is_train=True):
        self.processor = processor
        self.max_len = max_len
        self.is_train = is_train
        self.user_ids = processor.user_ids
        
        global_now_dt = pd.to_datetime(global_now_str)
        self.now_ordinal = global_now_dt.toordinal()
        self.now_week = global_now_dt.isocalendar().week

        if self.is_train: 
            self.verify_session_logic()
            
    def verify_session_logic(self):
        """디버깅용: 세션 분리 및 셔플링이 정상적으로 작동하는지 1명의 유저를 뽑아 콘솔에 출력합니다."""
        print("\n" + "="*75)
        print("🕵️‍♂️ [Dataset Monitor] Session Grouping & Shuffling Verification")
        print("="*75)
        
        sample_user = None
        for uid in self.user_ids:
            u_mapped_id = self.processor.user2id.get(uid, 0)
            if len(self.processor.u_seqs[u_mapped_id]) > 5:
                sample_user = uid
                break
                
        if not sample_user:
            print("충분한 시퀀스를 가진 유저가 없습니다.")
            return
            
        u_mapped_id = self.processor.user2id[sample_user]
        seq_raw = self.processor.u_seqs[u_mapped_id]
        time_deltas_raw = self.processor.u_deltas[u_mapped_id]
        
        session_ids_raw = [1]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == time_deltas_raw[i-1]:
                session_ids_raw.append(session_ids_raw[-1])
            else:
                session_ids_raw.append(session_ids_raw[-1] + 1)
                
        input_indices = list(range(len(seq_raw)))
        grouped_indices = []
        current_group = [input_indices[0]]
        for i in range(1, len(input_indices)):
            if time_deltas_raw[input_indices[i]] == time_deltas_raw[input_indices[i-1]]:
                current_group.append(input_indices[i])
            else:
                grouped_indices.append(current_group)
                current_group = [input_indices[i]]
        if current_group: grouped_indices.append(current_group)
        
        shuffled_indices = []
        for group in grouped_indices:
            group_copy = group.copy()
            if len(group_copy) > 1:
                random.shuffle(group_copy) 
            shuffled_indices.extend(group_copy)
            
        print(f"👤 Sample User ID: {sample_user}")
        print(f"{'Orig_Idx':<9} | {'Item_ID':<11} | {'Delta (Days)':<13} | {'Session_ID':<11} | {'Shuffled_Idx':<13}")
        print("-" * 75)
        
        for i in range(len(seq_raw)):
            orig_idx = i
            shuff_idx = shuffled_indices[i] 
            item_id = seq_raw[shuff_idx]
            delta = time_deltas_raw[shuff_idx]
            sess_id = session_ids_raw[shuff_idx]
            print(f"{orig_idx:<9} | {item_id:<11} | {delta:<13} | {sess_id:<11} | {shuff_idx:<13}")
        print("="*75 + "\n")

    def _shuffle_indices_within_session(self, indices, time_deltas_raw):
        if len(indices) <= 1: return indices
        grouped_indices = []
        current_group = [indices[0]]
        
        for i in range(1, len(indices)):
            idx = indices[i]
            prev_idx = indices[i-1] 
            
            if time_deltas_raw[idx] == time_deltas_raw[prev_idx]:
                current_group.append(idx)
            else:
                grouped_indices.append(current_group)
                current_group = [idx]
                
        if current_group: grouped_indices.append(current_group)
        
        shuffled_indices = []
        for group in grouped_indices:
            if len(group) > 1 and random.random() < 0.5:
                random.shuffle(group)
            shuffled_indices.extend(group)
        return shuffled_indices

    def __len__(self):
        return len(self.user_ids)

    def _get_view_features(self, u_mapped_id, indices, seq_mapped, time_buckets, session_ids_raw, d_time_full):
        """💡 단일 View에 대한 슬라이싱 및 패딩을 수행하여 딕셔너리로 반환하는 헬퍼 함수"""
        indices = indices[-self.max_len:]
        pad_len = self.max_len - len(indices)
        
        input_seq = [seq_mapped[i] for i in indices]
        input_time = [time_buckets[i] for i in indices]
        input_session = [session_ids_raw[i] for i in indices]
        
        d_buckets = self.processor.u_dyn_buckets[u_mapped_id][indices] 
        d_conts = self.processor.u_dyn_conts[u_mapped_id][indices]    
        d_cats = self.processor.u_dyn_cats[u_mapped_id][indices]      
        d_time = self.processor.u_dyn_time[u_mapped_id][indices]      

        input_dates = d_time[:, 0].tolist()
        current_weeks = d_time[:, 1]
        
        if self.is_train:
            target_seq = [seq_mapped[i + 1] for i in indices]
            target_indices = [idx + 1 for idx in indices]
            step_target_times = self.processor.u_dyn_time[u_mapped_id][target_indices, 0]
            step_target_weeks = self.processor.u_dyn_time[u_mapped_id][target_indices, 1] 
            dynamic_offsets = np.clip(step_target_times - d_time[:, 0], 0, 365).astype(np.int64)
        else:
            target_seq = []
            dynamic_offsets = np.clip(self.now_ordinal - d_time[:, 0], 0, 365).astype(np.int64)
            step_target_weeks = np.array([self.now_week] * len(indices))

        # 패딩 처리
        input_padded = [0] * pad_len + input_seq
        time_padded = [0] * pad_len + input_time
        target_padded = [0] * pad_len + target_seq if self.is_train else [0] * self.max_len
        padding_mask = [True] * pad_len + [False] * len(input_seq)
        session_padded = [0] * pad_len + input_session
        target_week_padded = [0] * pad_len + step_target_weeks.tolist()
        dates_padded = [0] * pad_len + input_dates
        
        # 아이템 사이드 정보
        item_side_info = self.processor.i_side_arr[input_padded]
        
        # 동적 피처 패딩
        pad_b = np.zeros((pad_len, 3), dtype=np.int64)
        pad_c = np.zeros((pad_len, 4), dtype=np.float32)
        pad_cat = np.zeros((pad_len, 1), dtype=np.int64)
        pad_1d = np.zeros(pad_len, dtype=np.int64)

        d_buckets_p = np.vstack([pad_b, d_buckets]) if len(indices) > 0 else pad_b
        d_conts_p = np.vstack([pad_c, d_conts]) if len(indices) > 0 else pad_c
        d_cats_p = np.vstack([pad_cat, d_cats]) if len(indices) > 0 else pad_cat
        offset_p = np.concatenate([pad_1d, dynamic_offsets]) if len(indices) > 0 else pad_1d
        week_p = np.concatenate([pad_1d, current_weeks]) if len(indices) > 0 else pad_1d

        return {
            'item_ids': torch.tensor(input_padded, dtype=torch.long),
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'time_bucket_ids': torch.tensor(time_padded, dtype=torch.long),
            'session_ids': torch.tensor(session_padded, dtype=torch.long),
            'type_ids': torch.tensor(item_side_info[:, 0], dtype=torch.long),
            'color_ids': torch.tensor(item_side_info[:, 1], dtype=torch.long),
            'graphic_ids': torch.tensor(item_side_info[:, 2], dtype=torch.long),
            'section_ids': torch.tensor(item_side_info[:, 3], dtype=torch.long),
            'price_bucket': torch.tensor(d_buckets_p[:, 0], dtype=torch.long),
            'cnt_bucket': torch.tensor(d_buckets_p[:, 1], dtype=torch.long),
            'recency_bucket': torch.tensor(d_buckets_p[:, 2], dtype=torch.long),
            'cont_feats': torch.tensor(d_conts_p, dtype=torch.float32),
            'channel_ids': torch.tensor(d_cats_p[:, 0], dtype=torch.long),
            'recency_offset': torch.tensor(offset_p, dtype=torch.long),
            'current_week': torch.tensor(week_p, dtype=torch.long),
            'target_week': torch.tensor(target_week_padded, dtype=torch.long),
            'interaction_dates': torch.tensor(dates_padded, dtype=torch.long),
        }

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
        
        seq_raw = self.processor.u_seqs[u_mapped_id]
        time_deltas_raw = self.processor.u_deltas[u_mapped_id]
        
        session_ids_raw = [1]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == time_deltas_raw[i-1]:
                session_ids_raw.append(session_ids_raw[-1]) 
            else:
                session_ids_raw.append(session_ids_raw[-1] + 1)
        
        seq_mapped = [self.processor.item2id.get(iid, 0) for iid in seq_raw]
        bins = np.array([0, 3, 7, 14, 30, 60, 180, 330, 395])
        time_buckets = np.digitize(time_deltas_raw, bins, right=False).tolist()

        # 정적 피처는 View 분리와 상관없이 공통입니다.
        s_buckets = self.processor.u_static_buckets[u_mapped_id]
        s_cats = self.processor.u_static_cats[u_mapped_id]
        
        base_dict = {
            'user_ids': user_id,
            'age_bucket': torch.tensor(s_buckets[0], dtype=torch.long),
            'club_status_ids': torch.tensor(s_cats[0], dtype=torch.long),
            'news_freq_ids': torch.tensor(s_cats[1], dtype=torch.long),
            'fn_ids': torch.tensor(s_cats[2], dtype=torch.long),
            'active_ids': torch.tensor(s_cats[3], dtype=torch.long),
        }

        if self.is_train:
            # 💡 [핵심] 훈련 시 두 번의 독립적인 셔플링 수행
            input_indices = list(range(len(seq_raw) - 1))
            indices_v1 = self._shuffle_indices_within_session(input_indices, time_deltas_raw)
            #indices_v2 = self._shuffle_indices_within_session(input_indices, time_deltas_raw)
            
            # View 1 생성 및 병합 (_v1)
            view_1 = self._get_view_features(u_mapped_id, indices_v1, seq_mapped, time_buckets, session_ids_raw, self.processor.u_dyn_time[u_mapped_id])
            for k, v in view_1.items():
                base_dict[f"{k}_v1"] = v
                
            # View 2 생성 및 병합 (_v2)
            #view_2 = self._get_view_features(u_mapped_id, indices_v2, seq_mapped, time_buckets, session_ids_raw, self.processor.u_dyn_time[u_mapped_id])
            #for k, v in view_2.items():
            #    base_dict[f"{k}_v2"] = v

        else:
            # 평가 시 셔플링 없이 단일 View 생성 (기존 로직과 동일)
            indices = list(range(len(seq_raw)))
            view_eval = self._get_view_features(u_mapped_id, indices, seq_mapped, time_buckets, session_ids_raw, self.processor.u_dyn_time[u_mapped_id])
            for k, v in view_eval.items():
                base_dict[k] = v

        return base_dict




class ContinuousFeatureTokenizer_v3(nn.Module):
    def __init__(self, num_cont_feats, embed_dim):
        super().__init__()
        self.num_cont_feats = num_cont_feats
        self.embed_dim = embed_dim
        
        # 가중치와 편향의 크기는 기존과 동일하게 유지 (피처별로 독립적인 파라미터)
        self.weight = nn.Parameter(torch.Tensor(num_cont_feats, embed_dim))
        self.bias = nn.Parameter(torch.Tensor(num_cont_feats, embed_dim))
        
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        입력 x: (Batch, seq_len, num_cont_feats)
        출력: (Batch, seq_len, num_cont_feats, embed_dim)
        """
        # 1. 입력의 마지막 차원 뒤에 하나를 추가하여 (B, S, F, 1)로 만듦
        x = x.unsqueeze(-1) 
        
        # 2. 가중치와 편향을 (1, 1, F, E) 형태로 view를 조정하여 브로드캐스팅 준비
        # 이렇게 하면 (B, S, F, 1) * (1, 1, F, E) 연산이 수행됩니다.
        # 즉, 시퀀스의 모든 시점(S)에 대해 동일한 피처 가중치가 적용됩니다.
        weight = self.weight.view(1, 1, self.num_cont_feats, self.embed_dim)
        bias = self.bias.view(1, 1, self.num_cont_feats, self.embed_dim)
        
        # 3. 선형 투영 수행
        tokenized_x = x * weight + bias
        
        return tokenized_x


class SASRecUserTower_v3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.max_len = args.max_len
        self.dropout_rate = args.dropout
        static_mlp_input_dim = 116
        # ==================================================================
        # 1. Sequence Embeddings (Dynamic: Short-term Intent)
        # ==================================================================
        self.item_proj = nn.Linear(args.pretrained_dim, self.d_model)
        self.item_id_emb = nn.Embedding(args.num_items + 1, self.d_model, padding_idx=0)
        
        self.type_emb = nn.Embedding(args.num_prod_types + 1, self.d_model, padding_idx=0)
        #self.color_emb = nn.Embedding(args.num_colors + 1, self.d_model, padding_idx=0)
        #self.graphic_emb = nn.Embedding(args.num_graphics + 1, self.d_model, padding_idx=0)
        #self.section_emb = nn.Embedding(args.num_sections + 1, self.d_model, padding_idx=0)

        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        num_time_buckets = 12 
        #self.time_emb = nn.Embedding(num_time_buckets, self.d_model, padding_idx=0)

        # 💡 [신규] Global Context Embeddings
        # recency_offset: 0~365일 (366개) / current_week: 1~53주차 (54개)
        
        #self.global_recency_emb = nn.Embedding(367, self.d_model)
        self.recency_proj = nn.Linear(1, self.d_model) # (추가) 연속된 숫자를 벡터로!
        self.week_proj = nn.Linear(2, self.d_model)
        self.global_week_emb = nn.Embedding(54, self.d_model)
        
        # ------------------------------------------------------------------
        # [게이트 및 기타 모듈 세팅 - 기존 동일]
        # ------------------------------------------------------------------
        self.seq_gate = nn.Parameter(torch.ones(9-4)) 
        self.static_gate = nn.Parameter(torch.ones(13-2))

        self.emb_ln_item = nn.LayerNorm(self.d_model)
        self.emb_dropout_item = nn.Dropout(self.dropout_rate)
        
        self.emb_ln_feat = nn.LayerNorm(self.d_model)
        self.emb_dropout_feat = nn.Dropout(self.dropout_rate)

        self.global_ln = nn.LayerNorm(self.d_model)
        
        # Item Transformer (2층, 4헤드)
        item_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model , nhead=args.nhead , dim_feedforward=self.d_model * 2,
            dropout=self.dropout_rate, activation='gelu', norm_first=True, batch_first=True
        )
        self.item_transformer = nn.TransformerEncoder(item_layer, num_layers=args.num_layers)

        # Feature Transformer (1층, 1헤드 다이어트)
        feat_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=1, dim_feedforward=self.d_model // 2,
            dropout=self.dropout_rate, activation='gelu', norm_first=True, batch_first=True
        )
        self.feature_transformer = nn.TransformerEncoder(feat_layer, num_layers=1)

        # ==================================================================
        # 2. Static Embeddings & MLP - 기존과 동일 (생략 없이 원형 유지)
        # ==================================================================
        mid_dim, low_dim = 16, 4
        self.age_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        self.price_emb = nn.Embedding(11, mid_dim, padding_idx=0)    
        #self.cnt_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        #self.recency_emb = nn.Embedding(11, mid_dim, padding_idx=0)  

        self.channel_emb = nn.Embedding(4, low_dim, padding_idx=0)   
        self.club_status_emb = nn.Embedding(4, low_dim, padding_idx=0) 
        self.news_freq_emb = nn.Embedding(3, low_dim, padding_idx=0)   
        self.fn_emb = nn.Embedding(3, low_dim, padding_idx=0)        
        self.active_emb = nn.Embedding(3, low_dim, padding_idx=0)    

        self.num_cont_feats = 4
        self.cont_embed_dim = 16 
        self.cont_tokenizer = ContinuousFeatureTokenizer_v3(self.num_cont_feats, self.cont_embed_dim)

        total_static_input_dim = (mid_dim * 4) + (low_dim * 5) + (self.num_cont_feats * self.cont_embed_dim)
        
        self.static_mlp = nn.Sequential(
            nn.Linear(static_mlp_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.apply(self._init_weights)
        
        
        self.seq_proj = nn.Linear(self.d_model, self.d_model)
        self.feat_proj = nn.Linear(self.d_model, self.d_model)
        self.static_proj = nn.Linear(self.d_model, self.d_model)

        # 3개 스트림 concat → 3개 gate 스칼라 출력
        self.fusion_gate = nn.Linear(self.d_model * 3, 3)

        # output_proj는 d_model * 3 → d_model에서 d_model → d_model로 변경
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # 💡 [신규] 의도 프로토타입 (Intent Prototypes) 파라미터 추가
        self.num_intents = 256  # 패션 도메인의 세부 스타일(하객룩, 바캉스룩, 고프코어 등)을 포괄할 충분한 K개
        self.intent_prototypes = nn.Parameter(torch.Tensor(self.num_intents, self.d_model))
        nn.init.xavier_uniform_(self.intent_prototypes)
        
        # 안전한 '소프트 랜딩' 초기화를 위한 플래그
        self.prototypes_initialized = False 

        
        # first -> sigmoid gate
        nn.init.xavier_normal_(self.week_proj.weight)
        nn.init.xavier_normal_(self.recency_proj.weight)

        #
        if self.week_proj.bias is not None:
            nn.init.constant_(self.week_proj.bias, 0)
        if self.recency_proj.bias is not None:
            nn.init.constant_(self.recency_proj.bias, 0)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    def forward(self, 
                pretrained_vecs, item_ids, time_bucket_ids, 
                type_ids, color_ids, graphic_ids, section_ids,
                age_bucket, price_bucket, cnt_bucket, recency_bucket,
                channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids,
                cont_feats, 
                recency_offset, current_week,target_week=None, # 💡 [신규 입력]
                padding_mask=None, training_mode=True):
        
        device = item_ids.device
        seq_len = item_ids.size(1)
        batch_size = item_ids.size(0)
        
        s_g_raw = torch.sigmoid(self.seq_gate) 
        u_g_raw = torch.sigmoid(self.static_gate)
        s_g = s_g_raw * torch.ones_like(s_g_raw)
        u_g = u_g_raw * torch.ones_like(u_g_raw)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeddings = self.pos_emb(positions)
        causal_mask = self.get_causal_mask(seq_len, device)

        # -----------------------------------------------------------
        # 💡 [핵심] Global Context 생성 (Batch, 1, d_model)
        # -----------------------------------------------------------

        age_bucket = age_bucket.unsqueeze(1).expand(-1, seq_len)
        club_status_ids = club_status_ids.unsqueeze(1).expand(-1, seq_len)
        news_freq_ids = news_freq_ids.unsqueeze(1).expand(-1, seq_len)
        fn_ids = fn_ids.unsqueeze(1).expand(-1, seq_len)
        active_ids = active_ids.unsqueeze(1).expand(-1, seq_len)
        
        
        # -----------------------------------------------------------
        # Phase 1: Global Context 생성
        # 💡 변경: recency_offset, current_week가 이제 (Batch, seq_len) 이므로 unsqueeze(1) 불필요!
        # -----------------------------------------------------------
        recency_scaled = (recency_offset.float() / 365.0).unsqueeze(-1) # (Batch, seq_len, 1)
        r_offset_vec = self.recency_proj(recency_scaled) * s_g[0] # (Batch, seq_len, d_model)
        
        week_rad = (current_week.float() / 52.0) * (2 * math.pi)
        week_sin = torch.sin(week_rad).unsqueeze(-1) 
        week_cos = torch.cos(week_rad).unsqueeze(-1) 
        week_cyclical = torch.cat([week_sin, week_cos], dim=-1) # (Batch, seq_len, 2)
        
        w_vec = self.week_proj(week_cyclical) * s_g[1] # (Batch, seq_len, d_model)
        
        # 기존의 .unsqueeze(1) 제거 (이미 seq_len 차원이 있음)
        global_context_raw = (r_offset_vec + w_vec) 
        global_context = self.global_ln(global_context_raw)
        
        # -----------------------------------------------------------
        # Phase 1: FDSA - Item ID Stream 
        # -----------------------------------------------------------
        item_emb = self.item_proj(pretrained_vecs) 
        item_emb += self.item_id_emb(item_ids) * s_g[2]

        item_emb += pos_embeddings
        
        # 🔥 Early Injection: 시퀀스 전체를 현재 날짜의 좌표계로 이동
        item_emb = item_emb + global_context
        
        item_emb = self.emb_ln_item(item_emb)
        item_emb = self.emb_dropout_item(item_emb)

        item_output = self.item_transformer(
            item_emb, mask=causal_mask, src_key_padding_mask=padding_mask# 💡 SDPA 엔진에게 Causal 연산임을 직접 알려줌
        )

        # -----------------------------------------------------------
        # Phase 2: FDSA - Feature Stream 
        # -----------------------------------------------------------
        feat_emb = self.type_emb(type_ids) * s_g[3]
        #item_emb += self.time_emb(time_bucket_ids) * s_g[1]
        #feat_emb += self.color_emb(color_ids) * s_g[3]
        #feat_emb += self.graphic_emb(graphic_ids) * s_g[4]
        #feat_emb += self.section_emb(section_ids) * s_g[5]
        feat_emb += pos_embeddings 
        
        # 🔥 Feature Stream에도 계절/시점 주입 (색상, 타입도 계절을 타기 때문)
        feat_emb = feat_emb + global_context
        
        feat_emb = self.emb_ln_feat(feat_emb)
        feat_emb = self.emb_dropout_feat(feat_emb)

        feat_output = self.feature_transformer(
            feat_emb, mask=causal_mask, src_key_padding_mask=padding_mask# 💡 SDPA 엔진에게 Causal 연산임을 직접 알려줌
        )

        # -----------------------------------------------------------
        # Phase 3: Static Encoding (기존 동일)
        # -----------------------------------------------------------
        emb_age = self.age_emb(age_bucket) * u_g[0]
        emb_price = self.price_emb(price_bucket) * u_g[1]
        #emb_cnt = self.cnt_emb(cnt_bucket) * u_g[2]
        #emb_rec = self.recency_emb(recency_bucket) * u_g[3]
        
        emb_chan = self.channel_emb(channel_ids) * u_g[2]
        emb_club = self.club_status_emb(club_status_ids) * u_g[3]
        emb_news = self.news_freq_emb(news_freq_ids) * u_g[4]
        emb_fn = self.fn_emb(fn_ids) * u_g[5]
        emb_act = self.active_emb(active_ids) * u_g[6]
        
        tokenized_x = self.cont_tokenizer(cont_feats) # cont_feats: (Batch, seq_len, 4)
        
        # Gate shape 조정 (Batch, seq_len, 4, 16)에 맞게 브로드캐스팅
        cont_gates = u_g[7:11].view(1, 1, 4, 1) 
        tokenized_x = tokenized_x * cont_gates
        cont_tokens = F.relu(tokenized_x).view(batch_size, seq_len, -1) # (Batch, seq_len, 64)
        
        static_input = torch.cat([
            emb_age, emb_price, #emb_cnt, emb_rec,
            emb_chan, emb_club, emb_news, emb_fn, emb_act,
            cont_tokens
        ], dim=-1) # 차원이 3D이므로 마지막 차원(dim=-1) 결합
        
        user_profile_vec = self.static_mlp(static_input) # 출력: (Batch, seq_len, d_model)
        # -----------------------------------------------------------
        if target_week is not None:
            # target_week: (Batch, seq_len)
            t_week_rad = (target_week.float() / 52.0) * (2 * math.pi)
            t_week_sin = torch.sin(t_week_rad).unsqueeze(-1) 
            t_week_cos = torch.cos(t_week_rad).unsqueeze(-1) 
            t_week_cyclical = torch.cat([t_week_sin, t_week_cos], dim=-1) # (Batch, seq_len, 2)
            
            # 🔥 핵심: current_week와 동일한 week_proj 가중치를 공유하여 투영!
            # 동일한 주차면 완전히 동일한 벡터 방향성을 갖게 됩니다.
            target_week_vec = self.week_proj(t_week_cyclical)* s_g[4] # (Batch, seq_len, d_model)

        # -----------------------------------------------------------
        # Phase 5: Late Fusion
        # 💡 변경: user_profile_vec를 확장(expand)할 필요가 없어짐
        # -----------------------------------------------------------
        if training_mode:
            # 1. 각 스트림 독립 projection
            h_seq = self.seq_proj(item_output)        # [B, S, D]
            h_feat = self.feat_proj(feat_output)      # [B, S, D]
            h_static = self.static_proj(user_profile_vec)  # [B, S, D]

            # 2. Dynamic gate 계산
            # concat된 3스트림을 입력으로 각 스트림의 기여도를 동적으로 결정
            gate_input = torch.cat([h_seq, h_feat, h_static], dim=-1)  # [B, S, D*3]
            gate = torch.softmax(self.fusion_gate(gate_input), dim=-1)  # [B, S, 3]
            g_seq, g_feat, g_static = gate.unbind(dim=-1)  # 각각 [B, S]

            # 3. 가중합 fusion
            final_vec = (
                g_seq.unsqueeze(-1) * h_seq +
                g_feat.unsqueeze(-1) * h_feat +
                g_static.unsqueeze(-1) * h_static
            )  # [B, S, D]

            # 4. output_proj 및 target_week 주입
            final_vec = self.output_proj(final_vec)
            if target_week is not None:
                final_vec = final_vec + target_week_vec

            return F.normalize(final_vec, p=2, dim=-1)

        else:
            # 추론 시 마지막 스텝만 사용
            h_seq = self.seq_proj(item_output[:, -1, :])        # [B, D]
            h_feat = self.feat_proj(feat_output[:, -1, :])      # [B, D]
            h_static = self.static_proj(user_profile_vec[:, -1, :])  # [B, D]

            # Dynamic gate (추론 시에도 동일하게 적용)
            gate_input = torch.cat([h_seq, h_feat, h_static], dim=-1)  # [B, D*3]
            gate = torch.softmax(self.fusion_gate(gate_input), dim=-1)  # [B, 3]
            g_seq, g_feat, g_static = gate.unbind(dim=-1)  # 각각 [B]

            final_vec = (
                g_seq.unsqueeze(-1) * h_seq +
                g_feat.unsqueeze(-1) * h_feat +
                g_static.unsqueeze(-1) * h_static
            )  # [B, D]

            final_vec = self.output_proj(final_vec)
            if target_week is not None:
                t_week_intent = target_week_vec[:, -1, :]
                final_vec = final_vec + t_week_intent
            return F.normalize(final_vec, p=2, dim=-1) 

        
import json
import torch


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

import torch
import torch.nn.functional as F



def inbatch_corrected_logq_loss_with_hybrid_hard_neg(
    user_emb, seq_item_emb, target_ids, user_ids, log_q_tensor,
    hn_item_emb=None, batch_hard_neg_ids=None,
    flat_history_item_ids=None,
    step_weights=None,
    final_idx=None, # 💡 [핵심 파라미터] 외부에서 계산된 마지막 스텝의 위치 정보
    temperature=0.1, 
    lambda_logq=1.0,          
    alpha=1.0,                
    margin=0.00,
    soft_penalty_weight=5.0,
    return_metrics=False     
):
    N = user_emb.size(0)
    device = user_emb.device
    SAFE_NEG_INF = -1e9
    
    # 가중치 텐서 준비
    if step_weights is not None:
        step_weights = torch.as_tensor(step_weights, device=device, dtype=torch.float32)
        weight_sum = step_weights.sum() + 1e-9
        sw_unsqueezed = step_weights.unsqueeze(1)
    else:
        weight_sum = None
        sw_unsqueezed = None

    # -----------------------------------------------------------
    # 1. In-batch Logits 계산 (전체 N개 스텝 대상)
    # -----------------------------------------------------------
    sim_matrix = torch.matmul(user_emb, seq_item_emb.T) 
    pos_sim = torch.diagonal(sim_matrix) # [N]
    labels = torch.arange(N, device=device)
    logits = sim_matrix / temperature

    if margin > 0.0:
        logits[labels, labels] -= (margin / temperature)

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # In-batch 내 가짜 네거티브(False Negative) 마스킹
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits = logits.masked_fill(false_neg_mask, SAFE_NEG_INF)
    
    metrics = {}
    
    # -----------------------------------------------------------
    # 2. Hard Negative Processing (오직 final_idx 위치만 연산 및 조립)
    # -----------------------------------------------------------
    num_hn_to_use = 50
    
    if hn_item_emb is not None and batch_hard_neg_ids is not None and final_idx is not None and final_idx.numel() > 0:
        
        num_finals = final_idx.size(0)
        pool_multiplier = 3
        num_pool = num_hn_to_use * pool_multiplier  
        skip_top_k = 10 

        u_emb_f = user_emb[final_idx]                  # [num_finals, D]
        pos_sim_f = pos_sim[final_idx]                 # [num_finals]
        
        hn_emb_f = hn_item_emb                         
        hn_ids_f = batch_hard_neg_ids                  
        
        flat_hist_f = flat_history_item_ids[final_idx] if flat_history_item_ids is not None else None

        # [STEP 1] 기울기(Gradient) 추적 없이 후보군 필터링
        with torch.no_grad():
            hn_emb_no_grad_f = hn_emb_f.detach() 
            hn_sim_no_grad_f = torch.bmm(u_emb_f.unsqueeze(1), hn_emb_no_grad_f.transpose(1, 2)).squeeze(1)
            
            absolute_fn_mask_f = torch.zeros_like(hn_sim_no_grad_f, dtype=torch.bool, device=device)
            if flat_hist_f is not None:
                absolute_fn_mask_f = (hn_ids_f.unsqueeze(2) == flat_hist_f.unsqueeze(1)).any(dim=2)
            
            boundary_ratio = 0.80
            dynamic_boundary_f = pos_sim_f.unsqueeze(1) * boundary_ratio 
            dynamic_fn_mask_f = hn_sim_no_grad_f >= dynamic_boundary_f
            
            final_fn_mask_f = absolute_fn_mask_f | dynamic_fn_mask_f
            masked_sims_f = hn_sim_no_grad_f.masked_fill(final_fn_mask_f, -1e4)
            
            _, top_idx_all_f = torch.topk(masked_sims_f, num_pool + skip_top_k, dim=1)
            top_idx_pool_f = top_idx_all_f[:, skip_top_k:]

            rand_idx_f = torch.rand(num_finals, num_pool, device=device).argsort(dim=1)[:, :num_hn_to_use]
            top_idx_f = torch.gather(top_idx_pool_f, 1, rand_idx_f)

        # [STEP 2] 본 학습용 임베딩 조립 (initial N개)
        batch_hn_ids_final_f = torch.gather(hn_ids_f, 1, top_idx_f)
        top_idx_expanded_f = top_idx_f.unsqueeze(-1).expand(-1, -1, hn_emb_f.size(2))
        hn_emb_final_f = torch.gather(hn_emb_f, 1, top_idx_expanded_f).detach()

        # [STEP 3] 본 연산 (Gradient 흐름)
        hn_sim_f = torch.bmm(u_emb_f.unsqueeze(1), hn_emb_final_f.transpose(1, 2)).squeeze(1) 
        hn_logits_f = (hn_sim_f / temperature) * alpha
        
        if lambda_logq > 0.0:
            hn_log_q_f = log_q_tensor[batch_hn_ids_final_f] 
            hn_logits_f = hn_logits_f - (hn_log_q_f * lambda_logq)
        
        final_safety_mask_f = hn_sim_f >= (pos_sim_f.unsqueeze(1) * boundary_ratio)
        hn_logits_f = hn_logits_f.masked_fill(final_safety_mask_f, SAFE_NEG_INF)

        # =======================================================
        # 💡 [핵심 결합 로직] 전체 N 차원 캔버스에 마지막 스텝(final_idx) 퍼즐 조각 끼워넣기
        # =======================================================
        hn_logits_full = torch.full((N, num_hn_to_use), SAFE_NEG_INF, device=device)
        hn_logits_full[final_idx] = hn_logits_f
        
        logits = torch.cat([logits, hn_logits_full], dim=1)
        
        if return_metrics:
            metrics['hn/discarded_ratio'] = dynamic_fn_mask_f.float().mean().item()
            valid_hn_sim_f = hn_sim_f[~final_safety_mask_f]
            metrics['sim/hn_true_hard'] = valid_hn_sim_f.mean().item() if valid_hn_sim_f.numel() > 0 else 0.0
            
    else: 
        # HNM이 비활성화되었거나 final_idx가 없을 때 (행렬 차원 유지를 위한 더미 결합)
        hn_logits_full = torch.full((N, num_hn_to_use), SAFE_NEG_INF, device=device)
        logits = torch.cat([logits, hn_logits_full], dim=1)

    logits = torch.clamp(logits, min=SAFE_NEG_INF, max=1e4)

    # -----------------------------------------------------------
    # 3. Loss 계산 (Cross Entropy)
    # -----------------------------------------------------------
    if step_weights is not None:
        loss_unreduced = F.cross_entropy(logits, labels, reduction='none')
        loss = (loss_unreduced * step_weights).sum() / weight_sum
        if return_metrics:
            metrics['sim/pos'] = ((pos_sim * step_weights).sum() / weight_sum).item()
    else:
        loss = F.cross_entropy(logits, labels)
        if return_metrics:
            metrics['sim/pos'] = pos_sim.mean().item()

    # -----------------------------------------------------------
    # 4. Probabilities Metrics
    # -----------------------------------------------------------
    if return_metrics:
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            if hn_item_emb is not None and batch_hard_neg_ids is not None and final_idx is not None and final_idx.numel() > 0:
                hn_probs_sum = probs[:, N:].sum(dim=1) 
                neg_probs_total = 1.0 - probs.diagonal() 
                relative_hn_ratio = hn_probs_sum / (neg_probs_total + 1e-9) 
                
                if step_weights is not None:
                    metrics['hn/influence_ratio'] = ((hn_probs_sum * step_weights).sum() / weight_sum).item()
                    metrics['hn/relative_influence'] = ((relative_hn_ratio * step_weights).sum() / weight_sum).item()
                else:
                    metrics['hn/influence_ratio'] = hn_probs_sum.mean().item()
                    metrics['hn/relative_influence'] = relative_hn_ratio.mean().item()

        return loss, metrics

    return loss
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
    stage_container()
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
        
        
        stage_container()
    
    except Exception as e:
        print(f"⚠️ [Warning] Failed to load Pretrained files: {e}. Using random init.")
        
    return aligned_weight

def create_dataloaders(processor, cfg: PipelineConfig, global_now_str, aligned_pretrained_vecs=None, is_train=True):
    """Dataset 및 DataLoader 인스턴스화"""
    mode_str = "Train" if is_train else "Validation"
    print(f"\n📦 [Phase 3-2] Creating {mode_str} DataLoaders...")
    
    # 💡 1. is_train 파라미터 전달
    dataset = SASRecDataset_v3_obsolete(processor, global_now_str = global_now_str, max_len=cfg.max_len, is_train=is_train)
    
    # Dataset 인스턴스에 정렬된 pretrained vector 룩업 테이블 주입
    dataset.pretrained_lookup = aligned_pretrained_vecs 
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        # 💡 2. 검증 시에는 셔플을 끄고, 자투리 데이터(마지막 배치)도 버리지 않고 모두 평가
        shuffle=is_train, 
        num_workers=0,
        pin_memory=True,
        #persistent_workers=True,
        drop_last=is_train 
    )
    
    print(f"✅ {mode_str} Loader Ready: {len(loader)} batches/epoch")
    stage_container()
    
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
    stage_container()
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
        stage_container()
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
    stage_container()
    return user_tower, item_tower


def dataset_peek_v3(dataset, processor):
    """동적 Offset 및 AS-OF 시퀀스 정합성 정밀 검수"""
    print("\n🔍 [Data Peek V2] Verifying Leak-Free Feature Integrity...")
    
    # 첫 번째 유저 샘플 (보통 데이터가 많은 유저를 보기 위해 dataset[0] 사용)
    sample = dataset[0]
    
    # 텐서 이동 (CPU)
    ids = sample['item_ids'].cpu().numpy()
    targets = sample['target_ids'].cpu().numpy()
    offsets = sample['recency_offset'].cpu().numpy()
    weeks = sample['current_week'].cpu().numpy()
    
    # 1. 시퀀스 Shift 및 데이터 마스킹 확인
    # 0(패딩)이 아닌 첫 데이터 위치 확인
    valid_mask = ids != 0
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 1:
        idx = valid_indices[0]
        print(f"✅ [Shift Check]")
        print(f"   - Input[t]:  {ids[idx]:>8} | Input[t+1]: {ids[idx+1]:>8}")
        print(f"   - Target[t]: {targets[idx]:>8} | Target[t+1]: {targets[idx+1]:>8}")
        if ids[idx+1] == targets[idx]:
            print("   👉 Result: Shift Logic is Correct.")
            
    # 2. 💡 동적 Recency Offset & Week 정합성 확인
    print(f"\n✅ [Time Feature Check]")
    if len(valid_indices) > 1:
        # 마지막 유효 인덱스 (Train 시점의 Target_Now 기준점 근처)
        last_idx = valid_indices[-1]
        
        print(f"   - Target Now 시점 기준 (Train Mode: {dataset.is_train})")
        print(f"   - Last Item Offset: {offsets[last_idx]} days (0에 가까울수록 정상)")
        print(f"   - Prev Item Offset: {offsets[last_idx-1]} days")
        
        # 주차 정보 (절대 계절성)
        print(f"   - Sequence Weeks:   {weeks[valid_indices][:5].tolist()} ...")
        
        # 검증: Offset은 시간 역순이어야 함 (과거일수록 숫자가 커야 함)
        if offsets[valid_indices[0]] >= offsets[valid_indices[-1]]:
            print("   👉 Result: Dynamic Offset Direction is Correct (Past has larger offset).")
        else:
            print("   👉 Result: ❌ Offset Error! Future items have larger offset.")

    # 3. 📊 AS-OF Dynamic Features (Continuous/Bucket) 확인
    print(f"\n✅ [AS-OF Sequence Check]")
    # 예시로 가격 버킷과 누적 횟수 확인
    price_b = sample['price_bucket'].cpu().numpy()
    cnt_b = sample['cnt_bucket'].cpu().numpy()
    
    # 동일 세션 내 값이 유지되는지 확인 (앞선 2~3개 샘플링)
    print(f"   - Price Buckets: {price_b[valid_indices][:5].tolist()}")
    print(f"   - Total Cnt Buckets: {cnt_b[valid_indices][:5].tolist()}")
    
    # 4. 👤 Static Metadata (Batch 단위 고정값) 확인
    print(f"\n✅ [Static Metadata Check]")
    print(f"   - Age Bucket: {sample['age_bucket'].item()}")
    print(f"   - Club Status: {sample['club_status_ids'].item()}")
    
    print("\n🚀 [Final Verdict]: If 'Last Item Offset' is near 0 and 'Past' is larger, your Leak-Free Logic is READY!")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """최소 학습률 하한선(min_lr_ratio)이 보장되는 코사인 스케줄러"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 0.0으로 떨어지지 않고 초기 LR의 1% 수준(min_lr_ratio)으로 수렴
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Two-Tower 구조를 위해 Best 상태만 판별해주는 조기 종료 도우미"""
    def __init__(self, patience=7, mode='max'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        self.is_best = False

    def __call__(self, val_score):
        score = val_score if self.mode == 'max' else -val_score
        self.is_best = False

        if self.best_score is None:
            self.best_score = score
            self.is_best = True
        elif score <= self.best_score:
            self.counter += 1
            print(f"⚠️ EarlyStopping 카운트: {self.counter} / {self.patience} (Current Best: {self.best_score if self.mode == 'max' else -self.best_score:.2f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.is_best = True
            self.counter = 0
            
            
            
            
            
            
            
            
            
            
            



def evaluate_model(model, item_tower, dataloader, target_df_path, device, processor, k_list=[10, 20, 200,500]):
    """
    Validation 데이터셋과 정답지(target_dict)를 이용해 Recall@K를 평가하는 함수
    """
    model.eval()
    item_tower.eval()
    print(f"🎯 Loading targets from: {target_df_path}")
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    del target_df
    # K값 중 가장 큰 값을 기준으로 한 번만 Top-K 연산을 수행하여 GPU 연산 절약
    max_k = max(k_list)
    
    total_hits = {k: 0.0 for k in k_list}
    total_valid_users = 0
    
    with torch.no_grad():
        # 1. 전체 아이템 임베딩 로드 및 정규화 (루프 밖에서 한 번만 수행)
        full_item_embeddings = item_tower.get_all_embeddings()
        norm_item_embeddings = F.normalize(full_item_embeddings, p=2, dim=1)
        
        '''
        print("\n🔍 [Eval Monitor] Item Tower Check")
        print(f"   - Shape: {full_item_embeddings.shape}")
        print(f"   - Mean: {full_item_embeddings.mean().item():.6f} | Std: {full_item_embeddings.std().item():.6f}")
            # 인덱스 1번(첫 번째 실제 아이템)의 앞 5개 차원 값 출력
        if full_item_embeddings.size(0) > 1:
            print(f"   - Item [1] Sample: {full_item_embeddings[1][:5].tolist()}")
        '''
        
        
        
        
        # tqdm을 이용해 진행 시간 및 상태 표시
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Dataloader에서 'user_ids'를 문자열 리스트로 바로 가져옴
            user_ids = batch['user_ids'] 
            
            item_ids = batch['item_ids'].to(device, non_blocking=True)
            padding_mask = batch['padding_mask'].to(device, non_blocking=True)
            time_bucket_ids = batch['time_bucket_ids'].to(device, non_blocking=True)
            session_ids = batch['session_ids'].to(device, non_blocking=True) # 💡 [신규 언패킹]
            type_ids = batch['type_ids'].to(device, non_blocking=True)
            color_ids = batch['color_ids'].to(device, non_blocking=True)
            graphic_ids = batch['graphic_ids'].to(device, non_blocking=True)
            section_ids = batch['section_ids'].to(device, non_blocking=True)
            age_bucket = batch['age_bucket'].to(device, non_blocking=True)
            price_bucket = batch['price_bucket'].to(device, non_blocking=True)
            cnt_bucket = batch['cnt_bucket'].to(device, non_blocking=True)
            recency_bucket = batch['recency_bucket'].to(device, non_blocking=True)
            channel_ids = batch['channel_ids'].to(device, non_blocking=True)
            club_status_ids = batch['club_status_ids'].to(device, non_blocking=True)
            news_freq_ids = batch['news_freq_ids'].to(device, non_blocking=True)
            fn_ids = batch['fn_ids'].to(device, non_blocking=True)
            active_ids = batch['active_ids'].to(device, non_blocking=True)
            cont_feats = batch['cont_feats'].to(device, non_blocking=True)        
            recency_offset = batch['recency_offset'].to(device, non_blocking=True)
            current_week = batch['current_week'].to(device, non_blocking=True)
            target_week = batch['target_week'].to(device)
            # Pretrained Vector 룩업 처리
            if 'pretrained_vecs' in batch:
                pretrained_vecs = batch['pretrained_vecs'].to(device, non_blocking=True)
                print("pretrained_vecs has been loaded")
            else:
                pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device)
            
            # =======================================================
            '''
            if batch_idx == 0:
                print(f"\n🔍 [Eval Monitor] Pretrained Vecs Check (Batch 0)")
                print(f"   - Shape: {pretrained_vecs.shape}")
                print(f"   - Mean: {pretrained_vecs.mean().item():.6f} | Std: {pretrained_vecs.std().item():.6f}")
                
                # 패딩(0)이 아닌 실제 아이템 ID 하나를 찾아 해당 벡터의 값 확인
                valid_mask = item_ids[0] != 0
                if valid_mask.any():
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0][0]
                    sample_item_id = item_ids[0][valid_idx].item()
                    print(f"   - Item [{sample_item_id}] Sample: {pretrained_vecs[0][valid_idx][:5].tolist()}")
            '''
            forward_kwargs = {
                'pretrained_vecs': pretrained_vecs,
                'item_ids': item_ids,
                'time_bucket_ids': time_bucket_ids,
                'type_ids': type_ids,
                'color_ids': color_ids,
                'graphic_ids': graphic_ids,
                'section_ids': section_ids,
                'age_bucket': age_bucket,
                'price_bucket': price_bucket,
                'cnt_bucket': cnt_bucket,
                'recency_bucket': recency_bucket,
                'channel_ids': channel_ids,
                'club_status_ids': club_status_ids,
                'news_freq_ids': news_freq_ids,
                'fn_ids': fn_ids,
                'active_ids': active_ids,
                'cont_feats': cont_feats,
                'recency_offset': recency_offset, 'current_week': current_week, 'target_week': target_week,
                'padding_mask': padding_mask,
                'training_mode': False # Dropout 비활성화
            }

            # 2. User Tower Forward
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = model(**forward_kwargs) # (Batch, Seq_Len, Dim)
                
            # 3. 실제 마지막 유효 시점의 벡터 추출 (단순히 -1이 아니라 Padding을 고려)
            if output.dim() == 3:
                lengths = (~padding_mask).sum(dim=1)
                last_indices = (lengths - 1).clamp(min=0)
                batch_range = torch.arange(output.size(0), device=device)
                last_user_emb = output[batch_range, last_indices]
            else:
                last_user_emb = output
                
            # L2 정규화
            last_user_emb = F.normalize(last_user_emb, p=2, dim=1)
            
            # 4. 정답지(target_dict)에 존재하는 유효한 유저만 필터링
            valid_idx_list = [i for i, uid in enumerate(user_ids) if uid in target_dict and len(target_dict[uid]) > 0]
            if not valid_idx_list: 
                continue 
                
            v_idx = torch.tensor(valid_idx_list, device=device)
            valid_user_emb = last_user_emb[v_idx]
            
            # 5. 전체 아이템과 내적하여 Score 계산
            scores = torch.matmul(valid_user_emb, norm_item_embeddings.T)
            
            # 6. Top-K 인덱스 추출
            _, topk_indices = torch.topk(scores, k=max_k, dim=-1)
            pred_ids = topk_indices.cpu().numpy() 
            
            # 7. 실제 정답(Set)과 교집합 비교하여 Recall@K 측정
            for i, original_idx in enumerate(valid_idx_list):
                u_id = user_ids[original_idx]
                
                # 💡 [안전 장치] 정답이 단일 문자열이든 리스트든 무조건 리스트로 취급하게 만듦
                raw_targets = target_dict[u_id]
                if isinstance(raw_targets, str) or not hasattr(raw_targets, '__iter__'):
                    raw_targets = [raw_targets]
                
                # 💡 리스트로 만들어진 raw_targets를 순회
                actual_indices = set(processor.item2id[iid] for iid in raw_targets if iid in processor.item2id)
                
                # 만약 정답 아이템들이 모델이 모르는(OOT/Unseen) 아이템이라 매핑 후 세트가 비어있다면, 
                # 맞출 가능성이 0이므로 평가 타겟 유저에서 제외 (분모 증가 방지)
                if not actual_indices:
                    continue
                
                total_valid_users += 1
                for k in k_list:
                    # 1. Top-K 예측 리스트를 Set으로 변환
                    pred_set = set(pred_ids[i, :k])
                    
                    # 2. 교집합(맞춘 아이템들) 추출
                    hit_items = actual_indices.intersection(pred_set)
                    
                    # 3. (맞춘 개수 / 실제 구매한 전체 개수) 비율 계산
                    user_recall = len(hit_items) / len(actual_indices)
                    
                    # 4. 해당 비율을 누적
                    total_hits[k] += user_recall
            
            
            if 'scores' in locals():
                del scores, topk_indices
            
            # 2. Forward 연산 결과 텐서 해제
            del output, last_user_emb
            if 'valid_user_emb' in locals():
                del valid_user_emb
                
            # 3. 입력 피처 및 kwargs 일괄 해제
            del item_ids, padding_mask, time_bucket_ids, pretrained_vecs
            del type_ids, color_ids, graphic_ids, section_ids
            del age_bucket, price_bucket, cnt_bucket, recency_bucket
            del channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids, cont_feats
            del recency_offset, current_week
            del forward_kwargs

    # 최종 Recall 퍼센티지 계산
    results = {}
    if total_valid_users > 0:
        for k in k_list:
            results[f'Recall@{k}'] = (total_hits[k] / total_valid_users) * 100
            
    print(f"\n📈 [Validation Results] Valid Users: {total_valid_users}")
    for k in k_list:
        print(f"   - Recall@{k:03d}: {results.get(f'Recall@{k}', 0):.2f}%")
        
    del full_item_embeddings, norm_item_embeddings
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return results



def train_user_tower_cl_enhance(epoch, model, item_tower,norm_item_embeddings, log_q_tensor, dataloader, optimizer, scaler, cfg, device, hard_neg_pool_tensor, scheduler, hn_scheduler ,seq_labels=None, static_labels=None):
    """단일 에포크 훈련 함수 (All Time Steps + Same-User Masking 적용) + Gradient Accumulation"""
    model.train()
    total_loss_accum = 0.0
    main_loss_accum = 0.0
    cl_loss_accum = 0.0
    epoch_danger_ratio_sum = 0.0
    num_batches = 0
    
    # 💡 [추가] 스케줄러 계산을 위한 누적 변수 초기화
    epoch_sim_pos_sum = 0.0
    epoch_sim_hn_sum = 0.0
    epoch_penalized_ratio_sum = 0.0
    num_hn_batches = 0

    # 💡 [핵심 1] 누적 스텝 설정 (384 * 2 = 768)
    accumulation_steps = 1
    force_mining_next_epoch = False
    
    seq_labels = seq_labels or []
    static_labels = static_labels or []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    # 💡 [핵심 2] 루프 시작 전, 혹시 남아있을지 모르는 이전 에포크의 그래디언트 초기화
    #optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(pbar):

        # -------------------------------------------------------
        # 1. Data Unpacking (Double-shuffled Views 적용)
        # -------------------------------------------------------
        # 🏷️ [공통 피처] 정적(Static) 속성: v1, v2에 똑같이 들어갑니다.
        item_ids = batch['item_ids'].to(device, non_blocking=True)
        target_ids = batch['target_ids'].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(device, non_blocking=True)
        time_bucket_ids = batch['time_bucket_ids'].to(device, non_blocking=True)
        session_ids = batch['session_ids'].to(device, non_blocking=True)
        type_ids = batch['type_ids'].to(device, non_blocking=True)
        color_ids = batch['color_ids'].to(device, non_blocking=True)
        graphic_ids = batch['graphic_ids'].to(device, non_blocking=True)
        section_ids = batch['section_ids'].to(device, non_blocking=True)
        
        age_bucket = batch['age_bucket'].to(device, non_blocking=True)
        price_bucket = batch['price_bucket'].to(device, non_blocking=True)
        cnt_bucket = batch['cnt_bucket'].to(device, non_blocking=True)
        recency_bucket = batch['recency_bucket'].to(device, non_blocking=True)
        
        channel_ids = batch['channel_ids'].to(device, non_blocking=True)
        club_status_ids = batch['club_status_ids'].to(device, non_blocking=True)
        news_freq_ids = batch['news_freq_ids'].to(device, non_blocking=True)
        fn_ids = batch['fn_ids'].to(device, non_blocking=True)
        active_ids = batch['active_ids'].to(device, non_blocking=True)
        cont_feats = batch['cont_feats'].to(device, non_blocking=True)
        
        recency_offset = batch['recency_offset'].to(device, non_blocking=True)
        current_week = batch['current_week'].to(device, non_blocking=True)
        target_week = batch['target_week'].to(device, non_blocking=True)
        
        # 이건 안들어가 모
        interaction_dates = batch['interaction_dates'].to(device, non_blocking=True)
        
        if 'pretrained_vecs' in batch:
            pretrained_vecs = batch['pretrained_vecs'].to(device, non_blocking=True)
        else:
            pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device, non_blocking=True)
            
        forward_kwargs_v1 = {
            'pretrained_vecs': pretrained_vecs,
            'item_ids': item_ids,
            'time_bucket_ids': time_bucket_ids,
            'type_ids': type_ids, 'color_ids': color_ids,
            'graphic_ids': graphic_ids, 'section_ids': section_ids,
            'age_bucket': age_bucket, 'price_bucket': price_bucket,
            'cnt_bucket': cnt_bucket, 'recency_bucket': recency_bucket,
            'channel_ids': channel_ids, 'club_status_ids': club_status_ids,
            'news_freq_ids': news_freq_ids, 'fn_ids': fn_ids,
            'active_ids': active_ids, 'cont_feats': cont_feats,
            'recency_offset': recency_offset, 'current_week': current_week, 'target_week': target_week,
            'padding_mask': padding_mask,
            'training_mode': True
        }
        import faiss
        if not getattr(model, 'prototypes_initialized', False):
            print("\n🌟 [CL Init] Faiss K-Means를 이용해 최적의 256개 초기 중심점을 계산합니다...")
            with torch.no_grad():
                # 1. 학습된 전체 아이템 임베딩을 numpy 배열로 가져옴
                item_embs = item_tower.item_matrix.weight.detach().float().cpu().numpy()
                
                # Faiss를 위한 L2 정규화 (Cosine 유사도 기반 군집화를 위해 필수)
                faiss.normalize_L2(item_embs)
                
                # 2. Faiss K-Means 설정 및 훈련 (256개 군집, 20번 반복 연산)
                d = item_embs.shape[1] # 임베딩 차원 (예: 64)
                kmeans = faiss.Kmeans(d=d, k=model.num_intents, niter=20, verbose=True, spherical=True)
                
                # 단 몇 초면 5~6만 개 아이템을 256개의 완벽한 군집으로 나눔
                kmeans.train(item_embs)
                
                # 3. 계산된 256개의 무게중심(Centroids)을 텐서로 변환
                optimal_centroids = torch.tensor(kmeans.centroids).to(device)
                
                # 4. 모델의 학습 가능한 파라미터(Intent Prototypes)에 덮어쓰기!
                model.intent_prototypes.copy_(optimal_centroids)
                model.prototypes_initialized = True
                
                print("✅ [CL Init] 완벽하게 분산된 256개의 패션 의도 기둥이 세워졌습니다!")
        # =======================================================
        # 🕵️‍♂️ [Data Peek] 첫 번째 에포크, 첫 번째 배치에서 데이터 캡처!
        # =======================================================
        if epoch == 1 and batch_idx == 0:
            print("\n" + "="*70)
            print("🕵️‍♂️ [Tensor Peek] First Batch, First User Verification")
            print("="*70)
            
            u_idx = 0 # 배치의 첫 번째 유저
            seq_len = item_ids.shape[1]
            valid_len = (~padding_mask[u_idx]).sum().item()
            
            print(f"✅ User Index in Batch: {u_idx} | Valid Length: {valid_len} / {seq_len}")
            
            # 1. 1명의 유저 시퀀스 데이터 확인 (패딩 포함 전체 리스트 출력)
            print("\n[1. Sequence Alignment Check (Left Padding Expected)]")
            print(f"📦 item_ids      : {item_ids[u_idx].tolist()}")
            print(f"🎯 target_ids    : {target_ids[u_idx].tolist()}")
            print(f"⏳ time_buckets  : {time_bucket_ids[u_idx].tolist()}")
            print(f"📆 recency_offset: {recency_offset[u_idx].tolist()}")
            print(f"💰 price_bucket  : {price_bucket[u_idx].tolist()}")
            
            # 2. forward_kwargs 형태(Shape) 확인
            print("\n[2. forward_kwargs Shape Check]")
            for k, v in forward_kwargs_v1.items():
                if isinstance(v, torch.Tensor):
                    # 텐서인 경우 차원(Shape) 출력
                    print(f" - {k:<15}: {list(v.shape)}")
                else:
                    # 텐서가 아닌 경우 (예: training_mode) 값 출력
                    print(f" - {k:<15}: {v}")
            print("="*70 + "\n")

        # -------------------------------------------------------
        # 2. Forward & Loss Calculation (AMP)
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            # 두 뷰에 대해 각각 모델 통과
            output_1 = model(**forward_kwargs_v1)
            #output_2 = model(**forward_kwargs_v1)
            
            # 💡 메인 타스크용 마스크 및 정보는 모두 'View 1' 기준
            valid_mask = ~padding_mask
            batch_size, seq_len = item_ids.shape
            
            # [Real Time-Decay Weighting]
            max_dates = interaction_dates.masked_fill(padding_mask, -1).max(dim=1, keepdim=True)[0]            
            delta_t = (max_dates - interaction_dates).float()
            min_weight, half_life = 0.2, 21.0
            import math
            decay_rate = math.log(2) / half_life
            seq_weights = min_weight + (1.0 - min_weight) * torch.exp(-decay_rate * delta_t)
            flat_weights = seq_weights[valid_mask] 
            
            # [Last Indices Masking] - v1과 v2 각각의 마지막 스텝 위치 계산
            seq_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            batch_range = torch.arange(batch_size, device=device)
            
            last_indices = torch.max(seq_positions.masked_fill(~valid_mask, -1), dim=1)[0].clamp(min=0)
            is_last_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            is_last_mask[batch_range, last_indices] = True

          
            # =======================================================
            # [1] Main Loss 계산 (View 1 타겟 및 임베딩만 사용)
            # =======================================================
            flat_output = output_1[valid_mask] 
            flat_targets = target_ids[valid_mask]
            
            batch_row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
            flat_user_ids = batch_row_indices[valid_mask] 
            flat_is_last = is_last_mask[valid_mask]
            MAX_FLAT_SIZE = 12000
            
            if flat_output.size(0) > MAX_FLAT_SIZE:
                print(f"⚠️ [Memory Protection] Batch elements ({flat_output.size(0)}) > {MAX_FLAT_SIZE}. Truncating oldest steps...")
                
                # 💡 핵심 로직: flat_weights가 클수록 최신 데이터임.
                # torch.topk를 사용해 가중치가 가장 높은 상위 MAX_FLAT_SIZE개의 인덱스만 추출 (정렬 연산 최소화로 매우 빠름)
                _, recent_idx = torch.topk(flat_weights, k=MAX_FLAT_SIZE)
                
                # 추출된 최신 인덱스로 텐서 덮어씌우기
                flat_output = flat_output[recent_idx]
                flat_targets = flat_targets[recent_idx]
                flat_user_ids = flat_user_ids[recent_idx]
                flat_weights = flat_weights[recent_idx]
                flat_is_last = flat_is_last[recent_idx]
            
            if flat_output.size(0) > 0:
                flat_user_emb = F.normalize(flat_output, p=2, dim=1)
                flat_history_item_ids = item_ids[flat_user_ids] 
                
                # 💡 grad 전파
                batch_seq_item_emb = item_tower.item_matrix.weight[flat_targets]
                batch_seq_item_emb = F.normalize(batch_seq_item_emb, p=2, dim=1)

                # =======================================================
                # 💡 [핵심 2] 하드 네거티브 VRAM 최적화 (Early Slicing)
                # =======================================================
                batch_hn_item_emb = None
                batch_hard_neg_ids = None
                final_idx = None # Loss 함수에서 조립할 위치 인덱스
                
                if hard_neg_pool_tensor is not None:
                    # flat된 데이터 중 진짜 마지막 스텝(True)의 인덱스만 추출 [num_finals]
                    
                    HNM_WEIGHT_THRESHOLD = 0.6
                    hnm_eligible = flat_weights >= HNM_WEIGHT_THRESHOLD
                    # 단, 마지막 스텝은 threshold 무관하게 항상 포함
                    hnm_eligible = hnm_eligible | flat_is_last
                    
                    
                    final_idx = hnm_eligible.nonzero(as_tuple=True)[0]
                    
                    if final_idx.numel() > 0:
                        # 오직 마지막 스텝에 해당하는 타겟 ID만 가져옵니다.
                        final_targets = flat_targets[final_idx]
                        
                        # [num_finals, 150] 및 [num_finals, 150, D] 크기로 최소한의 메모리만 사용!
                        batch_hard_neg_ids = hard_neg_pool_tensor[final_targets] 
                        batch_hn_item_emb = norm_item_embeddings[batch_hard_neg_ids]
                
                main_loss, b_metrics = inbatch_corrected_logq_loss_with_hybrid_hard_neg(
                    user_emb=flat_user_emb, seq_item_emb=batch_seq_item_emb,
                    target_ids=flat_targets, user_ids=flat_user_ids,
                    log_q_tensor=log_q_tensor, 
                    hn_item_emb=batch_hn_item_emb, batch_hard_neg_ids=batch_hard_neg_ids, 
                    flat_history_item_ids=flat_history_item_ids, step_weights=flat_weights,
                    final_idx=final_idx,  
                    temperature=0.07, lambda_logq=cfg.lambda_logq, alpha=1.0,
                    soft_penalty_weight=cfg.soft_penalty_weigh, margin=0.0, return_metrics=True
                )

            else:
                main_loss = torch.tensor(0.0, device=device)
            
            # =======================================================
            # [2] Semantic Contrastive Loss 계산
            # =======================================================

            cl_loss = torch.tensor(0.0, device=device)
            current_cl_weight = 0.0
            
            # 💡 View 1의 마지막 스텝(유저의 최종 의도)만 추출
            last_z1 = output_1[is_last_mask] # [N, D]
            
            if last_z1.size(0) > 1:
                # 1. 임베딩 정규화
                z = F.normalize(last_z1, p=2, dim=1) # 유저 [N, D]
                c = F.normalize(model.intent_prototypes, p=2, dim=1) # 중심점 [K, D]
                
                # 2. 유저와 모든 의도 클러스터 간의 유사도 (Logits)
                proto_logits = torch.matmul(z, c.T) # [N, K]
                
                with torch.no_grad():
                    # 3. Soft Assignment: 가장 유사도가 높은 클러스터를 정답으로 간주
                    target_clusters = torch.argmax(proto_logits, dim=1)
                
                # 4. 온도(Temperature) 스케일링으로 날카로운 군집화 유도
                tau_proto = 0.1 
                scaled_logits = proto_logits / tau_proto
                
                # 5. InfoNCE Loss (내 클러스터와는 당기고, 나머지 255개 클러스터와는 밀어냄!)
                # 💡 이 Push 효과 덕분에 공간이 쪼그라드는 Dimensional Collapse가 완벽히 방지됩니다.
                cl_loss_raw = F.cross_entropy(scaled_logits, target_clusters)
                
                # 6. CL 가중치 웜업 (공간 적응기)
                # 처음부터 강하게 당기지 않고, 중심점들이 유저 분포에 자연스럽게 스며들 시간을 줍니다.
                CL_WARMUP_EPOCHS = 3
                MAX_CL_WEIGHT = 0.05  # 보조 Loss이므로 메인 Loss를 방해하지 않는 선 유지
                
                if epoch <= CL_WARMUP_EPOCHS:
                    current_cl_weight = MAX_CL_WEIGHT * (epoch / CL_WARMUP_EPOCHS)
                else:
                    current_cl_weight = MAX_CL_WEIGHT
                    
                cl_loss = cl_loss_raw * current_cl_weight
                
                # 📊 모니터링 메트릭 로깅
                with torch.no_grad():
                    unique_clusters = torch.unique(target_clusters).numel()
                    b_metrics['cl/active_clusters'] = unique_clusters
                    b_metrics['cl/active_ratio'] = unique_clusters / model.num_intents
                    b_metrics['cl/proto_loss'] = cl_loss_raw.item()
                    b_metrics['cl/weight'] = current_cl_weight

            # 최종 Loss 결합
            total_loss = main_loss + cl_loss
            scaled_loss = total_loss / accumulation_steps
        
            

        # -------------------------------------------------------
        # 3 & 4. Backward & Optimizer Step (기존과 완전 동일)
        # -------------------------------------------------------
        scaler.scale(scaled_loss).backward()
        if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        # -------------------------------------------------------
        # 5. Logging & Memory Cleanup
        # -------------------------------------------------------
        total_loss_accum += total_loss.item()
        main_loss_accum += main_loss.item()
        cl_loss_accum += cl_loss.item() # 💡 에포크 종료 후 평균 CL 계산을 위해 추가
        
        pbar.set_postfix({
            'Main_Loss': f"{main_loss.item():.4f}", # 💡 Postfix에는 순수 main_loss 표기 권장
            'CL_Loss': f"{cl_loss.item():.4f}"
        })
        # 100번 단위 로깅도 미니배치(384) 기준 100번이 됩니다.
        if batch_idx % 100 == 0:
            wandb_log_dict = {
                "Train/Main_Loss": main_loss.item(),
                "Train/CL_Loss": cl_loss.item()
                }
            for k in ['sim/pos',  'sim/hn_all', 'sim/hn', 'hn/influence_ratio', 
                    'hn/penalized_ratio', 'sim/hn_penalized', 'sim/hn_true_hard','hn/discarded_ratio',
                      'hn/relative_influence', 'hn/danger_zone_ratio']:
                if k in b_metrics:
                    log_key = k if '/' in k else f"Train/{k}"
                    if k == 'sim/hn': log_key = "Train/sim:hn_candidate_mean"
                    if k == 'sim/hn_all': log_key = "Train/sim:hn_all_mean"
                    wandb_log_dict[log_key] = b_metrics[k]

            if 'sim/peer_top1_mean' in b_metrics:
                wandb_log_dict.update({
                    "Peer_Sim/Top1_Mean": b_metrics['sim/peer_top1_mean'],
                    "Peer_Sim/Top1_Max": b_metrics['sim/peer_top1_max'],
                    "Peer_Sim/Top1_Min": b_metrics['sim/peer_top1_min']
                })
            if 'sim/peer_top1_raw' in b_metrics:
                wandb_log_dict["Peer_Sim/Top1_Distribution"] = wandb.Histogram(b_metrics['sim/peer_top1_raw'].numpy())

            # 💡 [신규] DuoRec 지표 로깅 추가
            if 'cl/has_peer_ratio' in b_metrics:
                wandb_log_dict["CL_Stats/Has_Peer_Ratio"] = b_metrics['cl/has_peer_ratio']
                wandb_log_dict["CL_Stats/Avg_Peers_Per_Seq"] = b_metrics['cl/avg_peers']
            

            for k in ['cl/semantic_gap', 'cl/peer_user_sim','cl/proto_loss', 'cl/avg_peers', 'cl/target_margin','cl/active_ratio','cl/negative_user_sim', 'cl/active_clusters']:
                if k in b_metrics:
                    log_key = k
                    wandb_log_dict[log_key] = b_metrics[k]
            wandb.log(wandb_log_dict)       
        
        # mini-batch deloc
        del output_1, flat_output, flat_targets, flat_user_ids
        del main_loss, scaled_loss, total_loss
        
        # View 1 텐서 삭제
        del item_ids, target_ids, padding_mask, time_bucket_ids, pretrained_vecs
        del type_ids, color_ids, graphic_ids, section_ids, session_ids
        del price_bucket, cnt_bucket, recency_bucket, channel_ids, cont_feats
        del recency_offset, current_week, target_week, interaction_dates
        ''' 
        # View 2 텐서 삭제
        del item_ids_v2, target_ids_v2, padding_mask_v2, time_bucket_ids_v2, pretrained_vecs_v2
        del type_ids_v2, color_ids_v2, graphic_ids_v2, section_ids_v2, session_ids_v2
        del price_bucket_v2, cnt_bucket_v2, recency_bucket_v2, channel_ids_v2, cont_feats_v2
        del recency_offset_v2, current_week_v2, target_week_v2, interaction_dates_v2
        '''
        # 공통 정적 텐서 삭제
        del age_bucket, club_status_ids, news_freq_ids, fn_ids, active_ids
        
        if 'flat_user_emb' in locals():
            del flat_user_emb
        
    # 에포크 종료 후 평균 Loss 계산
    avg_loss = total_loss_accum / len(dataloader)
    avg_main = main_loss_accum / len(dataloader)
    avg_cl = cl_loss_accum / len(dataloader) if cl_loss_accum > 0 else 0.0



    avg_danger_ratio = epoch_danger_ratio_sum / num_batches if num_batches > 0 else 0.0
    print(f"📊 Epoch [{epoch}] Avg Danger Zone Ratio: {avg_danger_ratio:.4f}")

    # 1. HNM이 실제로 발동된 배치들의 평균 지표 계산
    if num_hn_batches > 0:
        avg_sim_pos = epoch_sim_pos_sum / num_hn_batches
        avg_sim_hn = epoch_sim_hn_sum / num_hn_batches
        avg_penalized_ratio = epoch_penalized_ratio_sum / num_hn_batches
    else:
        avg_sim_pos, avg_sim_hn, avg_penalized_ratio = 0.0, 0.0, 0.0

    print(f"📊 Epoch [{epoch}] Avg Margin: {(avg_sim_pos - avg_sim_hn):.4f} | Penalized Ratio: {avg_penalized_ratio:.4f}")

    # 2. 💡 [핵심] TrendBasedHNScheduler 호출 부분
    if hn_scheduler is not None and num_hn_batches > 0:
        
        # step 함수에 1 에포크 동안의 평균값 3개를 정확히 매핑하여 전달합니다.
        updated, new_ex_top_k, reason = hn_scheduler.step(
            sim_pos=avg_sim_pos, 
            sim_hn=avg_sim_hn, 
            penalized_ratio=avg_penalized_ratio
        )
        
        # 스케줄러가 위험을 감지하고 확장을 결정했다면 적용
        if updated:
            print(f"🚨 [Scheduler Alert] {reason}")
            print(f"📈 EX-TOP-K Expanded: {cfg.EX_TOP_K} -> {new_ex_top_k}")
            
            cfg.EX_TOP_K = new_ex_top_k
            force_mining_next_epoch = True # 다음 에포크 시작 시 확장된 K를 반영하여 재채굴 지시
    
    # Gate Weights Logging
    with torch.no_grad():
        s_weights = torch.sigmoid(model.seq_gate).cpu().numpy()
        u_weights = torch.sigmoid(model.static_gate).cpu().numpy()
        
        gate_log = {}
        if seq_labels and len(seq_labels) == len(s_weights):
            gate_log.update({f"Gate/Seq_{label}": w for label, w in zip(seq_labels, s_weights)})
        if static_labels and len(static_labels) == len(u_weights):
            gate_log.update({f"Gate/Static_{label}": w for label, w in zip(static_labels, u_weights)})
            
        if gate_log:
            wandb.log(gate_log)

    print(f"🏁 Epoch {epoch} Completed | Avg Total: {avg_loss:.4f} (Main: {avg_main:.4f}, CL: {avg_cl:.4f})")
    return avg_loss, force_mining_next_epoch

def resume_pipeline_session_weights():
    """
    Epoch 11까지 학습된 베이스라인 모델을 불러와서,
    Session-aware 가중치와 HNM을 결합하여 재학습(Resume)하는 엔트리 포인트
    """
    print("🚀 Starting User Tower Resume Pipeline (Session Weights & HNM)...")
    
    
    SEQ_LABELS = ['item_id', 'recency_curr', 'week_curr', 'item_type', 'target_week']
    STATIC_LABELS = [
        'age', 'price',
        'channel', 'club', 'news', 'fn', 'active',
        'cont_price_std', 'cont_last_diff', 'cont_repurch', 'cont_weekend'
    ] 
    
    # 1. Config & Env
    cfg = PipelineConfig()
    device = setup_environment()
    processor , val_processor, cfg = prepare_features(cfg)
    
   
    
    # -----------------------------------------------------------
    # 💡 하드 네거티브 및 하이퍼파라미터 세팅
    # -----------------------------------------------------------
    cfg.lr = 8e-4               # 기존 학습률 유지
    cfg.epochs = 30         # 이미 11에포크를 돌았으므로, 추가로 30에포크면 충분
    cfg.HN_K = 150  # 20개 추출 후 내부 0.95 제약으로 필터링
    cfg.EX_TOP_K = 0
    cfg.soft_penalty_weigh = 1
    cfg.batch_size = 512
    cfg.hn_scheduled = False
    cfg.dropout = 0.3
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # 2. Data & Metadata 가져오기
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    val_processor.i_side_arr = processor.i_side_arr
    
    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22", aligned_vecs, is_train=False)
    
    #json_path = os.path.join(cfg.base_dir, "filtered_data_reinforced.json")
    #item_category_ids = create_category_mapping_tensor(json_path, processor, device)
    
    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2",
        name=f"Resume_SessionWeight_lr{cfg.lr}_K{cfg.HN_K}", 
        config=cfg.__dict__ 
    )
    
    # -----------------------------------------------------------
    # 3. Models Setup & 💡 Epoch 11 베이스라인 가중치 로드
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, None, log_q_tensor) 
    
    # 💡 [요청 사항 1] 모델 경로 지정 및 로드
    base_user_pth = os.path.join(cfg.model_dir, "best_user_tower_0308_ft.pth")
    base_item_pth = os.path.join(cfg.model_dir, "best_item_tower_0308_ft.pth")
    
    save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_0308_delay_ft.pth")
    save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_0308_delay_ft.pth")

    item_tower.init_from_pretrained(aligned_vecs.to(device))
    
    print(f"📥 Loading Baseline Models from Epoch 11...")
    user_tower.load_state_dict(torch.load(base_user_pth, map_location=device))
    item_tower.load_state_dict(torch.load(base_item_pth, map_location=device))
    print(f"✅ Baseline Models loaded successfully.")
    
    
    print("🔥 Epoch 12 (Resume): Item Tower is UNFROZEN from the start!")
    item_tower.set_freeze_state(False)
    item_finetune_lr = cfg.lr * 0.05
    
    # Optimizer에 두 타워를 동시에 등록 (LR 비대칭)
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': cfg.lr},
        {'params': item_tower.parameters(), 'lr': item_finetune_lr}
    ], weight_decay=cfg.weight_decay, fused=True)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.025) 
    scheduler = get_warmup_hold_decay_schedule(
        optimizer,
        num_warmup_steps = warmup_steps,
        num_training_steps = total_steps,
        hold_ratio=0.7,
        min_lr_ratio=0.01) 
    
    early_stopping = EarlyStopping(patience=7, mode='max')

    # 스케줄러에 w조절 포함, if not chnage

        

    # HNM 동적 스케줄러
    if cfg.hn_scheduled:
        hn_scheduler = BidirectionalHNScheduler(
            initial_ex_top_k=cfg.EX_TOP_K, 
            margin_drop_ratio=0.95, penalty_rise_ratio=1.05,   
            margin_growth_ratio=1.05, penalty_drop_ratio=0.95, 
            window_size=2, step_size=10,
            min_ex_top_k=20, max_ex_top_k=120, cooldown_epochs=1
        )
    else:
        hn_scheduler = None
    
    force_mining_next_epoch = False
    epoch_hn_pool = None
    # -----------------------------------------------------------
    # 4. Training Loop (💡 자연스러운 출력을 위해 에포크 10부터 시작)
    # -----------------------------------------------------------
    start_epoch = 30
    end_epoch = start_epoch + cfg.epochs
    #prev_ex_top_k = cfg.EX_TOP_K 
    for epoch in range(start_epoch, end_epoch):
        
        
        item_tower.eval()
        with torch.no_grad():
            print(f"📦 [Epoch {epoch}] Caching All Item Embeddings for Training...")
            all_item_embs = item_tower.get_all_embeddings()
            norm_item_embeddings = F.normalize(all_item_embs, p=2, dim=1) # [50000, Dim]
            
            # 10에포크부터는 이 캐시된 임베딩을 마이닝에도 재사용하여 속도 극대화
            if epoch >= 10:
                if (epoch - 10) % 2 == 0 or force_mining_next_epoch:
                    print(f"🔍 Mining Global Hard Negatives using cached embeddings...")
                    epoch_hn_pool = mine_global_hard_negatives(
                        norm_item_embeddings, # 다시 계산 안 하고 캐시본 사용
                        exclusion_top_k=0, 
                        mine_k=200, 
                        batch_size=2048, 
                        device=device
                    )
                    force_mining_next_epoch = False
                    #if (hn_scheduler.ex_top_k if hn_scheduler else cfg.EX_TOP_K)<= 80:
                    #    if prev_ex_top_k < (hn_scheduler.ex_top_k if hn_scheduler else cfg.EX_TOP_K):
                    #        cfg.soft_penalty_weigh = cfg.soft_penalty_weigh - 0.1
                    #        print(f" return weight.....")
                    #    else:
                    #        cfg.soft_penalty_weigh = (cfg.EX_TOP_K - hn_scheduler.ex_top_k - 10) * 0.1 + 1.0
                    #        print(f" Add soft_penalty for protect embedding loss explosivee...")
                    #else:
                    #    cfg.soft_penalty_weigh = 1.0

            
                else:
                    print(f"♻️ [Epoch {epoch}] Reusing Previous Hard Negative Pool...")
            
            else:
                # 💡 Epoch 1 ~ 9: 워밍업 기간 (HNM 없음)
                epoch_hn_pool = None
                force_mining_next_epoch = False
                print(f"\n🌱 [Epoch {epoch} Start] Warm-up Phase: Using In-batch Random Negatives (No HNM)")
        item_tower.train()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}/{cfg.epochs}]  - Current LR: {current_lr:.8f}")
        
        # ------------------- 훈련 (Train) -------------------
        avg_loss, force_mining_next_epoch = train_user_tower_cl_enhance(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, 
            norm_item_embeddings=norm_item_embeddings,
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=epoch_hn_pool, # Epoch 1~9는 None, 10부터 텐서 주입
            scheduler=scheduler, 
            hn_scheduler=hn_scheduler,
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        del norm_item_embeddings
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        #prev_ex_top_k = hn_scheduler.ex_top_k if hn_scheduler else cfg.EX_TOP_K
                        
        # ------------------- 평가 (Evaluate) -------------------
        
        val_metrics = evaluate_model(
            model=user_tower, 
            item_tower=item_tower, 
            dataloader=val_loader,
            target_df_path=TARGET_VAL_PATH,
            device=device,
            processor=processor,
            k_list=[10, 20, 100, 500]
        )
        
        current_recall_20 = val_metrics.get('Recall@20', 0.0)
        
        # ------------------- 스케줄러 & Best Model 저장 -------------------
        early_stopping(current_recall_20)
        
        if early_stopping.is_best:
            print(f"🌟 [New Best!] Recall@20 updated: {current_recall_20:.2f}%")
            
        
            torch.save(user_tower.state_dict(), save_user_pth)
            torch.save(item_tower.state_dict(), save_item_pth)
            print(f"   💾 Best model weights saved to: {save_user_pth}")
            
        if early_stopping.early_stop:
            print(f"\n🛑 조기 종료 발동! {early_stopping.patience} Epoch 동안 Recall@20이 개선되지 않았습니다.")
            print(f"🏆 최종 최고 Recall@20: {early_stopping.best_score:.2f}%")
            break
            
    print("\n🎉 Resume Pipeline Execution Finished Successfully!")
    
    
def train_pipeline_from_scratch():
    """
    처음부터 모델을 학습하며, Epoch 1~9는 랜덤 네거티브로 워밍업,
    Epoch 10부터 HNM(Hard Negative Mining)을 도입하는 파이프라인
    """
    print("🚀 Starting User Tower Training Pipeline (From Scratch + Delayed HNM)...")
    
    SEQ_LABELS = ['item_id', 'recency_curr', 'week_curr', 'item_type', 'target_week']
    STATIC_LABELS = [
        'age', 'price',
        'channel', 'club', 'news', 'fn', 'active',
        'cont_price_std', 'cont_last_diff', 'cont_repurch', 'cont_weekend'
    ] 
    
    # 1. Config & Env
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    
    # -----------------------------------------------------------
    # 💡 하이퍼파라미터 세팅
    # -----------------------------------------------------------
    cfg.lr = 2e-3
    cfg.epochs = 40            # 처음부터 학습하므로 넉넉하게 40에포크 설정
    cfg.HN_K = 150             # HNM 발동 시 추출할 풀 사이즈
    cfg.EX_TOP_K = 0
    cfg.soft_penalty_weigh = 1.0
    cfg.hn_scheduled = False
    cfg.batch_size = 512
    cfg.dropout = 0.3
    FREEZE_ITEM_EPOCHS = 3
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # 2. Data & Metadata 가져오기
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    val_processor.i_side_arr = processor.i_side_arr
    
    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22", aligned_vecs, is_train=False)
    
    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2",
        name=f"FromScratch_DelayedHNM_lr{cfg.lr}_K{cfg.HN_K}", 
        config=cfg.__dict__ 
    )
    
    # -----------------------------------------------------------
    # 3. Models Setup (가중치 로드 없이 초기화)
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, None, log_q_tensor) 
    
    save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_from_scratch_local.pth")
    save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_from_scratch_local.pth")
    item_tower.init_from_pretrained(aligned_vecs.to(device))
    print(f"❄️ Item Tower is Frozen for the first {FREEZE_ITEM_EPOCHS} epochs.")
    item_tower.set_freeze_state(True)
    
    if all(x==1 for x in stage_status):
        print(f"all stage ready to use")
    else:
        print(f"⚠️ Warning: Some stages are not ready. Please check the data and configuration.")
    
    # 💡 [참고] 처음부터 학습할 때는 Item Tower의 LR을 낮추지 않고 동일하게 가는 것도 좋습니다.
    # 만약 Pretrained 임베딩이 많이 깨지는 것을 방지하고 싶다면 현재처럼 0.05배율을 유지하세요.
    item_finetune_lr = cfg.lr * 0.05
    
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': cfg.lr},
        {'params': item_tower.parameters(), 'lr': item_finetune_lr}
    ], weight_decay=cfg.weight_decay, fused=True)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.05) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01) 
    early_stopping = EarlyStopping(patience=7, mode='max')

    # HNM 동적 스케줄러
    if cfg.hn_scheduled:
        hn_scheduler = BidirectionalHNScheduler(
            initial_ex_top_k=cfg.EX_TOP_K, 
            margin_drop_ratio=0.95, penalty_rise_ratio=1.05,   
            margin_growth_ratio=1.05, penalty_drop_ratio=0.95, 
            window_size=2, step_size=5,
            min_ex_top_k=20, max_ex_top_k=100, cooldown_epochs=1
        )
    else:
        hn_scheduler = None
    
    force_mining_next_epoch = False
    epoch_hn_pool = None
    
    # -----------------------------------------------------------
    # 4. Training Loop (Epoch 1부터 시작)
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        
        
        if epoch == FREEZE_ITEM_EPOCHS + 1:
            print(f"🔥 Epoch {epoch}: Unfreezing Item Tower for Joint Training!")
            item_tower.set_freeze_state(False)
        
        item_tower.eval()
        with torch.no_grad():
            print(f"📦 [Epoch {epoch}] Caching All Item Embeddings for Training...")
            all_item_embs = item_tower.get_all_embeddings()
            norm_item_embeddings = F.normalize(all_item_embs, p=2, dim=1) # [50000, Dim]
            
            # 10에포크부터는 이 캐시된 임베딩을 마이닝에도 재사용하여 속도 극대화
            if epoch >= 10:
                if (epoch - 10) % 1 == 0 or force_mining_next_epoch:
                    print(f"🔍 Mining Global Hard Negatives using cached embeddings...")
                    epoch_hn_pool = mine_global_hard_negatives(
                        norm_item_embeddings, # 다시 계산 안 하고 캐시본 사용
                        exclusion_top_k=0 ,
                        mine_k=150, 
                        batch_size=2048, 
                        device=device
                    )
                    force_mining_next_epoch = False
                    #if hn_scheduler.ex_top_k <= 40:
                    #    cfg.soft_penalty_weigh = (cfg.EX_TOP_K - hn_scheduler.ex_top_k) * 0.1 + 1.0
                    #    print(f" Add soft_penalty for protect embedding loss explosivee...")
                    #else:
                    #    cfg.soft_penalty_weigh = 1.0

            else:
                # 💡 Epoch 1 ~ 9: 워밍업 기간 (HNM 없음)
                epoch_hn_pool = None
                force_mining_next_epoch = False
                print(f"\n🌱 [Epoch {epoch} Start] Warm-up Phase: Using In-batch Random Negatives (No HNM)")
        
        item_tower.train() # 학습 모드 복구
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}/{cfg.epochs}]  - Current LR: {current_lr:.8f}")
        
        # ------------------- 훈련 (Train) -------------------
        avg_loss, force_mining_next_epoch = train_user_tower_cl_enhance(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, 
            norm_item_embeddings=norm_item_embeddings,
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=epoch_hn_pool, # Epoch 1~9는 None, 10부터 텐서 주입
            scheduler=scheduler, 
            hn_scheduler=hn_scheduler,
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        del norm_item_embeddings
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # ------------------- 평가 (Evaluate) -------------------
        val_metrics = evaluate_model(
            model=user_tower, 
            item_tower=item_tower, 
            dataloader=val_loader,
            target_df_path=TARGET_VAL_PATH,
            device=device,
            processor=processor,
            k_list=[10, 20, 100, 500]
        )
        
        current_recall_20 = val_metrics.get('Recall@20', 0.0)
        
        # ------------------- 스케줄러 & Best Model 저장 -------------------
        early_stopping(current_recall_20)
        
        if early_stopping.is_best:
            print(f"🌟 [New Best!] Recall@20 updated: {current_recall_20:.2f}%")
            torch.save(user_tower.state_dict(), save_user_pth)
            torch.save(item_tower.state_dict(), save_item_pth)
            print(f"   💾 Best model weights saved to: {save_user_pth}")
        if epoch == 9:
            print(f"🔔 [Checkpoint] Epoch {epoch} completed. Current Recall@20: {current_recall_20:.2f}%")
            base_user_pth =os.path.join(cfg.model_dir, "best_user_tower_from_scratch_base_dout.pth")
            base_item_pth= os.path.join(cfg.model_dir, "best_item_tower_from_scratch_base_dout.pth")
            
            torch.save(user_tower.state_dict(), base_user_pth)
            torch.save(item_tower.state_dict(), base_item_pth)
        if early_stopping.early_stop:
            print(f"\n🛑 조기 종료 발동! {early_stopping.patience} Epoch 동안 Recall@20이 개선되지 않았습니다.")
            print(f"🏆 최종 최고 Recall@20: {early_stopping.best_score:.2f}%")
            break
            
    print("\n🎉 From-Scratch Pipeline Execution Finished Successfully!")
    
if __name__ == "__main__":
    # 5에포크까지 학습했으므로 6번부터 재개
    #run_resume_pipeline(resume_epoch=16, last_best_recall=22.60)
    #run_pipeline_opt_v2()
    #import torch.multiprocessing as mp
    #mp.freeze_support()
    
    resume_pipeline_session_weights()
    #run_resume_pipeline_v2()
    #train_pipeline_from_scratch()