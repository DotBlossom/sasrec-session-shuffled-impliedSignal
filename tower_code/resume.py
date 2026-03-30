from dataclasses import dataclass
import gc
import os, sys


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
import pickle
import math
import os
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import wandb

from tower_code.params_config import PipelineConfig
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Garbage Collection

from tower_code.sheduler import AdaptiveHNScheduler, BidirectionalHNScheduler, TrendBasedHNScheduler, get_warmup_hold_decay_schedule



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
        print(f"✅ Loaded {len(self.seqs):,} cleanly pre-filtered users.")

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
                row['price'],     
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
        pad_c = np.zeros((pad_len, 5), dtype=np.float32)
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


class SASRecUserTower_v3_prev(nn.Module):
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
        
        # Item Transformer (2층, 2헤드)
        item_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model , nhead=args.nhead , dim_feedforward=self.d_model,
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
        

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# StreamFusionGate (SENet 스타일 차원별 융합)
# ================================================================
class StreamFusionGate(nn.Module):
    """
    두 스트림(seq, static)을 차원별로 융합
    스칼라 softmax 대비: d_model 각 차원마다 독립적 중요도 학습
    """
    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.d_model = d_model
        self.se = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),    # squeeze
            nn.LayerNorm(d_model // reduction),
            nn.ReLU(),
            nn.Linear(d_model // reduction, d_model * 2),  # excite (두 스트림)
            nn.Sigmoid()
        )

    def forward(self, h_seq, h_static):
        """
        h_seq, h_static: (B, S, D) or (B, 1, D)
        returns: (B, S, D)
        """
        combined = h_seq + h_static                        # (B, S, D)
        gate     = self.se(combined)                       # (B, S, D*2)
        g_seq    = gate[:, :, :self.d_model]               # (B, S, D)
        g_static = gate[:, :, self.d_model:]               # (B, S, D)
        return g_seq * h_seq + g_static * h_static         # (B, S, D)


# ================================================================
# ContinuousFeatureMLP
# asof 4개 피처를 함께 투영하여 피처 간 상호작용 학습
# 기존 ContinuousFeatureTokenizer(독립 투영) 대체
# ================================================================
class ContinuousFeatureMLP(nn.Module):
    """
    asof 연속 피처 4개(price_std, last_price_diff, repurchase, weekend)
    → MLP로 상호작용 학습 → d_model
    """
    def __init__(self, num_asof_feats: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_asof_feats, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, S, num_asof_feats)
        returns: (B, S, d_model)
        """
        return self.mlp(x)


# ================================================================
# SASRecUserTower_v4
# ================================================================
class SASRecUserTower_v3(nn.Module):
    """
    변경 사항 요약:
      [제거]
        - feature_transformer (type/color/graphic/section 노이즈)
        - ContinuousFeatureTokenizer → ContinuousFeatureMLP로 교체
        - seq_gate[0~3] sigmoid gate → 컴포넌트별 LayerNorm으로 대체
        - static_gate[0~5] sigmoid gate → static_mlp weight가 중요도 직접 학습
        - feat_proj, fusion_gate(Linear)

      [추가]
        - price_item_proj: 개별 아이템 가격 → item_emb 직접 주입
        - ln_pretrained / ln_item_id / ln_price: 컴포넌트별 LayerNorm
        - ContinuousFeatureMLP: asof 4개 상호작용 학습
        - StreamFusionGate: SENet 차원별 융합
        - seq_gate[4] (target_week): 유일하게 유지 (최종 output injection)
    """
    def __init__(self, args):
        super().__init__()
        self.d_model      = args.d_model
        self.max_len      = args.max_len
        self.dropout_rate = args.dropout

        # ── [1] Item Stream ──────────────────────────────────────
        self.item_proj       = nn.Linear(args.pretrained_dim, self.d_model)
        self.item_id_emb     = nn.Embedding(args.num_items + 1, self.d_model, padding_idx=0)
        self.price_item_proj = nn.Linear(1, self.d_model)   # 개별 가격 전용
        self.pos_emb         = nn.Embedding(self.max_len, self.d_model)

        # 컴포넌트별 LayerNorm (스케일 균형)
        # sigmoid gate 대신 각 컴포넌트가 동일 스케일로 합산되도록 보장
        self.ln_pretrained = nn.LayerNorm(self.d_model)
        self.ln_item_id    = nn.LayerNorm(self.d_model)
        self.ln_price      = nn.LayerNorm(self.d_model)

        # Global context (recency + week)
        self.recency_proj = nn.Linear(1, self.d_model)
        self.week_proj    = nn.Linear(2, self.d_model)
        self.global_ln    = nn.LayerNorm(self.d_model)

        # item_emb 최종 정규화 + dropout
        self.emb_ln_item      = nn.LayerNorm(self.d_model)
        self.emb_dropout_item = nn.Dropout(self.dropout_rate)

        # Item Transformer (2층, nhead)
        item_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.nhead,
            dim_feedforward=self.d_model,
            dropout=self.dropout_rate,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.item_transformer = nn.TransformerEncoder(
            item_layer, num_layers=args.num_layers
        )

        # ── [2] Static Stream ────────────────────────────────────
        mid_dim, low_dim = 16, 4

        # 이산 피처 Embedding (gate 제거 → Embedding weight 자체가 중요도 학습)
        self.age_emb         = nn.Embedding(11, mid_dim, padding_idx=0)
        self.channel_emb     = nn.Embedding(4,  low_dim, padding_idx=0)
        self.club_status_emb = nn.Embedding(4,  low_dim, padding_idx=0)
        self.news_freq_emb   = nn.Embedding(3,  low_dim, padding_idx=0)
        self.fn_emb          = nn.Embedding(3,  low_dim, padding_idx=0)
        self.active_emb      = nn.Embedding(3,  low_dim, padding_idx=0)

        # asof 연속 피처 MLP (gate 제거 → MLP weight가 중요도 학습)
        # cont_feats index: 0=price(item_emb용), 1~4=asof 4개
        num_asof = 4
        self.cont_mlp = ContinuousFeatureMLP(
            num_asof_feats=num_asof,
            d_model=32,
            dropout=self.dropout_rate
        )

        # static_mlp 입력 차원
        # age(16) + chan(4) + club(4) + news(4) + fn(4) + act(4) + cont_mlp(d_model)
        cont_out_dim = 32  # cont_mlp d_model과 동일하게
        static_input_dim = mid_dim + low_dim * 5 + cont_out_dim  # 36 + 32 = 68

        self.static_mlp = nn.Sequential(
            nn.Linear(static_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        # ── [3] Fusion ───────────────────────────────────────────
        self.seq_proj    = nn.Linear(self.d_model, self.d_model)
        self.static_proj = nn.Linear(self.d_model, self.d_model)

        # SENet 차원별 융합 (스칼라 softmax 대체)
        self.fusion_gate = StreamFusionGate(self.d_model, reduction=4)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # ── [4] target_week gate (유일하게 유지) ─────────────────
        # 최종 output에 additive injection → 스케일 조절이 실질적으로 의미 있음
        self.target_week_gate = nn.Parameter(torch.ones(1))

        # 가중치 초기화
        self.apply(self._init_weights)

        # week/recency xavier 재초기화 (apply 이후 덮어쓰기)
        nn.init.xavier_normal_(self.week_proj.weight)
        nn.init.xavier_normal_(self.recency_proj.weight)
        nn.init.constant_(self.week_proj.bias, 0)
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
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

    def _build_week_vec(self, week_tensor):
        """week 텐서(B,S) → cyclical 투영 (B,S,D)"""
        week_rad = (week_tensor.float() / 52.0) * (2 * math.pi)
        week_cyclical = torch.cat([
            torch.sin(week_rad).unsqueeze(-1),
            torch.cos(week_rad).unsqueeze(-1)
        ], dim=-1)                             # (B, S, 2)
        return self.week_proj(week_cyclical)   # (B, S, D)

    def forward(self,
                pretrained_vecs, item_ids, time_bucket_ids,
                type_ids, color_ids, graphic_ids, section_ids,
                age_bucket, price_bucket, cnt_bucket, recency_bucket,
                channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids,
                cont_feats,
                recency_offset, current_week, session_ids, target_week=None,
                padding_mask=None, training_mode=True):

        device     = item_ids.device
        seq_len    = item_ids.size(1)
        batch_size = item_ids.size(0)

        # ── position / causal mask ────────────────────────────────
        #positions      = torch.arange(seq_len, device=device).unsqueeze(0)
        #pos_embeddings = self.pos_emb(positions)                    # (1, S, D)
        #causal_mask    = self.get_causal_mask(seq_len, device)
        prev_session_ids = torch.cat([
            torch.zeros_like(session_ids[:, :1]), 
            session_ids[:, :-1]
        ], dim=1)
        
        # 2. session_id가 변경된 지점을 감지 (변경되었으면 1, 같으면 0)
        is_session_change = (session_ids != prev_session_ids).long()
        
        # 3. 누적합(cumsum)을 통해 세션별 고유 포지션 인덱스 생성
        # 예: session_ids = [10, 10, 12, 12] -> is_change = [1, 0, 1, 0] -> positions = [1, 1, 2, 2]
        positions = torch.cumsum(is_session_change, dim=1)
        
        # 4. 패딩 토큰 위치는 포지션 인덱스 0으로 마스킹 초기화
        positions = positions.masked_fill(padding_mask, 0)
        
        # 5. 최대 max_len을 초과하지 않도록 안전장치(clamp) 적용 후 임베딩 통과
        positions = positions.clamp(max=self.max_len - 1)
        pos_embeddings = self.pos_emb(positions)  # 결과 shape: (B, S, D)

        causal_mask = self.get_causal_mask(seq_len, device)


        # ── static 이산 피처 시퀀스 차원 확장 ────────────────────
        age_bucket      = age_bucket.unsqueeze(1).expand(-1, seq_len)
        club_status_ids = club_status_ids.unsqueeze(1).expand(-1, seq_len)
        news_freq_ids   = news_freq_ids.unsqueeze(1).expand(-1, seq_len)
        fn_ids          = fn_ids.unsqueeze(1).expand(-1, seq_len)
        active_ids      = active_ids.unsqueeze(1).expand(-1, seq_len)

        # ════════════════════════════════════════════════════════════
        # Phase 1: Global Context (recency + week)
        # ════════════════════════════════════════════════════════════
        recency_scaled = (recency_offset.float() / 365.0).unsqueeze(-1)  # (B,S,1)
        r_offset_vec   = self.recency_proj(recency_scaled)                # (B,S,D)
        w_vec          = self._build_week_vec(current_week)               # (B,S,D)

        # global_context: 두 벡터 합산 후 LayerNorm
        # (게이트 제거: global_ln이 스케일 균형 담당)
        global_context = self.global_ln(r_offset_vec + w_vec)            # (B,S,D)

        # ════════════════════════════════════════════════════════════
        # Phase 2: Item Stream
        # 각 컴포넌트 독립 LayerNorm → 동일 스케일 합산
        # sigmoid gate 제거: LayerNorm이 스케일 균형 보장
        # ════════════════════════════════════════════════════════════
        comp_pretrained = self.ln_pretrained(self.item_proj(pretrained_vecs))
        comp_item_id    = self.ln_item_id(self.item_id_emb(item_ids))

        item_price      = cont_feats[:, :, 0:1]                           # (B,S,1)
        comp_price      = self.ln_price(self.price_item_proj(item_price)) # (B,S,D)

        # 합산: pretrained + item_id + price + position + global_context
        item_emb = (
            comp_pretrained
            + comp_item_id
            + comp_price
            + pos_embeddings
            + global_context
        )

        item_emb = self.emb_ln_item(item_emb)
        item_emb = self.emb_dropout_item(item_emb)

        item_output = self.item_transformer(
            item_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )                                                                  # (B,S,D)

        # ════════════════════════════════════════════════════════════
        # Phase 3: Static Stream
        # gate 제거: Embedding weight + static_mlp weight가 중요도 직접 학습
        # ════════════════════════════════════════════════════════════
        emb_age  = self.age_emb(age_bucket)                               # (B,S,16)
        emb_chan = self.channel_emb(channel_ids)                          # (B,S,4)
        emb_club = self.club_status_emb(club_status_ids)                  # (B,S,4)
        emb_news = self.news_freq_emb(news_freq_ids)                      # (B,S,4)
        emb_fn   = self.fn_emb(fn_ids)                                    # (B,S,4)
        emb_act  = self.active_emb(active_ids)                            # (B,S,4)

        # asof 4개 MLP (price 제외, index 1~4)
        asof_feats = cont_feats[:, :, 1:]                                 # (B,S,4)
        cont_vec   = self.cont_mlp(asof_feats)                            # (B,S,D)

        static_input = torch.cat([
            emb_age, emb_chan, emb_club,
            emb_news, emb_fn, emb_act,
            cont_vec
        ], dim=-1)                              # (B,S, 16+4*5+D = 36+D)

        user_profile_vec = self.static_mlp(static_input)                  # (B,S,D)

        # ════════════════════════════════════════════════════════════
        # Phase 4: target_week 벡터 (gate 유지)
        # 최종 output에 직접 더해지므로 스케일 조절 의미 있음
        # ════════════════════════════════════════════════════════════
        if target_week is not None:
            target_week_vec = (
                self._build_week_vec(target_week)
                * torch.sigmoid(self.target_week_gate)
            )                                                              # (B,S,D)

        # ════════════════════════════════════════════════════════════
        # Phase 5: SENet Fusion
        # ════════════════════════════════════════════════════════════
        if training_mode:
            h_seq    = self.seq_proj(item_output)                         # (B,S,D)
            h_static = self.static_proj(user_profile_vec)                 # (B,S,D)

            # StreamFusionGate: 차원별 중요도로 두 스트림 융합
            fused     = self.fusion_gate(h_seq, h_static)                 # (B,S,D)
            final_vec = self.output_proj(fused)                           # (B,S,D)

            if target_week is not None:
                final_vec = final_vec + target_week_vec

            return F.normalize(final_vec, p=2, dim=-1)

        else:
            # 추론: 마지막 스텝만 사용
            h_seq    = self.seq_proj(item_output[:, -1:, :])              # (B,1,D)
            h_static = self.static_proj(user_profile_vec[:, -1:, :])     # (B,1,D)

            fused     = self.fusion_gate(h_seq, h_static)                 # (B,1,D)
            final_vec = self.output_proj(fused).squeeze(1)                # (B,D)

            if target_week is not None:
                final_vec = final_vec + target_week_vec[:, -1, :]

            return F.normalize(final_vec, p=2, dim=-1)




        
import json
import torch

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

def mine_global_hard_negatives_manual(item_embs, exclusion_top_k=50, mine_k=5, batch_size=2048, device='cuda'):
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
def mine_global_hard_negatives(
    item_embs, sbert_embs, 
    fn_threshold=0.85,   # 상한: FN 제거
    fn_lower=0.50,       # 하한: easy negative 제거 (None이면 미적용)
    exclusion_top_k=5,
    mine_k=150, 
    batch_size=2048,
    device='cuda'
):
    num_items = item_embs.size(0)
    hard_neg_pool = torch.zeros((num_items, mine_k), dtype=torch.long, device=device)

    total_masked_upper  = 0   # FN 마스킹 (상한)
    total_masked_lower  = 0   # easy negative 마스킹 (하한)
    total_available     = 0

    shield_label = f"S-BERT >= {fn_threshold} masked"
    if fn_lower is not None:
        shield_label += f", < {fn_lower} masked"
    print(f"🔍 Mining Hard Negatives with Semantic Shield ({shield_label})...")

    for i in range(0, num_items, batch_size):
        end_i     = min(i + batch_size, num_items)
        batch_len = end_i - i

        batch_model_embs = item_embs[i:end_i]
        batch_sbert_embs = sbert_embs[i:end_i]

        sim_matrix   = torch.matmul(batch_model_embs, item_embs.T)  # [B, N]
        semantic_sim = torch.matmul(batch_sbert_embs, sbert_embs.T) # [B, N]

        # ── 상한 마스킹: FN 제거 (기존) ───────────────────────
        upper_mask = semantic_sim >= fn_threshold                    # [B, N]
        batch_n_masked_upper = upper_mask.sum().item()

        # ── 하한 마스킹: easy negative 제거 (신규) ────────────
        if fn_lower is not None:
            lower_mask = semantic_sim < fn_lower                     # [B, N]
            batch_n_masked_lower = lower_mask.sum().item()
        else:
            lower_mask = None
            batch_n_masked_lower = 0

        # ── 통계 누적 ──────────────────────────────────────────
        total_masked_upper += batch_n_masked_upper
        total_masked_lower += batch_n_masked_lower

        total_n_masked = batch_n_masked_upper + batch_n_masked_lower
        masked_per_item    = total_n_masked / batch_len
        available_per_item = num_items - masked_per_item - 2  # 자기자신 + padding
        total_available   += max(available_per_item, 0) * batch_len

        # ── 마스킹 적용 ────────────────────────────────────────
        sim_matrix.masked_fill_(upper_mask, -float('inf'))
        if lower_mask is not None:
            sim_matrix.masked_fill_(lower_mask, -float('inf'))

        # 자기자신 / padding 마스킹
        batch_indices = torch.arange(i, end_i, device=device)
        sim_matrix[torch.arange(batch_len, device=device), batch_indices] = -float('inf')
        sim_matrix[:, 0] = -float('inf')

        # ── Top-K 추출 ─────────────────────────────────────────
        total_k = min(exclusion_top_k + mine_k, num_items - 1)
        _, topk_indices_global = torch.topk(sim_matrix, total_k, dim=1)
        hard_negs = topk_indices_global[:, exclusion_top_k: exclusion_top_k + mine_k]

        if hard_negs.size(1) < mine_k:
            pad_size  = mine_k - hard_negs.size(1)
            pad_idx   = hard_negs[:, [0]].expand(-1, pad_size)
            hard_negs = torch.cat([hard_negs, pad_idx], dim=1)

        hard_neg_pool[i:end_i] = hard_negs

    # ── 최종 통계 ──────────────────────────────────────────────
    n_sq = num_items * num_items + 1e-9

    shield_metrics = {
        # 상한 마스킹 비율 (FN 제거)
        'HNM/sbert_mask_ratio_upper':      total_masked_upper / n_sq,
        'HNM/sbert_masked_per_item_upper': total_masked_upper / (num_items + 1e-9),

        # 하한 마스킹 비율 (easy negative 제거)
        'HNM/sbert_mask_ratio_lower':      total_masked_lower / n_sq,
        'HNM/sbert_masked_per_item_lower': total_masked_lower / (num_items + 1e-9),

        # 전체 마스킹 비율
        'HNM/sbert_mask_ratio_total':      (total_masked_upper + total_masked_lower) / n_sq,

        # 아이템당 실제 유효 후보 수 (mine_k 대비 충분한지 확인)
        'HNM/available_per_item':          total_available / (num_items + 1e-9),
    }

    return hard_neg_pool, shield_metrics
def mine_global_hard_negatives_PRRT(item_embs, sbert_embs, fn_threshold=0.85, exclusion_top_k=5, mine_k=150, batch_size=2048, device='cuda'):

    num_items = item_embs.size(0)
    hard_neg_pool = torch.zeros((num_items, mine_k), dtype=torch.long, device=device)
    
    print(f"🔍 Mining Hard Negatives with Semantic Shield (S-BERT >= {fn_threshold} masked)...")

    for i in range(0, num_items, batch_size):
        end_i = min(i + batch_size, num_items)
        
        # [Batch, dim]
        batch_model_embs = item_embs[i:end_i] 
        batch_sbert_embs = sbert_embs[i:end_i]
        
        # 1. 모델 임베딩 공간의 유사도 계산 (누구를 오답으로 뽑을 것인가?)
        sim_matrix = torch.matmul(batch_model_embs, item_embs.T) # [Batch, num_items]
        
        # 2. 💡 [핵심] 시맨틱 방어막(Semantic Shield) 계산 및 마스킹
        # 현재 배치의 S-BERT 벡터와 전체 아이템의 S-BERT 벡터 간의 절대 유사도 계산
        semantic_sim_matrix = torch.matmul(batch_sbert_embs, sbert_embs.T)
        
        # S-BERT 기준 0.85 이상인 대체재들은 모델 공간에서 -inf로 강제 마스킹!
        sim_matrix.masked_fill_(semantic_sim_matrix >= fn_threshold, -float('inf'))
        
        # 3. 기본 마스킹: 자기 자신 및 패딩(0번 인덱스) 제외
        batch_indices = torch.arange(i, end_i, device=device)
        sim_matrix[torch.arange(end_i - i, device=device), batch_indices] = -float('inf')
        sim_matrix[:, 0] = -float('inf') # 0번 인덱스(패딩)가 오답으로 뽑히지 않도록 영구 제외
        
        # 4. Top-K 추출 (모델 공간에서 가까운 순서대로)
        total_k = min(exclusion_top_k + mine_k, num_items - 1)
        _, topk_indices_global = torch.topk(sim_matrix, total_k, dim=1)
        
        # 5. 모델 오류(Model Collapse) 대비용 최소한의 exclusion_top_k만 건너뛰고 채굴
        # (이미 완벽한 시맨틱 대체재는 날아갔으므로, exclusion_top_k를 5~10 수준으로 낮춰도 무방합니다)
        hard_negs = topk_indices_global[:, exclusion_top_k : exclusion_top_k + mine_k]
        
        # 예외 처리 (수량 부족 시 패딩)
        if hard_negs.size(1) < mine_k:
            pad_size = mine_k - hard_negs.size(1)
            pad_idx = hard_negs[:, [0]].expand(-1, pad_size)
            hard_negs = torch.cat([hard_negs, pad_idx], dim=1)
            
        hard_neg_pool[i:end_i] = hard_negs
        
    return hard_neg_pool
import torch
import torch.nn.functional as F



def inbatch_corrected_logq_loss_with_hybrid_hard_neg_prev22(
    user_emb, seq_item_emb, target_ids, user_ids, log_q_tensor,
    hn_item_emb=None, batch_hard_neg_ids=None,item_embedding_weight=None,
    flat_history_item_ids=None,
    step_weights=None,
    temperature=0.07,
    lambda_logq=1.0,
    margin=0.00,
    # ✅ [MNS 신규 파라미터]
    T_HN=0.14,
    beta=0.25,
    T_sample=0.5, 
    boundary_ratio=0.85,
    return_metrics=False
):
    N = user_emb.size(0)
    device = user_emb.device
    SAFE_NEG_INF = -1e9

    # -----------------------------------------------------------
    # 0. 가중치 준비
    # -----------------------------------------------------------
    if step_weights is not None:
        step_weights = torch.as_tensor(step_weights, device=device, dtype=torch.float32)
        weight_sum = step_weights.sum() + 1e-9
    else:
        weight_sum = None

    # -----------------------------------------------------------
    # 1. In-batch Logits 계산 (log-Q 보정 포함)
    # -----------------------------------------------------------
    sim_matrix = torch.matmul(user_emb, seq_item_emb.T)  # [N, N]
    pos_sim = torch.diagonal(sim_matrix)                  # [N]
    labels = torch.arange(N, device=device)
    logits = sim_matrix / temperature

    if margin > 0.0:
        logits[labels, labels] -= (margin / temperature)

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # False Negative 마스킹 (동일 아이템 / 동일 유저)
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits = logits.masked_fill(false_neg_mask, SAFE_NEG_INF)
    logits = torch.clamp(logits, min=SAFE_NEG_INF, max=1e4)

    metrics = {}

    # -----------------------------------------------------------
    # 2. Hard Negative Mining & MNS Loss 계산
    # -----------------------------------------------------------
    num_hn_to_use = 40
    loss_hn = None

    if hn_item_emb is not None and batch_hard_neg_ids is not None:
        #pool_multiplier = 2
        num_pool = int(num_hn_to_use * 3.5)
        #num_pool = num_hn_to_use * pool_multiplier
        #skip_top_k = 5
        #avg_pos_sim = pos_sim.detach().mean().item()


        # [STEP 1] Gradient 없이 HN 후보 필터링 & 선택
        with torch.no_grad():
            hn_emb_no_grad = hn_item_emb.detach()
            hn_sim_no_grad = torch.bmm(
                user_emb.unsqueeze(1),
                hn_emb_no_grad.transpose(1, 2)
            ).squeeze(1)  # [N, pool_size]

            # Absolute FN 마스크 (유저 히스토리에 있는 아이템)
            absolute_fn_mask = torch.zeros_like(hn_sim_no_grad, dtype=torch.bool)
            if flat_history_item_ids is not None:
                absolute_fn_mask = (
                    batch_hard_neg_ids.unsqueeze(2) == flat_history_item_ids.unsqueeze(1)
                ).any(dim=2)

            # Dynamic FN 마스크 (pos_sim에 너무 가까운 아이템)
            dynamic_boundary = pos_sim.unsqueeze(1) * boundary_ratio
            dynamic_fn_mask = hn_sim_no_grad >= dynamic_boundary
            final_fn_mask = absolute_fn_mask | dynamic_fn_mask

            # Top-k 선택 후 무작위 서브샘플링
            masked_sims = hn_sim_no_grad.masked_fill(final_fn_mask, -1e4)
            
            
            # ✅ 수정: Hardness-aware Stochastic Sampling
            # Step A. skip_top_k 제거 (가장 쉬운 1개 제외하는 건 의미 없음, 가장 어려운 1개 제외가 목적)
            _, top_idx_pool = torch.topk(masked_sims, num_pool, dim=1)  # 바로 num_pool개
            pool_sims = torch.gather(masked_sims, 1, top_idx_pool)  # [N, num_pool] 해당 유사도 추출

            # Step B. 하드니스 점수를 확률로 변환 (T_sample로 exploration 조절)
            # 낮을수록 harder에 집중, 높을수록 uniform에 수렴
            sampling_weights = F.softmax(pool_sims / T_sample, dim=1)  # [N, num_pool]

            # Step C. 가중치 비례 비복원 추출 (multinomial)
            top_idx_local = torch.multinomial(
                sampling_weights,
                num_samples=num_hn_to_use,
                replacement=False
            )  # [N, num_hn_to_use] — pool 내 로컬 인덱스
            top_idx = torch.gather(top_idx_pool, 1, top_idx_local)  # [N, num_hn_to_use] — 글로벌 인덱스


        # [STEP 2] ✅ 핵심 수정: 선택된 인덱스로 live item_tower에서 직접 임베딩 조립
        batch_hn_ids_final = torch.gather(batch_hard_neg_ids, 1, top_idx)  # [N, num_hn]

        if item_embedding_weight is not None:
            # item_tower.item_matrix.weight에서 직접 조회 → gradient 흐름 ✅
            hn_emb_flat = item_embedding_weight[batch_hn_ids_final.view(-1)]     # [N*num_hn, D]
            hn_emb_final = F.normalize(hn_emb_flat, p=2, dim=1).view(
                batch_hn_ids_final.size(0), batch_hn_ids_final.size(1), -1
            )  # [N, num_hn, D]
        else:
            # fallback: 기존 방식 (gradient 없음)
            top_idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, hn_item_emb.size(2))
            hn_emb_final = torch.gather(hn_item_emb, 1, top_idx_expanded).detach()
        
        
        # [STEP 3] HN similarity 계산
        hn_sim = torch.bmm(
            user_emb.unsqueeze(1),
            hn_emb_final.transpose(1, 2)
        ).squeeze(1)  # [N, num_hn]

        # Safety mask (최종 FN 제거)
        final_safety_mask = hn_sim >= (pos_sim.unsqueeze(1) * boundary_ratio)
        hn_sim_safe = hn_sim.masked_fill(final_safety_mask, -1e4)

        # ✅ [MNS] HN loss: log-Q 보정 없이, 별도 온도(T_HN)로 분리 계산
        # pos를 index 0으로, HN들과 경쟁하는 구조
        hn_loss_input = torch.cat([
            pos_sim.unsqueeze(1) / T_HN,   # [N, 1]
            hn_sim_safe / T_HN             # [N, num_hn]
        ], dim=1)  # [N, 1 + num_hn]
        hn_loss_input = torch.clamp(hn_loss_input, min=SAFE_NEG_INF, max=1e4)
        hn_labels = torch.zeros(N, dtype=torch.long, device=device)

        if step_weights is not None:
            loss_hn = F.cross_entropy(hn_loss_input, hn_labels, reduction='none')
            loss_hn = (loss_hn * step_weights).sum() / weight_sum
        else:
            loss_hn = F.cross_entropy(hn_loss_input, hn_labels)

        # Metrics
        if return_metrics:
            metrics['hn/discarded_ratio'] = dynamic_fn_mask.float().mean().item()

            valid_hn_sim = hn_sim[~final_safety_mask]
            metrics['sim/hn_true_hard'] = (
                valid_hn_sim.mean().item() if valid_hn_sim.numel() > 0 else 0.0
            )

            with torch.no_grad():
                masked_hn_sim = hn_sim.masked_fill(final_safety_mask, -1e4)
                max_hn_per_user, _ = masked_hn_sim.max(dim=1)
                valid_max_hn = max_hn_per_user[max_hn_per_user > -1.0]
                metrics['sim/hn_max_mean'] = (
                    valid_max_hn.mean().item() if valid_max_hn.numel() > 0 else 0.0
                )

    # -----------------------------------------------------------
    # 3. In-batch Loss 계산
    # -----------------------------------------------------------
    if step_weights is not None:
        loss_inbatch = F.cross_entropy(logits, labels, reduction='none')
        loss_inbatch = (loss_inbatch * step_weights).sum() / weight_sum
        if return_metrics:
            metrics['sim/pos'] = ((pos_sim * step_weights).sum() / weight_sum).item()
    else:
        loss_inbatch = F.cross_entropy(logits, labels)
        if return_metrics:
            metrics['sim/pos'] = pos_sim.mean().item()

    # -----------------------------------------------------------
    # 4. 최종 Loss 조합 (In-batch + MNS HN)
    # -----------------------------------------------------------
    if loss_hn is not None:
        loss = loss_inbatch + beta * loss_hn
        if return_metrics:
            metrics['loss/inbatch'] = loss_inbatch.item()
            metrics['loss/hn'] = loss_hn.item()
            metrics['loss/hn_ratio'] = (beta * loss_hn).item() / (loss_inbatch.item() + 1e-9)
    else:
        loss = loss_inbatch

    # -----------------------------------------------------------
    # 5. Probability-based Metrics (기존 유지)
    # -----------------------------------------------------------
    if return_metrics:
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            pos_probs = probs.diagonal()
            neg_probs_total = 1.0 - pos_probs

            if loss_hn is not None:
                hn_probs = F.softmax(hn_loss_input, dim=1)[:, 1:]  # index 0이 pos이므로
                hn_probs_sum = hn_probs.sum(dim=1)
                relative_hn_ratio = hn_probs_sum / (neg_probs_total + 1e-9)

                if step_weights is not None:
                    metrics['hn/influence_ratio'] = (
                        (hn_probs_sum * step_weights).sum() / weight_sum
                    ).item()
                    metrics['hn/relative_influence'] = (
                        (relative_hn_ratio * step_weights).sum() / weight_sum
                    ).item()
                else:
                    metrics['hn/influence_ratio'] = hn_probs_sum.mean().item()
                    metrics['hn/relative_influence'] = relative_hn_ratio.mean().item()

        return loss, metrics

    return loss


def inbatch_corrected_logq_loss_with_hybrid_hard_neg(
    user_emb, seq_item_emb, target_ids, user_ids, log_q_tensor,
    hn_item_emb=None, batch_hard_neg_ids=None,item_embedding_weight=None,
    flat_history_item_ids=None,
    step_weights=None,
    temperature=0.07,
    lambda_logq=1.0,
    margin=0.00,
    T_HN=0.14,
    beta=0.25,
    T_sample=0.5, 
    boundary_ratio=0.85, # (더 이상 내부 마스킹에 사용하지 않음)
    return_metrics=False
):
    N = user_emb.size(0)
    device = user_emb.device
    SAFE_NEG_INF = -1e9

    # -----------------------------------------------------------
    # 0. 가중치 준비
    # -----------------------------------------------------------
    if step_weights is not None:
        step_weights = torch.as_tensor(step_weights, device=device, dtype=torch.float32)
        weight_sum = step_weights.sum() + 1e-9
    else:
        weight_sum = None

    # -----------------------------------------------------------
    # 1. In-batch Logits 계산
    # -----------------------------------------------------------
    sim_matrix = torch.matmul(user_emb, seq_item_emb.T)  # [N, N]
    pos_sim = torch.diagonal(sim_matrix)                  # [N]
    labels = torch.arange(N, device=device)
    logits = sim_matrix / temperature

    if margin > 0.0:
        logits[labels, labels] -= (margin / temperature)

    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[target_ids]
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    same_item_mask = torch.eq(target_ids.unsqueeze(1), target_ids.unsqueeze(0))
    same_user_mask = torch.eq(user_ids.unsqueeze(1), user_ids.unsqueeze(0))
    false_neg_mask = (same_item_mask | same_user_mask) & ~diag_mask
    logits = logits.masked_fill(false_neg_mask, SAFE_NEG_INF)
    logits = torch.clamp(logits, min=SAFE_NEG_INF, max=1e4)

    metrics = {}

    # -----------------------------------------------------------
    # 2. Hard Negative Mining & MNS Loss 계산
    # -----------------------------------------------------------
    num_hn_to_use = 40
    loss_hn = None

    if hn_item_emb is not None and batch_hard_neg_ids is not None:
        num_pool = int(num_hn_to_use * 3.5)

        # [STEP 1] Gradient 없이 HN 후보 필터링 & 선택
        with torch.no_grad():
            hn_emb_no_grad = hn_item_emb.detach()
            hn_sim_no_grad = torch.bmm(
                user_emb.unsqueeze(1),
                hn_emb_no_grad.transpose(1, 2)
            ).squeeze(1)  # [N, pool_size]

            # 💡 [유지] Absolute FN 마스크: 유저가 이미 클릭했던(히스토리) 아이템 제외
            final_fn_mask = torch.zeros_like(hn_sim_no_grad, dtype=torch.bool)
            if flat_history_item_ids is not None:
                final_fn_mask = (
                    batch_hard_neg_ids.unsqueeze(2) == flat_history_item_ids.unsqueeze(1)
                ).any(dim=2)

            # 🚀 [삭제] dynamic_boundary 마스킹 삭제 완료! (S-BERT가 이미 일함)
            masked_sims = hn_sim_no_grad.masked_fill(final_fn_mask, -1e4)
            
            
            # 가장 유사한 skip_top_k개는 버리고, 그 다음부터 num_pool개만큼 가져옵니다.
            _, top_idx_pool = torch.topk(masked_sims, num_pool, dim=1)
            pool_sims = torch.gather(masked_sims, 1, top_idx_pool)

            # Step B. 하드니스 점수를 확률로 변환 (T_sample로 exploration 조절)
            sampling_weights = F.softmax(pool_sims / T_sample, dim=1)  # [N, num_pool]

            # Step C. 가중치 비례 비복원 추출 (multinomial)
            top_idx_local = torch.multinomial(
                sampling_weights,
                num_samples=num_hn_to_use,
                replacement=False
            )  # [N, num_hn_to_use]
            
            # 글로벌 인덱스 복구
            top_idx = torch.gather(top_idx_pool, 1, top_idx_local)  # [N, num_hn_to_use]

        # [STEP 2] 선택된 인덱스로 live item_tower에서 직접 임베딩 조립
        batch_hn_ids_final = torch.gather(batch_hard_neg_ids, 1, top_idx)  # [N, num_hn]

        if item_embedding_weight is not None:
            hn_emb_flat = item_embedding_weight[batch_hn_ids_final.view(-1)]
            hn_emb_final = F.normalize(hn_emb_flat, p=2, dim=1).view(
                batch_hn_ids_final.size(0), batch_hn_ids_final.size(1), -1
            )  # [N, num_hn, D]
        else:
            top_idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, hn_item_emb.size(2))
            hn_emb_final = torch.gather(hn_item_emb, 1, top_idx_expanded).detach()
        
        # [STEP 3] HN similarity 계산
        hn_sim = torch.bmm(
            user_emb.unsqueeze(1),
            hn_emb_final.transpose(1, 2)
        ).squeeze(1)  # [N, num_hn]

        # 🚀 [삭제] final_safety_mask 도 제거 (이미 깨끗한 풀이므로 안심하고 때림)
        
        # [MNS] HN loss 계산
        hn_loss_input = torch.cat([
            pos_sim.unsqueeze(1) / T_HN,   # [N, 1] (정답)
            hn_sim / T_HN                  # [N, num_hn] (오답들)
        ], dim=1)  
        
        hn_loss_input = torch.clamp(hn_loss_input, min=SAFE_NEG_INF, max=1e4)
        hn_labels = torch.zeros(N, dtype=torch.long, device=device)

        if step_weights is not None:
            loss_hn = F.cross_entropy(hn_loss_input, hn_labels, reduction='none')
            loss_hn = (loss_hn * step_weights).sum() / weight_sum
        else:
            loss_hn = F.cross_entropy(hn_loss_input, hn_labels)

        # Metrics 기록
        if return_metrics:
            metrics['sim/hn_true_hard'] = hn_sim.mean().item() if hn_sim.numel() > 0 else 0.0
            
            with torch.no_grad():
                max_hn_per_user, _ = hn_sim.max(dim=1)
                metrics['sim/hn_max_mean'] = max_hn_per_user.mean().item() if max_hn_per_user.numel() > 0 else 0.0

    # -----------------------------------------------------------
    # 3. In-batch Loss 계산
    # -----------------------------------------------------------
    if step_weights is not None:
        loss_inbatch = F.cross_entropy(logits, labels, reduction='none')
        loss_inbatch = (loss_inbatch * step_weights).sum() / weight_sum
        if return_metrics:
            metrics['sim/pos'] = ((pos_sim * step_weights).sum() / weight_sum).item()
    else:
        loss_inbatch = F.cross_entropy(logits, labels)
        if return_metrics:
            metrics['sim/pos'] = pos_sim.mean().item()

    # -----------------------------------------------------------
    # 4. 최종 Loss 조합 (In-batch + MNS HN)
    # -----------------------------------------------------------
    if loss_hn is not None:
        loss = loss_inbatch + beta * loss_hn
        if return_metrics:
            metrics['loss/inbatch'] = loss_inbatch.item()
            metrics['loss/hn'] = loss_hn.item()
            metrics['loss/hn_ratio'] = (beta * loss_hn).item() / (loss_inbatch.item() + 1e-9)
    else:
        loss = loss_inbatch

    return loss if not return_metrics else (loss, metrics)
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
        user_path = os.path.join(cfg.base_dir, "features_user_w_meta_nonleak_v2.parquet") 
        item_path = os.path.join(cfg.base_dir, "features_item.parquet")
        #seq_path = os.path.join(cfg.base_dir, "features_sequence_cleaned.parquet")
        
        TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
        USER_VAL_FEAT_PATH = os.path.join(cfg.base_dir, "features_user_w_meta_nonleak_val_v2.parquet")
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
                'session_ids': session_ids, 'padding_mask': padding_mask,
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


def log_feature_contributions_v4(model, wandb, epoch=None):
    """
    SASRecUserTower_v4 전용 WandB 피처 기여도 로깅
    
    v3 (sigmoid gate 값) → v4 (weight norm 기반 실질 기여도)
    
    모니터링 항목:
      1. Item Stream: 컴포넌트별 LayerNorm weight norm
         → pretrained / item_id / price 각각의 실질 스케일
      2. Static Stream: Embedding weight norm (피처별 행 평균)
         → static_mlp 첫 번째 레이어 열 norm (피처 블록별)
      3. cont_mlp: 입력 가중치 norm (asof 4개 피처별)
      4. SENet gate: fusion_gate 평균 활성화 (seq vs static 비율)
      5. target_week_gate: 스칼라 값 (유일하게 남은 gate)
    """
    log_dict = {}
 
    with torch.no_grad():
 
        # ════════════════════════════════════════════════════════
        # 1. Item Stream 컴포넌트별 LayerNorm weight norm
        #    LayerNorm.weight = 각 차원의 스케일 파라미터
        #    norm이 클수록 해당 컴포넌트가 item_emb에서 큰 스케일 유지
        # ════════════════════════════════════════════════════════
        ln_norms = {
            'ItemStream/ln_pretrained': model.ln_pretrained.weight,
            'ItemStream/ln_item_id':    model.ln_item_id.weight,
            'ItemStream/ln_price':      model.ln_price.weight,
        }
        for name, weight in ln_norms.items():
            log_dict[f"{name}_norm"] = weight.norm().item()
            log_dict[f"{name}_mean"] = weight.mean().item()
 
        # ════════════════════════════════════════════════════════
        # 2. Static Stream Embedding weight norm (피처별)
        #    각 Embedding의 weight matrix row norm 평균
        #    → 유효 인덱스(padding 제외)의 평균 활성화 크기
        # ════════════════════════════════════════════════════════
        emb_modules = {
            'Static/emb_age':   model.age_emb,
            'Static/emb_chan':   model.channel_emb,
            'Static/emb_club':  model.club_status_emb,
            'Static/emb_news':  model.news_freq_emb,
            'Static/emb_fn':    model.fn_emb,
            'Static/emb_act':   model.active_emb,
        }
        for name, emb in emb_modules.items():
            # padding_idx=0 제외하고 norm 계산
            valid_weights = emb.weight[1:]                  # (vocab-1, dim)
            row_norms = valid_weights.norm(dim=1)           # (vocab-1,)
            log_dict[f"{name}_norm_mean"] = row_norms.mean().item()
            log_dict[f"{name}_norm_max"]  = row_norms.max().item()
 
        # ════════════════════════════════════════════════════════
        # 3. cont_mlp 입력 가중치 norm (asof 피처별 기여도)
        #    cont_mlp.mlp[0] = Linear(4, d_model)
        #    weight shape: (d_model, 4)
        #    각 열(column)의 norm = 해당 asof 피처가 출력에 미치는 영향
        # ════════════════════════════════════════════════════════
        cont_linear = model.cont_mlp.mlp[0]                # Linear(4, d_model)
        col_norms = cont_linear.weight.norm(dim=0)         # (4,) 피처별 열 norm
 
        asof_labels = [
            'price_std',
            'last_price_diff',
            'repurchase_ratio',
            'weekend_ratio'
        ]
        for label, norm_val in zip(asof_labels, col_norms):
            log_dict[f"ContMLP/asof_{label}_col_norm"] = norm_val.item()
 
        # ════════════════════════════════════════════════════════
        # 4. static_mlp 첫 번째 레이어 열 norm (피처 블록별)
        #    static_mlp[0] = Linear(36+D, D)
        #    weight shape: (D, 36+D)
        #    피처 블록별 열 norm 합산 → 각 피처 그룹의 실질 기여도
        # ════════════════════════════════════════════════════════
        d_model = model.d_model
        static_linear = model.static_mlp[0]               # Linear(36+D, D)
        col_norms_static = static_linear.weight.norm(dim=0)  # (36+D,)
 
        # 피처 블록 인덱스 정의
        # age(16) | chan(4) | club(4) | news(4) | fn(4) | act(4) | cont_mlp(D)
        static_blocks = {
            'StaticMLP/age':       (0,  16),
            'StaticMLP/channel':   (16, 20),
            'StaticMLP/club':      (20, 24),
            'StaticMLP/news':      (24, 28),
            'StaticMLP/fn':        (28, 32),
            'StaticMLP/active':    (32, 36),
            'StaticMLP/cont_mlp':  (36, 36 + d_model),
        }
        for name, (start, end) in static_blocks.items():
            block_norm = col_norms_static[start:end].mean().item()
            log_dict[f"{name}_col_norm"] = block_norm
 
        # ════════════════════════════════════════════════════════
        # 5. target_week_gate (유일하게 남은 sigmoid gate)
        # ════════════════════════════════════════════════════════
        gate_val = torch.sigmoid(model.target_week_gate).item()
        log_dict['Gate/target_week'] = gate_val
 
        # ════════════════════════════════════════════════════════
        # 6. SENet fusion_gate se 레이어 가중치 norm
        #    se[-2] = Linear(D//4, D*2): excite 레이어
        #    seq 절반 vs static 절반 norm 비교
        #    → SENet이 두 스트림을 얼마나 구분하는지
        # ════════════════════════════════════════════════════════
        excite_weight = model.fusion_gate.se[-2].weight   # (D*2, D//4)
        seq_half    = excite_weight[:d_model, :]           # (D, D//4)
        static_half = excite_weight[d_model:, :]           # (D, D//4)
 
        seq_norm    = seq_half.norm().item()
        static_norm = static_half.norm().item()
        total_norm  = seq_norm + static_norm + 1e-9
 
        log_dict['SENet/seq_excite_norm']    = seq_norm
        log_dict['SENet/static_excite_norm'] = static_norm
        log_dict['SENet/seq_ratio']          = seq_norm / total_norm
        log_dict['SENet/static_ratio']       = static_norm / total_norm
 
    wandb.log(log_dict)
 

def train_user_tower_session_sampler_with_intent_point(epoch, model, item_tower,norm_item_embeddings, log_q_tensor, 
                                                       dataloader, optimizer, scaler, cfg, device, 
                                                       hard_neg_pool_tensor, scheduler, hn_scheduler , T_sample, beta, 
                                                       hn_refresh_interval,hn_exclusion_top_k,hn_mine_k,
                                                       seq_labels=None, static_labels=None):
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
    epoch_discard_ratio_sum = 0.0
    num_hn_metric_batches = 0

    
    current_hn_pool = hard_neg_pool_tensor
    current_norm_item_embs = norm_item_embeddings

    #prefetch_loader = CUDAPrefetcher(dataloader, device)  
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    

    # 💡 [핵심 2] 루프 시작 전, 혹시 남아있을지 모르는 이전 에포크의 그래디언트 초기화
    #optimizer.zero_grad()
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(pbar):
        # ✅ 에포크 내 pool 갱신 (hard_neg_pool이 존재할 때만)
        if (
            current_hn_pool is not None
            and hn_refresh_interval > 0
            and batch_idx > 0                        # 첫 배치는 파이프라인 캐시 그대로 사용
            and batch_idx % hn_refresh_interval == 0
        ):
            item_tower.eval()
            with torch.no_grad():
                all_item_embs = item_tower.get_all_embeddings()
                current_norm_item_embs = F.normalize(all_item_embs, p=2, dim=1)
                current_hn_pool = mine_global_hard_negatives(
                    current_norm_item_embs,
                    exclusion_top_k=hn_exclusion_top_k,
                    mine_k=hn_mine_k,
                    batch_size=2048,
                    device=device
                )
            item_tower.train()

        
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
            'recency_offset': recency_offset, 'current_week': current_week, 
            'target_week': target_week, 'session_ids': session_ids,
            'padding_mask': padding_mask,
            
            'training_mode': True
        }

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
            output_1 = model(**forward_kwargs_v1)
            
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
            
            # =======================================================
            # 💡 [리팩토링 1] 각 세션의 마지막 스텝만 추출하는 마스크 생성
            # =======================================================
            # shifted_session_ids: 현재 세션 ID 배열을 왼쪽으로 1칸 밉니다.
            shifted_session_ids = torch.roll(session_ids, shifts=-1, dims=1)
            shifted_session_ids[:, -1] = 0 # 맨 마지막 열은 패딩(0)으로 초기화 (roll 방지)
            
            # 현재 스텝의 세션 ID와 다음 스텝의 세션 ID가 다르다면, 거기가 세션의 마지막 스텝입니다.
            # (유효 데이터의 마지막 스텝은 다음 스텝이 패딩(0)이 되므로 자연스럽게 포함됩니다)
            is_session_change = (session_ids != shifted_session_ids)
            session_last_mask = is_session_change & valid_mask

            
            # 2. 💡 [신규] 세션 마지막이 아닌 '중간 스텝'들 중에서 랜덤 샘플링
            # 확률 변수 (예: 15% 확률로 중간 스텝을 학습에 참여시킴)
            intermediate_prob = 0.2
            intermediate_mask = valid_mask & ~session_last_mask
            random_sample_mask = torch.rand_like(intermediate_mask, dtype=torch.float) < intermediate_prob
                        
                        
            final_loss_mask = session_last_mask | (intermediate_mask & random_sample_mask)
            
            # 기존의 진짜 마지막 인덱스(전체 시퀀스 기준)도 추적해 둡니다. (HNM 용도)
            seq_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            batch_range = torch.arange(batch_size, device=device)
            last_indices = torch.max(seq_positions.masked_fill(~valid_mask, -1), dim=1)[0].clamp(min=0)
            
            is_last_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            is_last_mask[batch_range, last_indices] = True

            # =======================================================
            # [1] Main Loss 계산 (세션별 마지막 임베딩만 사용!)
            # =======================================================
            
            flat_output = output_1[final_loss_mask] 
            flat_targets = target_ids[final_loss_mask]
            
            batch_row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
            flat_user_ids = batch_row_indices[final_loss_mask] 
            
            # 가중치와 진짜 마지막 여부도 세션 마지막 텐서들 기준으로 필터링
            flat_weights = seq_weights[final_loss_mask] 
            flat_is_last = is_last_mask[final_loss_mask]
            
            
            MAX_FLAT_SIZE = 8500
            
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
                # 💡 HN 임베딩 조립: 후보 선택(캐시) / 학습(live) 분리
                # =======================================================
                batch_hn_item_emb_cached = None
                batch_hard_neg_ids = None

                if current_hn_pool is not None:
                    batch_hard_neg_ids = current_hn_pool[flat_targets]
                    batch_hn_item_emb_cached = current_norm_item_embs[batch_hard_neg_ids]
                
                main_loss, b_metrics = inbatch_corrected_logq_loss_with_hybrid_hard_neg(
                    user_emb=flat_user_emb, seq_item_emb=batch_seq_item_emb,
                    target_ids=flat_targets, user_ids=flat_user_ids,
                    log_q_tensor=log_q_tensor,
                    hn_item_emb=batch_hn_item_emb_cached,          # 후보 선택 전용 (캐시)
                    batch_hard_neg_ids=batch_hard_neg_ids,
                    item_embedding_weight=item_tower.item_matrix.weight,  # ✅ live re-embedding용
                    flat_history_item_ids=flat_history_item_ids, step_weights=flat_weights,
                    temperature=0.07, lambda_logq=cfg.lambda_logq, margin = 0.0,
                    T_HN=0.14, beta=beta, T_sample= T_sample,
                    return_metrics=True
                )

            else:
                main_loss = torch.tensor(0.0, device=device)
            
            # 에포크 평균용 누적 for pipeline 내
            if 'hn/discarded_ratio' in b_metrics:
                epoch_discard_ratio_sum += b_metrics['hn/discarded_ratio']
                num_hn_metric_batches += 1

            # =======================================================
            # [2] Semantic Contrastive Loss 계산
            # =======================================================
            
       
            
            total_loss = main_loss 
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
        #cl_loss_accum += cl_loss.item() # 💡 에포크 종료 후 평균 CL 계산을 위해 추가
        
        pbar.set_postfix({
            'Main_Loss': f"{main_loss.item():.4f}", # 💡 Postfix에는 순수 main_loss 표기 권장
         #   'CL_Loss': f"{cl_loss.item():.4f}"
        })
        # 100번 단위 로깅도 미니배치(384) 기준 100번이 됩니다.
        if batch_idx % 100 == 0:
            wandb_log_dict = {
                "Train/Main_Loss": main_loss.item(),
             #   "Train/CL_Loss": cl_loss.item()
                }
            for k in ['sim/pos',  'sim/hn_all', 'sim/hn', 'hn/influence_ratio', 
                    'hn/penalized_ratio', 'sim/hn_penalized', 'sim/hn_true_hard','hn/discarded_ratio',
                      'hn/relative_influence', 'hn/danger_zone_ratio', 'sim/hn_max_mean']:
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
            

            
            for k in ['loss/hn','loss/inbatch','loss/hn_ratio']:
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

    avg_discard_ratio = (
        epoch_discard_ratio_sum / num_hn_metric_batches
        if num_hn_metric_batches > 0 else 0.0
    )

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
    '''
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
    '''
    log_feature_contributions_v4(model, wandb)

    print(f"🏁 Epoch {epoch} Completed | Avg Total: {avg_loss:.4f} (Main: {avg_main:.4f}, CL: {avg_cl:.4f})")
    return avg_loss, force_mining_next_epoch,  avg_discard_ratio
import torch
import torch.nn.functional as F
import wandb
import pandas as pd
from collections import Counter
from tqdm import tqdm

def run_diagnostic_analysis(
    user_embs,               # [Num_Users, Dim] : 평가셋의 유저 벡터 (Session-last)
    item_embs,               # [Num_Items, Dim] : 전체 아이템 벡터
    target_dict,             # dict: user_id -> target_item_ids
    user_ids,                # list: user_embs와 인덱스가 매칭되는 user_id 리스트
    item_id_to_category,     # dict: item_id -> product_type_name (예: 'T-shirt')
    device='cuda'
):
    print("\n" + "="*60)
    print("🕵️‍♂️ Starting Diagnostic Analysis (Phase 1 & 2)")
    print("="*60)
    
    # 임베딩 정규화
    user_embs = F.normalize(user_embs.to(device), p=2, dim=1)
    item_embs = F.normalize(item_embs.to(device), p=2, dim=1)
    
    num_items = item_embs.size(0)
    
    # -----------------------------------------------------------
    # Phase 1. 임베딩 지형 분석 (Isotropy & Category Variance)
    # -----------------------------------------------------------
    print("📊 [Phase 1] Analyzing Item Embedding Topology...")
    
    # 1-1. Global Uniformity (전체 아이템 랜덤 1만 쌍 유사도)
    idx1 = torch.randint(0, num_items, (10000,), device=device)
    idx2 = torch.randint(0, num_items, (10000,), device=device)
    global_sims = (item_embs[idx1] * item_embs[idx2]).sum(dim=1)
    
    wandb.log({"Analysis/Global_Item_Similarity": wandb.Histogram(global_sims.cpu().numpy())})
    print(f"   -> Global Sim Mean: {global_sims.mean().item():.4f}")

    # 1-2. Intra-Category Similarity (같은 카테고리 내 아이템들끼리의 유사도)
    # 메모리를 위해 카테고리별로 샘플링하여 계산
    intra_sims = []
    category_to_item_ids = {} # { 'T-shirt': [1, 5, 12...], ... }
    for iid, cat in item_id_to_category.items():
        if cat not in category_to_item_ids: category_to_item_ids[cat] = []
        category_to_item_ids[cat].append(iid)
        
    for cat, ids in category_to_item_ids.items():
        if len(ids) > 10: # 아이템이 충분히 있는 카테고리만
            ids_tensor = torch.tensor(ids, device=device)
            # 카테고리 내 랜덤 200쌍 추출
            c_idx1 = ids_tensor[torch.randint(0, len(ids), (200,))]
            c_idx2 = ids_tensor[torch.randint(0, len(ids), (200,))]
            sims = (item_embs[c_idx1] * item_embs[c_idx2]).sum(dim=1)
            intra_sims.append(sims)
            
    if intra_sims:
        intra_sims_tensor = torch.cat(intra_sims)
        wandb.log({"Analysis/Intra_Category_Similarity": wandb.Histogram(intra_sims_tensor.cpu().numpy())})
        print(f"   -> Intra-Category Sim Mean: {intra_sims_tensor.mean().item():.4f}")

    # -----------------------------------------------------------
    # Phase 2. Error Analysis (R@100 성공, R@10 실패 케이스)
    # -----------------------------------------------------------
    print("🔍 [Phase 2] Extracting Error Cases (R@100 Hit, R@10 Miss)...")
    
    # WandB Table 초기화
    error_table = wandb.Table(columns=[
        "User ID", "Target Category", "Predicted Top 1-3 Categories", 
        "Top 10 Category Dist.", "Target Rank"
    ])
    
    # 점수 계산 (배치로 처리 권장하나, 분석용이므로 간단히 행렬 곱)
    scores = torch.matmul(user_embs, item_embs.T)
    _, top100_indices = torch.topk(scores, k=100, dim=-1)
    top100_preds = top100_indices.cpu().numpy()
    
    error_count = 0
    max_log_cases = 100 # WandB 테이블에 너무 많이 올라가는 것 방지
    
    for i, u_id in enumerate(tqdm(user_ids, desc="Scanning Users")):
        if error_count >= max_log_cases: break
        
        # 💡 [수정됨] NumPy/List Array의 길이를 검사하도록 변경
        if u_id not in target_dict or len(target_dict[u_id]) == 0: 
            continue
        
        # 💡 [수정됨] Target이 단일 스칼라값인지 반복 가능한 배열인지 안전하게 처리
        raw_targets = target_dict[u_id]
        if isinstance(raw_targets, str) or not hasattr(raw_targets, '__iter__'):
            raw_targets = [raw_targets]
            
        target_id = raw_targets[0] # 첫 번째 정답 아이템 기준
        target_cat = item_id_to_category.get(target_id, "Unknown")
        
        preds = top100_preds[i]
        
        if target_id in preds:
            # np.where를 안전하게 사용하기 위해 list로 변환하여 index 탐색
            rank = list(preds).index(target_id) + 1
            
            if 10 < rank <= 100:
                top10_ids = preds[:10]
                top10_cats = [item_id_to_category.get(iid, "Unknown") for iid in top10_ids]
                
                top3_str = " | ".join(top10_cats[:3])
                cat_dist = str(dict(Counter(top10_cats)))
                
                error_table.add_data(
                    str(u_id), target_cat, top3_str, cat_dist, rank
                )
                error_count += 1

    wandb.log({"Analysis/Retrieval_Error_Notes": error_table})
    print(f"   -> Logged {error_count} error cases to WandB Table.")
    print("="*60 + "\n")
    
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from tqdm import tqdm

def find_optimal_hnm_boundary_via_metadata(
    item_embs,               # [Num_Items, Dim] : 학습된 아이템 임베딩
    item_metadata_dict,      # dict: item_id -> {"MAT": [...], "CAT": [...], "DET": [...]}
    device='cuda',
    jaccard_threshold=0.75   # "이 정도 속성이 겹치면 사실상 같은 옷(FN)이다"의 기준
):
    print("\n" + "="*70)
    print("🔍 [Phase 3] Metadata-based False Negative Boundary Measurement")
    print("="*70)
    
    item_embs = F.normalize(item_embs.to(device), p=2, dim=1)
    num_items = item_embs.size(0)
    
    # 1. 아이템별 속성 Set 구축
    item_sets = {}
    for iid, meta in item_metadata_dict.items():
        # MAT, CAT, DET를 하나의 1차원 Set으로 병합
        attributes = set()
        for key in ["MAT", "CAT", "DET"]:
            if key in meta and isinstance(meta[key], list):
                # 소문자로 통일하여 텍스트 정규화
                attributes.update([str(v).strip().lower() for v in meta[key]])
        item_sets[iid] = attributes
        
    valid_ids = list(item_sets.keys())
    
    # 2. 샘플링을 통한 상관관계 분석 (VRAM 고려 2,000개 아이템만 샘플링)
    sample_size = min(2000, len(valid_ids))
    sample_ids = np.random.choice(valid_ids, sample_size, replace=False)
    sample_idx_tensor = torch.tensor(sample_ids, device=device)
    
    sample_embs = item_embs[sample_idx_tensor]
    
    # 임베딩 코사인 유사도 행렬 계산 [2000, 2000]
    latent_sim_matrix = torch.matmul(sample_embs, sample_embs.T).cpu().numpy()
    
    fn_latent_sims = [] # 자카드 유사도가 높은(FN) 쌍들의 임베딩 유사도 보관
    hn_latent_sims = [] # 자카드 유사도가 중간(HN) 쌍들의 임베딩 유사도 보관
    
    print("⏳ Calculating Jaccard Similarities and Mapping to Latent Space...")
    for i in tqdm(range(sample_size)):
        set_A = item_sets[sample_ids[i]]
        if not set_A: continue
            
        for j in range(i+1, sample_size):
            set_B = item_sets[sample_ids[j]]
            if not set_B: continue
                
            # 자카드 유사도 계산 (교집합 / 합집합)
            intersection = len(set_A.intersection(set_B))
            union = len(set_A.union(set_B))
            jaccard_sim = intersection / union if union > 0 else 0
            
            latent_sim = latent_sim_matrix[i, j]
            
            # 통계 수집
            if jaccard_sim >= jaccard_threshold:
                # 메타데이터 상 "거의 같은 옷" (False Negative)
                fn_latent_sims.append(latent_sim)
            elif 0.3 <= jaccard_sim < jaccard_threshold:
                # 카테고리 정도만 겹치는 "적절한 오답" (Hard Negative)
                hn_latent_sims.append(latent_sim)
                
    # 3. 통계 결과 출력 및 최적 Boundary 제안
    fn_sims = np.array(fn_latent_sims)
    hn_sims = np.array(hn_latent_sims)
    
    print("\n📊 [Statistical Report]")
    print(f"  - 👗 사실상 동일 아이템 (Jaccard >= {jaccard_threshold}) 쌍 개수: {len(fn_sims)}")
    print(f"  - 👖 적절한 헷갈림 오답 (0.3 <= Jaccard < {jaccard_threshold}) 쌍 개수: {len(hn_sims)}")
    
    if len(fn_sims) > 0:
        fn_mean = fn_sims.mean()
        # 하위 10% 백분위수를 경계로 삼음 (보수적 접근)
        optimal_boundary = np.percentile(fn_sims, 10) 
        
        print(f"\n💡 [Actionable Insight]")
        print(f"  - 사실상 동일한 옷들은 모델 임베딩 공간에서 평균적으로 [ {fn_mean:.4f} ] 의 코사인 유사도를 가집니다.")
        print(f"  - 따라서 HNM 진행 시, 모델 유사도가 [ {optimal_boundary:.4f} ] 이상인 아이템들은 오답으로 때리지 말고 무시(Skip)해야 합니다!")
        print(f"  - 🚀 추천 세팅: cfg.boundary_ratio = {optimal_boundary:.4f} (또는 dynamic boundary 계산 시 활용)")
        
        # WandB 로깅
        wandb.log({
            "HNM_Analysis/False_Negative_Latent_Sims": wandb.Histogram(fn_sims),
            "HNM_Analysis/Hard_Negative_Latent_Sims": wandb.Histogram(hn_sims),
            "HNM_Analysis/Suggested_Boundary": optimal_boundary
        })
    else:
        print("\n⚠️ 샘플 내에 자카드 유사도가 높은 쌍이 부족합니다. 임계값을 낮추거나 샘플을 늘려주세요.")
        
    print("="*70 + "\n")
    return optimal_boundary if len(fn_sims) > 0 else None

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# 💡 9가지 전체 필드 정의
ALL_FIELDS = ["CAT", "MAT", "FIT", "DET", "FNC", "CTX", "COL", "SPC", "LOC"]

def load_and_parse_json(json_path):
    print(f"📥 Loading JSON from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_metadata = json.load(f)
        
    item_dict = {}
    for meta in raw_metadata:
        iid = str(meta.get('article_id', meta.get('item_id', '')))
        if not iid: continue
        
        rf = meta.get("reinforced_feature", {})
        
        # 9개 필드를 동적으로 파싱
        item_dict[iid] = {
            field: [str(val).lower().strip() for val in rf.get(field, [])]
            for field in ALL_FIELDS
        }
    print(f"✅ Loaded {len(item_dict)} items.")
    return item_dict

def extract_unique_attributes(item_dict):
    unique_attrs = {field: set() for field in ALL_FIELDS}
    
    for attrs in item_dict.values():
        for field in ALL_FIELDS:
            unique_attrs[field].update(attrs[field])
            
    # 빈 문자열 제거 및 통계 출력
    print("📊 Unique Attribute Counts:")
    for field in ALL_FIELDS:
        unique_attrs[field].discard("")
        print(f"  - {field}: {len(unique_attrs[field])} unique words")
        
    return {k: list(v) for k, v in unique_attrs.items()}

def build_aspect_item_embeddings(item_dict, sbert_model_name='all-MiniLM-L6-v2', device='cuda', weights=None):
    if weights is None:
        # 💡 패션 도메인 최적화 가중치 (총합 1.0)
        # 색상(COL) 비중을 낮춰서 색상만 다른 동일 상품을 FN으로 묶어냄
        weights = {
            "CAT": 0.40, 
            "MAT": 0.20, 
            "FIT": 0.15, 
            "DET": 0.10, 
            "FNC": 0.05, 
            "CTX": 0.05, 
            "COL": 0.03, 
            "SPC": 0.01, 
            "LOC": 0.01
        }
        
    print(f"\n🧠 Loading Sentence-BERT model: {sbert_model_name}...")
    model = SentenceTransformer(sbert_model_name, device=device)
    
    unique_attrs_list = extract_unique_attributes(item_dict)
    
    # 1. 고유 단어들만 S-BERT로 임베딩 (캐싱)
    print("\n⚡ Embedding unique attributes (Super Fast!)...")
    attr_embeddings = {field: {} for field in ALL_FIELDS}
    
    for field in ALL_FIELDS:
        words = unique_attrs_list[field]
        if not words: continue
        # S-BERT 인코딩
        embs = model.encode(words, convert_to_tensor=True, show_progress_bar=False)
        for w, emb in zip(words, embs):
            attr_embeddings[field][w] = emb
            
    zero_vec = torch.zeros(384, device=device)
    
    # 2. 아이템별 최종 벡터 조립 (가중합)
    print("🧩 Assembling final item vectors based on 9 weighted aspects...")
    item_ids = list(item_dict.keys())
    final_embs = []
    
    for iid in tqdm(item_ids):
        attrs = item_dict[iid]
        v_final = torch.zeros(384, device=device)
        
        for field in ALL_FIELDS:
            # 해당 아이템이 가진 특정 필드의 단어 벡터들 수집
            field_vecs = [attr_embeddings[field][w] for w in attrs[field] if w in attr_embeddings[field]]
            
            # 단어 벡터들의 평균을 구함 (해당 필드의 값이 없다면 zero_vec)
            v_field = torch.stack(field_vecs).mean(dim=0) if field_vecs else zero_vec
            
            # 가중치를 곱해서 최종 벡터에 누적
            v_final += weights[field] * v_field
            
        final_embs.append(v_final)
        
    final_embs_tensor = torch.stack(final_embs)
    # L2 정규화
    final_embs_tensor = F.normalize(final_embs_tensor, p=2, dim=1)
    
    return item_ids, final_embs_tensor
def analyze_semantic_similarities(item_ids, embs_tensor, sample_size=5000):
    print(f"\n📏 Calculating cosine similarities for a sample of {sample_size} items...")
    # 6만 개를 통째로 $N^2$ 행렬곱 하면 RAM이 터질 수 있으므로 샘플링
    total_items = embs_tensor.size(0)
    sample_indices = np.random.choice(total_items, min(sample_size, total_items), replace=False)
    
    sample_embs = embs_tensor[sample_indices]
    # [Sample, Sample] 크기의 코사인 유사도 행렬
    sim_matrix = torch.matmul(sample_embs, sample_embs.T).cpu().numpy()
    
    # 대각선(자기 자신, 유사도 1.0)과 중복 계산 제외 (Upper triangle)
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    sim_values = sim_matrix[upper_tri_indices]
    
    # 임계치(Threshold)별 쌍(Pairs) 개수 측정
    thresholds = [0.85, 0.90, 0.95]
    print("\n📊 [Semantic Similarity Distribution]")
    for t in thresholds:
        count = np.sum(sim_values >= t)
        print(f"  - 유사도 {t:.2f} 이상인 쌍 개수: {count:,} 개")
        
    # 히스토그램 시각화 저장
    plt.figure(figsize=(10, 6))
    plt.hist(sim_values, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(0.85, color='red', linestyle='dashed', linewidth=2, label='0.85 (False Negative Zone)')
    plt.title("S-BERT Semantic Similarity Distribution of Items")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("sbert_similarity_distribution.png", dpi=300)
    print("📈 Histogram saved as 'sbert_similarity_distribution.png'")
    
    # 상위 0.1% 유사도를 기준으로 False Negative Boundary 추천
    recommended_boundary = np.percentile(sim_values, 99.9)
    print(f"\n💡 [Conclusion] S-BERT 기반 상위 0.1% 유사도 점수는 {recommended_boundary:.4f} 입니다.")
    print("이 이상의 점수를 가진 아이템 쌍을 HNM에서 강제로 배제(Masking)하는 기준값으로 활용하세요.")

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def get_or_build_aligned_sbert_embeddings(processor, base_dir, device='cuda'):
    """
    JSON을 읽어 9가지 속성 기반 S-BERT 임베딩을 생성하고, 
    processor.item_ids의 순서(1-based)에 맞춰 정렬된 텐서(N+1, 384)를 반환합니다.
    최초 1회 실행 시 .pt 파일로 저장하여 이후에는 빠르게 로드합니다.
    """
    cache_path = os.path.join(base_dir, "aligned_sbert_embs.pt")
    
    # 💡 1. 캐시 파일이 있으면 즉시 로드 (학습 속도 저하 방지)
    if os.path.exists(cache_path):
        print(f"♻️ [Phase 3-SBERT] Loading cached aligned S-BERT embeddings from {cache_path}...")
        return torch.load(cache_path, map_location=device)
        
    print("\n🧠 [Phase 3-SBERT] Building Aligned S-BERT Semantic Vectors (First Time Only)...")
    json_path = os.path.join(base_dir, "filtered_data_reinforced.json")
    
    # 1. JSON 로드 및 매핑 딕셔너리 생성
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load JSON: {e}")
        
    ALL_FIELDS = ["CAT", "MAT", "FIT", "DET", "FNC", "CTX", "COL", "SPC", "LOC"]
    metadata_dict = {}
    for meta in raw_metadata:
        iid = str(meta.get('article_id', meta.get('item_id', '')))
        if iid:
            rf = meta.get("reinforced_feature", {})
            metadata_dict[iid] = {f: [str(v).lower().strip() for v in rf.get(f, [])] for f in ALL_FIELDS}
            
    # 2. 고유 속성 추출 및 단 1번씩만 인코딩 (초고속화)
    unique_attrs = {f: set() for f in ALL_FIELDS}
    for attrs in metadata_dict.values():
        for f in ALL_FIELDS:
            unique_attrs[f].update(attrs[f])
            
    for f in ALL_FIELDS: unique_attrs[f].discard("")

    print("⚡ Encoding unique attributes with S-BERT...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    attr_embs = {f: {} for f in ALL_FIELDS}
    
    for field in ALL_FIELDS:
        words = list(unique_attrs[field])
        if not words: continue
        embs = model.encode(words, convert_to_tensor=True, show_progress_bar=False)
        for w, emb in zip(words, embs):
            attr_embs[field][w] = emb
            
    # 3. processor.item_ids 순서에 맞춰 (N+1, 384) 텐서 조립
    num_items = processor.num_items + 1
    # S-BERT all-MiniLM-L6-v2의 차원은 384
    aligned_sbert_arr = torch.zeros((num_items, 384), device=device)
    
    weights = {"CAT": 0.40, "MAT": 0.20, "FIT": 0.15, "DET": 0.10, 
               "FNC": 0.05, "CTX": 0.05, "COL": 0.03, "SPC": 0.01, "LOC": 0.01}
    zero_vec = torch.zeros(384, device=device)
    
    matched = 0
    print(f"🧩 Aligning to processor items (1 to {processor.num_items})...")
    for i, current_id_str in enumerate(tqdm(processor.item_ids)):
        idx = i + 1 # 1-based index (0은 zero vector로 유지)
        
        if current_id_str in metadata_dict:
            attrs = metadata_dict[current_id_str]
            v_final = torch.zeros(384, device=device)
            
            for field in ALL_FIELDS:
                vecs = [attr_embs[field][w] for w in attrs[field] if w in attr_embs[field]]
                v_field = torch.stack(vecs).mean(dim=0) if vecs else zero_vec
                v_final += weights[field] * v_field
                
            aligned_sbert_arr[idx] = v_final
            matched += 1
            
    # L2 정규화 (코사인 유사도 연산용)
    aligned_sbert_arr = F.normalize(aligned_sbert_arr, p=2, dim=1)
    
    print(f"✅ S-BERT Vectors Aligned: {matched}/{len(processor.item_ids)}")
    
    # 메모리 해제 및 캐싱
    del model, attr_embs, raw_metadata, metadata_dict
    torch.cuda.empty_cache()
    
    torch.save(aligned_sbert_arr, cache_path)
    print(f"💾 Saved aligned S-BERT embeddings to {cache_path}")
    
    return aligned_sbert_arr

def analysis_model_and_vectors():
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
    cfg.lr = 1e-3               
    cfg.epochs = 30       
    cfg.HN_K = 150 
    cfg.EX_TOP_K = 10
    cfg.soft_penalty_weigh = 1
    cfg.batch_size = 512
    cfg.hn_scheduled = False
    cfg.dropout = 0.25
    cfg.boubdary_ratio = 0.90
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
        name=f"analysis_model_SessionWeight_lr{cfg.lr}_K{cfg.HN_K}", 
        config=cfg.__dict__ 
    )
    
    # -----------------------------------------------------------
    # 3. Models Setup & 💡 Epoch 11 베이스라인 가중치 로드
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, None, log_q_tensor) 
    
    # 💡 [요청 사항 1] 모델 경로 지정 및 로드
    base_user_pth = os.path.join(cfg.model_dir, "best_user_tower_from_scratch_base_feature.pth")
    base_item_pth = os.path.join(cfg.model_dir, "best_item_tower_from_scratch_base_feature.pth")
    
    save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_0312_session_v4_hm.pth")
    save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_0312_session_v4_hm.pth")

    item_tower.init_from_pretrained(aligned_vecs.to(device))
    
    print(f"📥 Loading Baseline Models from Epoch 11...")
    user_tower.load_state_dict(torch.load(base_user_pth, map_location=device))
    item_tower.load_state_dict(torch.load(base_item_pth, map_location=device))
    print(f"✅ Baseline Models loaded successfully.")
    
    
    # =====================================================================
    # 💡 [신규 삽입] Phase 1~3: 훈련 전 임베딩 지형 및 메타데이터 기반 HNM 경계 분석
    # =====================================================================
    print("\n" + "="*70)
    print("🔬 [Pre-Training Analysis] Running Topology & Boundary Diagnostics")
    print("="*70)
    
    item_tower.eval()
    user_tower.eval()
    
    with torch.no_grad():
        # 1. 전체 아이템 임베딩 추출
        all_item_embs = item_tower.get_all_embeddings()
        
        # 2. 메타데이터 JSON 로드 및 딕셔너리 변환 (경로는 환경에 맞게 수정)
        import json
        metadata_json_path = os.path.join(cfg.base_dir, "filtered_data_reinforced.json")
        with open(metadata_json_path, 'r', encoding='utf-8') as f:
            raw_metadata = json.load(f)
            
        # 함수 요구사항에 맞게 딕셔너리 매핑
        # (주의: raw_metadata의 키가 string 형태의 item_id라고 가정, processor.item2id로 변환)
        item_metadata_dict = {}
        item_id_to_category = {}
        
        # 💡 [수정됨] raw_metadata가 리스트(List) 형태이므로 바로 순회합니다.
        for meta in raw_metadata:
            # ⚠️ 주의: JSON 내에서 아이템 ID를 나타내는 키를 확인해서 맞춰주세요!
            # (보통 H&M 데이터는 'article_id'를 많이 쓰며, 일반적으론 'item_id'를 씁니다)
            str_iid = str(meta.get('article_id', meta.get('item_id', ''))) 
            
            if not str_iid:
                continue # ID가 없는 잘못된 데이터는 스킵
                
            if str_iid in processor.item2id:
                encoded_iid = processor.item2id[str_iid]
                
                # Phase 3 용 (MAT, CAT, DET)
                item_metadata_dict[encoded_iid] = meta.get("reinforced_feature", {})
                
                # Phase 1, 2 용 (단일 카테고리 추출)
                cat_list = meta.get("reinforced_feature", {}).get("CAT", ["Unknown"])
                item_id_to_category[encoded_iid] = cat_list[0] if cat_list else "Unknown"
        # -----------------------------------------------------------------
        # [실행 A] Phase 3: 메타데이터 자카드 유사도 기반 최적 Boundary 추출
        # -----------------------------------------------------------------
        optimal_boundary = find_optimal_hnm_boundary_via_metadata(
            item_embs=all_item_embs,
            item_metadata_dict=item_metadata_dict,
            device=device,
            jaccard_threshold=0.75 # 75% 이상 속성이 같으면 FN으로 간주
        )
        
        # 💡 [핵심] 통계적으로 도출된 경계값을 파이프라인 cfg에 동적 덮어쓰기!
        if optimal_boundary is not None:
            # 보수적으로 살짝 낮춰서(예: 0.02) 여유를 줍니다.
            applied_boundary = max(0.80, optimal_boundary - 0.02) 
            print(f"🔄 Updating cfg.boundary_ratio: {cfg.boubdary_ratio} -> {applied_boundary:.4f}")
            cfg.boubdary_ratio = applied_boundary # 훈련 루프에서 이 값을 사용하게 됨
            wandb.config.update({"dynamic_boundary_ratio": applied_boundary}, allow_val_change=True)

        # -----------------------------------------------------------------
        # [실행 B] Phase 1 & 2: R@100 Hit & R@10 Miss 오답 노트 분석
        # -----------------------------------------------------------------
        # 분석을 위해 검증 데이터셋에서 샘플 유저 벡터 추출 (시간 절약을 위해 10 배치만 추출)
        print("⏳ Extracting sample user embeddings for error analysis...")
        sample_user_embs = []
        sample_user_ids = []
        
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= 10: break # 샘플링 10배치
            u_ids = batch['user_ids']
            padding_mask = batch['padding_mask'].to(device)
            
            forward_kwargs_eval = {
                'pretrained_vecs': val_loader.dataset.pretrained_lookup[batch['item_ids'].cpu()].to(device),
                'item_ids': batch['item_ids'].to(device),
                'time_bucket_ids': batch['time_bucket_ids'].to(device),
                'type_ids': batch['type_ids'].to(device), 'color_ids': batch['color_ids'].to(device),
                'graphic_ids': batch['graphic_ids'].to(device), 'section_ids': batch['section_ids'].to(device),
                'age_bucket': batch['age_bucket'].to(device), 'price_bucket': batch['price_bucket'].to(device),
                'cnt_bucket': batch['cnt_bucket'].to(device), 'recency_bucket': batch['recency_bucket'].to(device),
                'channel_ids': batch['channel_ids'].to(device), 'club_status_ids': batch['club_status_ids'].to(device),
                'news_freq_ids': batch['news_freq_ids'].to(device), 'fn_ids': batch['fn_ids'].to(device),
                'active_ids': batch['active_ids'].to(device), 'cont_feats': batch['cont_feats'].to(device),
                'recency_offset': batch['recency_offset'].to(device), 'current_week': batch['current_week'].to(device), 
                'target_week': batch['target_week'].to(device), 'session_ids': batch['session_ids'].to(device),
                'padding_mask': padding_mask,
                'training_mode': False
            }
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                out = user_tower(**forward_kwargs_eval)
            
            # 💡 [핵심 수정] evaluate_model과 동일한 안전한 차원 추출 로직
            if out.dim() == 3:
                lengths = (~padding_mask).sum(dim=1)
                last_idx = (lengths - 1).clamp(min=0)
                batch_range = torch.arange(out.size(0), device=device)
                u_emb = out[batch_range, last_idx]
            else:
                u_emb = out
                
            # 혹시라도 1D로 풀리는 것을 방지
            if u_emb.dim() == 1:
                u_emb = u_emb.unsqueeze(0)
                
            sample_user_embs.append(u_emb)
            sample_user_ids.extend(u_ids)

        # 이제 안전하게 [Total_Samples, Dim] 형태의 2D 텐서로 결합됩니다.
        sample_user_embs = torch.cat(sample_user_embs, dim=0)
        
        target_df = pd.read_parquet(TARGET_VAL_PATH)
        target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
        del target_df

        run_diagnostic_analysis(
            user_embs=sample_user_embs,
            item_embs=all_item_embs,
            target_dict=target_dict,
            user_ids=sample_user_ids,
            item_id_to_category=item_id_to_category,
            device=device
        )
        
        del sample_user_embs, all_item_embs
        torch.cuda.empty_cache()

    # =====================================================================
    # (종료) 다시 정상 학습 파이프라인으로 복귀
    # =====================================================================

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
    cfg.lr = 2e-4               
    cfg.epochs = 60      
    cfg.HN_K = 150 
    cfg.EX_TOP_K = 3
    cfg.soft_penalty_weigh = 1
    cfg.batch_size = 512
    cfg.hn_scheduled = False
    cfg.dropout = 0.25
    cfg.boubdary_ratio = 0.90
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
    
    aligned_sbert_embs =get_or_build_aligned_sbert_embeddings(processor, cfg.base_dir, device)
    
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
    base_user_pth = os.path.join(cfg.model_dir, "best_user_tower_0312_session_v4_hm.pth")
    base_item_pth = os.path.join(cfg.model_dir, "best_item_tower_0312_session_v4_hm.pth")
    
    save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_0312_session_v4_hm_r1.pth")
    save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_0312_session_v4_hm_r1.pth")

    item_tower.init_from_pretrained(aligned_vecs.to(device))
    
    print(f"📥 Loading Baseline Models from Epoch 11...")
    user_tower.load_state_dict(torch.load(base_user_pth, map_location=device))
    item_tower.load_state_dict(torch.load(base_item_pth, map_location=device))
    print(f"✅ Baseline Models loaded successfully.")
    
    
    print("🔥 Epoch 12 (Resume): Item Tower is UNFROZEN from the start!")
    item_tower.set_freeze_state(False)
    item_finetune_lr = cfg.lr * 0.15
    
    # Optimizer에 두 타워를 동시에 등록 (LR 비대칭)
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': cfg.lr},
        {'params': item_tower.parameters(), 'lr': item_finetune_lr}
    ], weight_decay=cfg.weight_decay, fused=True)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.001) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01) 

    
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
    
    current_beta = 0.20
    for epoch in range(start_epoch, end_epoch):
        
        
        item_tower.eval()
        with torch.no_grad():
            print(f"📦 [Epoch {epoch}] Caching All Item Embeddings for Training...")
            all_item_embs = item_tower.get_all_embeddings()
            norm_item_embeddings = F.normalize(all_item_embs, p=2, dim=1) # [50000, Dim]
            
            # 10에포크부터는 이 캐시된 임베딩을 마이닝에도 재사용하여 속도 극대화
            if epoch >= 14:
                if (epoch - 10) % 1 == 0 or force_mining_next_epoch:
                    print(f"🔍 Mining Global Hard Negatives using cached embeddings...")
                    epoch_hn_pool, hn_metrics = mine_global_hard_negatives(
                        item_embs=norm_item_embeddings,    # 다시 계산 안 하고 캐시본 사용
                        sbert_embs=aligned_sbert_embs,     # 💡 [수정 사항 2] 시맨틱 방어막 텐서 주입
                        fn_threshold=0.85,
                        fn_lower=0.50, # 💡 [수정 사항 3] 시맨틱 유사도 0.85 이상은 오답 풀에서 영구 배제
                        exclusion_top_k=cfg.EX_TOP_K,      # (안전지대)
                        mine_k=200, 
                        batch_size=2048, 
                        device=device
                    )
                    wandb.log(hn_metrics)
                    #epoch_hn_pool = mine_global_hard_negatives(
                    #    norm_item_embeddings, # 다시 계산 안 하고 캐시본 사용
                    #    exclusion_top_k=cfg.EX_TOP_K, 
                    #    mine_k=200, 
                    #    batch_size=2048, 
                    #    device=device
                    #)
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
        
        
        T_sample = max(0.2, 0.50 - (int((epoch - start_epoch)/2)) * 0.015)
        # ------------------- 훈련 (Train) -------------------
        avg_loss, force_mining_next_epoch, avg_discard_ratio = train_user_tower_session_sampler_with_intent_point(
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
            T_sample =  T_sample,
            beta= current_beta,
            hn_refresh_interval=0,      # N배치마다 pool 재갱신
            hn_exclusion_top_k=5,         # mining 파라미터 직접 전달
            hn_mine_k=200,
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        del norm_item_embeddings
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        #prev_ex_top_k = hn_scheduler.ex_top_k if hn_scheduler else cfg.EX_TOP_K
        
        # Beta curriculum
        #if avg_discard_ratio < 0.25:
        #    current_beta = 0.15
        #elif avg_discard_ratio < 0.40:
        #    current_beta = 0.20
        #else:
        #    current_beta = 0.25

        print(f"📊 [Epoch {epoch}] avg_discard_ratio={avg_discard_ratio:.3f} "
              f"→ next beta={current_beta:.2f}, T_sample={T_sample:.3f}")
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
    
    SEQ_LABELS = [ 'recency_curr', 'week_curr', 'item_id','item_price', 'target_week']
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
    cfg.dropout = 0.25
    FREEZE_ITEM_EPOCHS = 1
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
    
    save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_from_scratch_session_last_feature.pth")
    save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_from_scratch_session_last_feature.pth")
    item_tower.init_from_pretrained(aligned_vecs.to(device))
    print(f"❄️ Item Tower is Frozen for the first {FREEZE_ITEM_EPOCHS} epochs.")
    item_tower.set_freeze_state(True)
    

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
    current_beta = 0.15
    # -----------------------------------------------------------
    # 4. Training Loop (Epoch 1부터 시작)
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        
        
        if epoch == FREEZE_ITEM_EPOCHS + 1:
            print(f"🔥 Epoch {epoch}: Unfreezing Item Tower for Joint Training!")
            item_tower.set_freeze_state(False)
        
        
        if epoch == 10:
            print(f"📈 [Epoch {epoch}] Increasing Item Tower LR multiplier from 0.05x to 0.15x!")
            new_item_base_lr = cfg.lr * 0.15
            
            # param_groups[0]: user_tower / param_groups[1]: item_tower
            optimizer.param_groups[1]['initial_lr'] = new_item_base_lr
            
            # 스케줄러가 계산의 기준으로 삼는 base_lrs도 업데이트 (매우 중요!)
            if hasattr(scheduler, 'base_lrs'):
                scheduler.base_lrs[1] = new_item_base_lr
        
        
        item_tower.eval()
        
 
        with torch.no_grad():
            print(f"📦 [Epoch {epoch}] Caching All Item Embeddings for Training...")
            all_item_embs = item_tower.get_all_embeddings()
            norm_item_embeddings = F.normalize(all_item_embs, p=2, dim=1) # [50000, Dim]
            
            # 10에포크부터는 이 캐시된 임베딩을 마이닝에도 재사용하여 속도 극대화
            if epoch >= 14:
                if (epoch - 10) % 1 == 0 or force_mining_next_epoch:
                    print(f"🔍 Mining Global Hard Negatives using cached embeddings...")
                    epoch_hn_pool = mine_global_hard_negatives(
                        norm_item_embeddings, # 다시 계산 안 하고 캐시본 사용
                        exclusion_top_k=5 ,
                        mine_k=200, 
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
        
     
        
        T_sample = max(0.25, 0.5 - (epoch - 14) * 0.025)
        # ------------------- 훈련 (Train) -------------------
        avg_loss, force_mining_next_epoch, avg_discard_ratio = train_user_tower_session_sampler_with_intent_point(
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
            T_sample =  T_sample,
            beta= current_beta,
            hn_refresh_interval=0,      # N배치마다 pool 재갱신
            hn_exclusion_top_k=2,         # mining 파라미터 직접 전달
            hn_mine_k=150,
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        del norm_item_embeddings
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        #prev_ex_top_k = hn_scheduler.ex_top_k if hn_scheduler else cfg.EX_TOP_K
        
        # Beta curriculum
        if avg_discard_ratio < 0.25:
            current_beta = 0.15
        elif avg_discard_ratio < 0.40:
            current_beta = 0.20
        else:
            current_beta = 0.25

        print(f"📊 [Epoch {epoch}] avg_discard_ratio={avg_discard_ratio:.3f} "
              f"→ next beta={current_beta:.2f}, T_sample={T_sample:.3f}")
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
        if epoch == 13:
            print(f"🔔 [Checkpoint] Epoch {epoch} completed. Current Recall@20: {current_recall_20:.2f}%")
            base_user_pth =os.path.join(cfg.model_dir, "best_user_tower_from_scratch_base_feature.pth")
            base_item_pth= os.path.join(cfg.model_dir, "best_item_tower_from_scratch_base_feature.pth")
            
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
    import torch.multiprocessing as mp
    #mp.freeze_support()
    mp.set_start_method('spawn', force=True)  # Windows 필수
    #analysis_model_and_vectors()
    #cfg = PipelineConfig()
    #JSON_PATH = os.path.join(cfg.base_dir, "filtered_data_reinforced.json")
    #item_dict = load_and_parse_json(JSON_PATH)
    
    # 함수 호출 시 기본 세팅된 9가지 가중치가 자동 적용됩니다.
    #ids, embs = build_aspect_item_embeddings(item_dict)
    
    #analyze_semantic_similarities(ids, embs, sample_size=5000)
    resume_pipeline_session_weights()
    #run_resume_pipeline_v2()
    #train_pipeline_from_scratch()