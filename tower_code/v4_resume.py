
import sys

import wandb
import os
import torch
import torch.nn.functional as F

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
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
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from tqdm import tqdm
import wandb

from tower_code.params_config import PipelineConfig
from tower_code.sheduler import BidirectionalHNScheduler

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


# ================================================================
# Unsupervised CL Loss (가중치 적용 버전)
# ================================================================
def unsupervised_cl_loss(emb_v1, emb_v2, temperature=0.1, weights=None):
    """
    emb_v1: (N, D) - 원본 셔플 세션 마지막 벡터
    emb_v2: (N, D) - 다른 셔플 세션 마지막 벡터
    weights: (N,) - Time-decay 등 샘플별 중요도 가중치
    """
    N      = emb_v1.size(0)
    device = emb_v1.device

    if N < 2:
        return torch.tensor(0.0, device=device)

    v1 = F.normalize(emb_v1, p=2, dim=1)
    v2 = F.normalize(emb_v2, p=2, dim=1)

    # [v1_0..v1_N, v2_0..v2_N] concat
    all_emb    = torch.cat([v1, v2], dim=0)                            # (2N, D)
    sim_matrix = torch.matmul(all_emb, all_emb.T) / temperature        # (2N, 2N)

    # 자기자신 마스킹
    eye_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(eye_mask, -1e4)

    # positive label: v1_i ↔ v2_i
    labels = torch.cat([
        torch.arange(N, 2 * N, device=device),   # v1_i의 positive = v2_i
        torch.arange(0, N,     device=device),   # v2_i의 positive = v1_i
    ], dim=0)                                                          # (2N,)

    # 💡 [핵심] reduction='none'으로 설정하여 샘플(2N)별 Loss를 추출
    loss_per_sample = F.cross_entropy(sim_matrix, labels, reduction='none') # (2N,)

    # 가중치 적용 로직
    if weights is not None:
        weights_2n = torch.cat([weights, weights], dim=0) # v1, v2용 가중치 이어붙이기 (2N,)
        weight_sum = weights_2n.sum() + 1e-9
        loss = (loss_per_sample * weights_2n).sum() / weight_sum
    else:
        loss = loss_per_sample.mean()

    return loss
 
# ================================================================
# Supervised CL Loss (가중치 적용 버전)
# ================================================================
def supervised_cl_loss(session_vecs, session_user_ids, temperature=0.1, weights=None):
    """
    session_vecs:     (N, D) 세션별 마지막 벡터
    session_user_ids: (N,)   각 세션의 배치 내 행 인덱스
    weights:          (N,)   샘플별 가중치
    """
    N      = session_vecs.size(0)
    device = session_vecs.device

    if N < 2:
        return torch.tensor(0.0, device=device), {}

    vecs       = F.normalize(session_vecs, p=2, dim=1)
    sim_matrix = torch.matmul(vecs, vecs.T) / temperature              # (N, N)

    eye_mask   = torch.eye(N, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(eye_mask, -1e4)

    # same-user positive mask (자기자신 제외)
    same_user  = (
        session_user_ids.unsqueeze(0) == session_user_ids.unsqueeze(1)
    ) & ~eye_mask                                                      # (N, N)

    has_positive = same_user.any(dim=1)                                # (N,)

    cl_metrics = {
        'cl/sup_has_peer_ratio': has_positive.float().mean().item(),
        'cl/sup_avg_peers':      same_user.float().sum(dim=1).mean().item(),
    }

    if not has_positive.any():
        return torch.tensor(0.0, device=device), cl_metrics

    # 유효한(peer가 있는) 샘플만 추출 -> 크기가 M으로 줄어듦
    sim_active  = sim_matrix[has_positive]                             # (M, N)
    same_active = same_user[has_positive]                              # (M, N)

    log_prob      = F.log_softmax(sim_active, dim=1)
    pos_log_prob  = (log_prob * same_active.float()).sum(dim=1)
    num_positives = same_active.float().sum(dim=1).clamp(min=1)

    # 💡 [핵심] 평균(mean)을 구하지 않고 샘플별(M개) Loss로 보존
    loss_per_sample = -(pos_log_prob / num_positives)                  # (M,)

    # 가중치 적용 로직
    if weights is not None:
        active_weights = weights[has_positive] # 유효한 샘플의 가중치만 추출 (M,)
        weight_sum = active_weights.sum() + 1e-9
        loss = (loss_per_sample * active_weights).sum() / weight_sum
    else:
        loss = loss_per_sample.mean()

    return loss, cl_metrics
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
 
# ================================================================
# SASRecDataset_v4
# v3 대비 유일한 변경: indices_v2 (두 번째 셔플) 생성 및 반환
# ================================================================
class SASRecDataset_v4(Dataset):
    """
    v3 대비 변경:
      - __getitem__에서 view2용 셔플 시퀀스(indices_v2) 생성
      - 반환 dict에 v2 관련 키 추가
        (item_ids_v2, target_ids_v2, time_bucket_ids_v2, ...)
      - v2는 v1과 동일한 세션 구성이지만 독립적으로 셔플된 순서
    """
 
    def __init__(self, processor, global_now_str="2020-09-22", max_len=30, is_train=True):
        self.processor    = processor
        self.max_len      = max_len
        self.is_train     = is_train
        self.user_ids     = processor.user_ids
 
        global_now_dt     = pd.to_datetime(global_now_str)
        self.now_ordinal  = global_now_dt.toordinal()
        self.now_week     = global_now_dt.isocalendar().week
 
        if self.is_train:
            self.verify_session_logic()
 
    def verify_session_logic(self):
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
 
        u_mapped_id     = self.processor.user2id[sample_user]
        seq_raw         = self.processor.u_seqs[u_mapped_id]
        time_deltas_raw = self.processor.u_deltas[u_mapped_id]
 
        session_ids_raw = [1]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == time_deltas_raw[i-1]:
                session_ids_raw.append(session_ids_raw[-1])
            else:
                session_ids_raw.append(session_ids_raw[-1] + 1)
 
        input_indices   = list(range(len(seq_raw)))
        grouped_indices = []
        current_group   = [input_indices[0]]
        for i in range(1, len(input_indices)):
            if time_deltas_raw[input_indices[i]] == time_deltas_raw[input_indices[i-1]]:
                current_group.append(input_indices[i])
            else:
                grouped_indices.append(current_group)
                current_group = [input_indices[i]]
        if current_group:
            grouped_indices.append(current_group)
 
        shuffled_v1, shuffled_v2 = [], []
        for group in grouped_indices:
            g1 = group.copy()
            g2 = group.copy()
            if len(g1) > 1:
                random.shuffle(g1)
                random.shuffle(g2)
            shuffled_v1.extend(g1)
            shuffled_v2.extend(g2)
 
        print(f"👤 Sample User ID: {sample_user}")
        print(f"{'Pos':<5} | {'Item(v1)':<11} | {'Session':<9} | {'Idx_v1':<8} | {'Item(v2)':<11} | {'Idx_v2':<8}")
        print("-" * 65)
        for i in range(len(seq_raw)):
            i1, i2 = shuffled_v1[i], shuffled_v2[i]
            print(f"{i:<5} | {seq_raw[i1]:<11} | {session_ids_raw[i1]:<9} | {i1:<8} | {seq_raw[i2]:<11} | {i2:<8}")
        print("="*75 + "\n")
 
    def _shuffle_indices_within_session(self, indices, time_deltas_raw):
        """세션 내부 순서를 50% 확률로 셔플 (세션 경계는 유지)"""
        if len(indices) <= 1:
            return indices
 
        grouped_indices = []
        current_group   = [indices[0]]
        for i in range(1, len(indices)):
            idx      = indices[i]
            prev_idx = indices[i-1]
            if time_deltas_raw[idx] == time_deltas_raw[prev_idx]:
                current_group.append(idx)
            else:
                grouped_indices.append(current_group)
                current_group = [idx]
        if current_group:
            grouped_indices.append(current_group)
 
        shuffled = []
        for group in grouped_indices:
            if len(group) > 1 and random.random() < 0.5:
                random.shuffle(group)
            shuffled.extend(group)
        return shuffled
 
    def _build_view(self, shuffled_indices, seq_mapped, time_buckets,
                    session_ids_raw, time_deltas_raw, u_mapped_id, is_v2=False):
        """
        shuffled_indices로부터 view 데이터 딕셔너리 생성
        v1, v2 공통 로직을 하나로 묶어 중복 제거
        """
        shuffled_indices = shuffled_indices[-self.max_len:]
        pad_len          = self.max_len - len(shuffled_indices)
 
        input_seq     = [seq_mapped[i]      for i in shuffled_indices]
        input_time    = [time_buckets[i]    for i in shuffled_indices]
        input_session = [session_ids_raw[i] for i in shuffled_indices]
        target_seq    = [seq_mapped[i + 1]  for i in shuffled_indices]
 
        d_buckets   = self.processor.u_dyn_buckets[u_mapped_id][shuffled_indices]
        d_conts     = self.processor.u_dyn_conts[u_mapped_id][shuffled_indices]
        d_cats      = self.processor.u_dyn_cats[u_mapped_id][shuffled_indices]
        d_time      = self.processor.u_dyn_time[u_mapped_id][shuffled_indices]
        input_dates = d_time[:, 0].tolist()
 
        target_indices    = [i + 1 for i in shuffled_indices]
        step_target_times = self.processor.u_dyn_time[u_mapped_id][target_indices, 0]
        step_target_weeks = self.processor.u_dyn_time[u_mapped_id][target_indices, 1]
        dynamic_offsets   = np.clip(step_target_times - d_time[:, 0], 0, 365).astype(np.int64)
        current_weeks     = d_time[:, 1]
 
        # 패딩
        input_padded       = [0] * pad_len + input_seq
        time_padded        = [0] * pad_len + input_time
        target_padded      = [0] * pad_len + target_seq
        padding_mask       = [True]  * pad_len + [False] * len(input_seq)
        session_padded     = [0]     * pad_len + input_session
        target_week_padded = [0]     * pad_len + step_target_weeks.tolist()
        dates_padded       = [0]     * pad_len + input_dates
 
        item_side_info = self.processor.i_side_arr[input_padded]
 
        pad_b   = np.zeros((pad_len, 3), dtype=np.int64)
        pad_c   = np.zeros((pad_len, 5), dtype=np.float32)
        pad_cat = np.zeros((pad_len, 1), dtype=np.int64)
        pad_1d  = np.zeros(pad_len, dtype=np.int64)
 
        d_buckets_p = np.vstack([pad_b,   d_buckets]) if len(shuffled_indices) > 0 else pad_b
        d_conts_p   = np.vstack([pad_c,   d_conts])   if len(shuffled_indices) > 0 else pad_c
        d_cats_p    = np.vstack([pad_cat, d_cats])     if len(shuffled_indices) > 0 else pad_cat
        offset_p    = np.concatenate([pad_1d, dynamic_offsets]) if len(shuffled_indices) > 0 else pad_1d
        week_p      = np.concatenate([pad_1d, current_weeks])   if len(shuffled_indices) > 0 else pad_1d
 
        suffix = '_v2' if is_v2 else ''
        return {
            f'item_ids{suffix}':          torch.tensor(input_padded,       dtype=torch.long),
            f'target_ids{suffix}':        torch.tensor(target_padded,      dtype=torch.long),
            f'padding_mask{suffix}':      torch.tensor(padding_mask,       dtype=torch.bool),
            f'time_bucket_ids{suffix}':   torch.tensor(time_padded,        dtype=torch.long),
            f'session_ids{suffix}':       torch.tensor(session_padded,     dtype=torch.long),
            f'type_ids{suffix}':          torch.tensor(item_side_info[:, 0], dtype=torch.long),
            f'color_ids{suffix}':         torch.tensor(item_side_info[:, 1], dtype=torch.long),
            f'graphic_ids{suffix}':       torch.tensor(item_side_info[:, 2], dtype=torch.long),
            f'section_ids{suffix}':       torch.tensor(item_side_info[:, 3], dtype=torch.long),
            f'price_bucket{suffix}':      torch.tensor(d_buckets_p[:, 0],  dtype=torch.long),
            f'cnt_bucket{suffix}':        torch.tensor(d_buckets_p[:, 1],  dtype=torch.long),
            f'recency_bucket{suffix}':    torch.tensor(d_buckets_p[:, 2],  dtype=torch.long),
            f'cont_feats{suffix}':        torch.tensor(d_conts_p,          dtype=torch.float32),
            f'channel_ids{suffix}':       torch.tensor(d_cats_p[:, 0],     dtype=torch.long),
            f'recency_offset{suffix}':    torch.tensor(offset_p,           dtype=torch.long),
            f'current_week{suffix}':      torch.tensor(week_p,             dtype=torch.long),
            f'target_week{suffix}':       torch.tensor(target_week_padded, dtype=torch.long),
            f'interaction_dates{suffix}': torch.tensor(dates_padded,       dtype=torch.long),
        }
 
    def __len__(self):
        return len(self.user_ids)
 
    def __getitem__(self, idx):
        user_id     = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
 
        seq_raw         = self.processor.u_seqs[u_mapped_id]
        time_deltas_raw = self.processor.u_deltas[u_mapped_id]
 
        # 세션 ID 생성
        session_ids_raw = [1]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == time_deltas_raw[i-1]:
                session_ids_raw.append(session_ids_raw[-1])
            else:
                session_ids_raw.append(session_ids_raw[-1] + 1)
 
        seq_mapped   = [self.processor.item2id.get(iid, 0) for iid in seq_raw]
        bins         = np.array([0, 3, 7, 14, 30, 60, 180, 330, 395])
        time_buckets = np.digitize(time_deltas_raw, bins, right=False).tolist()
 
        # 정적 피처
        s_buckets = self.processor.u_static_buckets[u_mapped_id]
        s_cats    = self.processor.u_static_cats[u_mapped_id]
        static_feats = {
            'user_ids':       user_id,
            'age_bucket':     torch.tensor(s_buckets[0], dtype=torch.long),
            'club_status_ids': torch.tensor(s_cats[0],  dtype=torch.long),
            'news_freq_ids':  torch.tensor(s_cats[1],   dtype=torch.long),
            'fn_ids':         torch.tensor(s_cats[2],   dtype=torch.long),
            'active_ids':     torch.tensor(s_cats[3],   dtype=torch.long),
        }
 
        if self.is_train:
            input_indices = list(range(len(seq_raw) - 1))
 
            # ── View 1: 기존 셔플 ─────────────────────────────────
            indices_v1 = self._shuffle_indices_within_session(
                input_indices, time_deltas_raw
            )
 
            # ── View 2: 독립적인 두 번째 셔플 (Unsupervised CL용) ─
            indices_v2 = self._shuffle_indices_within_session(
                input_indices, time_deltas_raw
            )
 
            view1 = self._build_view(
                indices_v1, seq_mapped, time_buckets,
                session_ids_raw, time_deltas_raw, u_mapped_id, is_v2=False
            )
            view2 = self._build_view(
                indices_v2, seq_mapped, time_buckets,
                session_ids_raw, time_deltas_raw, u_mapped_id, is_v2=True
            )
 
            return {**static_feats, **view1, **view2}
 
        else:
            # 추론: view2 불필요, 기존 방식 그대로
            indices = list(range(len(seq_raw)))
            view1   = self._build_view(
                indices, seq_mapped, time_buckets,
                session_ids_raw, time_deltas_raw, u_mapped_id, is_v2=False
            )
            return {**static_feats, **view1}
        
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            shuffled_indices = input_indices
            #shuffled_indices = self._shuffle_indices_within_session(input_indices, time_deltas_raw)
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
         
# ================================================================
# StreamFusionGate (SENet 스타일 차원별 융합)
# ================================================================
class StreamFusionGate(nn.Module):
    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.d_model = d_model
        
        # 💡 [핵심 1] 입력 차원을 d_model * 2로 확장
        self.se = nn.Sequential(
            nn.Linear(d_model * 2, d_model // reduction),    # squeeze
            nn.LayerNorm(d_model // reduction),
            nn.ReLU(),
            nn.Linear(d_model // reduction, d_model * 2),  # excite (두 스트림 각각에 할당)
            nn.Sigmoid()
        )

    def forward(self, h_seq, h_static):
        # 💡 [핵심 2] 덧셈(+)으로 뭉개지 않고, 나란히 붙여서(Concat) 게이트에 전달
        combined = torch.cat([h_seq, h_static], dim=-1)        # (B, S, D*2)
        
        gate     = self.se(combined)                           # (B, S, D*2)
        g_seq    = gate[:, :, :self.d_model]                   # (B, S, D)
        g_static = gate[:, :, self.d_model:]                   # (B, S, D)
        
        return g_seq * h_seq + g_static * h_static             # (B, S, D)


# ================================================================
# ContinuousFeatureMLP
# asof 4개 피처를 함께 투영하여 피처 간 상호작용 학습
# 기존 ContinuousFeatureTokenizer(독립 투영) 대체
# ================================================================
class ContinuousFeatureMLP(nn.Module):
    """
    asof 연속 피처(price_std, last_price_diff, repurchase, weekend 등)
    → 1차: 피처 간 상호작용 학습 (hidden_dim)
    → 2차: 이산형 피처와의 덧셈(+) 결합을 위한 체급 팽창 (d_model)
    """
    def __init__(self, num_asof_feats: int, d_model: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            # ── 1차: 연속형 피처 간의 밀도 있는 상호작용 (Cross-feature) 학습 ──
            nn.Linear(num_asof_feats, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            # ── 2차: d_model 공간으로 투영 및 최종 스케일 고정 ──
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x: (B, S, num_asof_feats)
        returns: (B, S, d_model)
        """
        return self.mlp(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecoupledTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, attn_dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        # ----------------------------------------------------
        # 1. Decoupled Parallel Projections (ID & Prompt 독립 투영)
        # ----------------------------------------------------
        # ID 스트림용 Q, K, V
        self.q_id = nn.Linear(d_model, d_model)
        self.k_id = nn.Linear(d_model, d_model)
        self.v_id = nn.Linear(d_model, d_model)
        self.v_prompt = nn.Linear(d_model, d_model)
        self.dynamic_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.out_proj_id = nn.Linear(d_model, d_model)
        self.out_proj_prompt = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # ----------------------------------------------------
        # 2. Feedforward Networks & LayerNorms (각각 평행하게 유지)
        # ----------------------------------------------------
        self.norm1_id = nn.LayerNorm(d_model)
        self.norm1_prompt = nn.LayerNorm(d_model)
        self.dropout1_id = nn.Dropout(dropout)
        self.dropout1_prompt = nn.Dropout(dropout)

        self.ffn_id = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ffn_prompt = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm2_id = nn.LayerNorm(d_model)
        self.norm2_prompt = nn.LayerNorm(d_model)
        self.dropout2_id = nn.Dropout(dropout)
        self.dropout2_prompt = nn.Dropout(dropout)
        
        self._last_gate_mean = 0.0

    def forward(self, id_stream, prompt_stream, src_mask=None, src_key_padding_mask=None):
        """
        id_stream: (B, S, D)
        prompt_stream: (B, S, D)
        """
        B, S, D = id_stream.size()

        # [Pre-LayerNorm]
        normed_id = self.norm1_id(id_stream)
        normed_prompt = self.norm1_prompt(prompt_stream)

        # 💡 [Q2 해결] Multi-Head 쪼개기용 Helper 함수 정의
        def shape(x):
            # (B, S, D) -> (B, S, nhead, head_dim) -> (B, nhead, S, head_dim)
            return x.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # 1. Q, K, V 추출 (Prompt는 V만 추출)
        Q_id = shape(self.q_id(normed_id))
        K_id = shape(self.k_id(normed_id))
        V_id = shape(self.v_id(normed_id))
        
        V_p = shape(self.v_prompt(normed_prompt)) 

        # 2. [NOVA 핵심] 비침습적 어텐션 스코어 (ID 100% 주도)
        attn_scores = torch.matmul(Q_id, K_id.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. Masking & Softmax
        if src_mask is not None:
            attn_scores = attn_scores.masked_fill(src_mask, float('-inf'))
        if src_key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        #attn_scores = attn_scores.clamp(min=-1e4)   
        attn_weights = self.attn_dropout(torch.nan_to_num(F.softmax(attn_scores, dim=-1), nan=0.0))
        # 4. Value 적용 (동일한 시선으로 두 공간의 정보를 각각 퍼옵니다)
        out_id = torch.matmul(attn_weights, V_id)
        out_p = torch.matmul(attn_weights, V_p)

        # 차원 복구: (B, nhead, S, head_dim) -> (B, S, D)
        out_id = out_id.transpose(1, 2).contiguous().view(B, S, D)
        out_p = out_p.transpose(1, 2).contiguous().view(B, S, D)

        # 5. [MISSRec 핵심] Token-level Dynamic Gating
        gate_input = torch.cat([out_id, out_p], dim=-1)
        gate = self.dynamic_gate(gate_input)   # (B, S, D) 차원별 게이트 생성
        
        self._last_gate_mean = gate.detach().mean().item()
        
        gated_out_p = gate * out_p             # 필요한 프롬프트 정보만 통과

        # 6. 💡 [버그 수정] Value-level Injection (ID 스트림에 수혈)
        # ID 스트림이 다음 예측을 잘 할 수 있도록 통제된 프롬프트 지식을 더해줍니다.
        fused_out = out_id + gated_out_p
        id_stream = id_stream + self.dropout1_id(self.out_proj_id(fused_out))
        
        # 프롬프트 스트림은 정렬 손실(Alignment Loss) 계산 및 다음 레이어 전달을 위해 본연의 길을 갑니다.
        # (만약 프롬프트 스트림에도 노이즈를 안 넣고 싶다면 순수한 out_p를 씁니다)
        prompt_stream = prompt_stream + self.dropout1_prompt(self.out_proj_prompt(out_p))

        # 7. FFN & Residual Connection 2
        id_stream = id_stream + self.dropout2_id(self.ffn_id(self.norm2_id(id_stream)))
        prompt_stream = prompt_stream + self.dropout2_prompt(self.ffn_prompt(self.norm2_prompt(prompt_stream)))

        return id_stream, prompt_stream

# ================================================================
# SASRecUserTower_v4 
# ================================================================
class SASRecUserTower_v4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model      = args.d_model
        self.max_len      = args.max_len
        self.dropout_rate = args.dropout
        
        # ── [1] 입력 투영 및 임베딩 ──────────────────────────────
        self.item_proj = nn.Sequential(
            nn.Linear(args.pretrained_dim, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model),
            #nn.LayerNorm(self.d_model)
        )
        self.item_id_emb     = nn.Embedding(args.num_items + 1, self.d_model, padding_idx=0)
        self.price_item_proj = nn.Linear(1, self.d_model)
        self.pos_emb         = nn.Embedding(self.max_len, self.d_model)

        # Prompt와 ID를 독립적으로 정규화 (스케일 균형 보장)
        self.ln_prompt = nn.LayerNorm(self.d_model)
        self.ln_id     = nn.LayerNorm(self.d_model)
        self.ln_price  = nn.LayerNorm(self.d_model)

        self.emb_dropout = nn.Dropout(self.dropout_rate)
        
        self.recency_proj = nn.Linear(1, self.d_model)
        self.week_proj    = nn.Linear(2, self.d_model)
        self.global_ln    = nn.LayerNorm(self.d_model)

        # ── [2] Prompt-Integrated Transformer ────────────────────
        # 기존 nn.TransformerEncoder를 대체

        self.prompt_final_ln = nn.LayerNorm(self.d_model)
        self.id_final_ln     = nn.LayerNorm(self.d_model)
        
        self.num_layers = args.num_layers
        self.decoupled_layers = nn.ModuleList([
            DecoupledTransformerLayer(
                d_model=self.d_model,
                nhead=args.nhead,
                dim_feedforward=self.d_model * 2,
                dropout=0.1,
                attn_dropout=0.05,
            ) for _ in range(self.num_layers)
        ])
        
        self.final_enc_ln_id = nn.LayerNorm(self.d_model)
        
   # ── [2] Static Stream ────────────────────────────────────
        mid_dim, low_dim = 16, 4

        # 이산 피처 Embedding (gate 제거 → Embedding weight 자체가 중요도 학습)
        self.age_emb         = nn.Embedding(11, mid_dim, padding_idx=0)
        self.channel_emb     = nn.Embedding(4,  low_dim, padding_idx=0)
        self.club_status_emb = nn.Embedding(4,  low_dim, padding_idx=0)
        self.news_freq_emb   = nn.Embedding(3,  low_dim, padding_idx=0)
        self.fn_emb          = nn.Embedding(3,  low_dim, padding_idx=0)
        self.active_emb      = nn.Embedding(3,  low_dim, padding_idx=0)
        discrete_dim = mid_dim + low_dim * 5
        # 💡 [핵심 1] 이산형 피처 전용 투영기 (d_model로 확장 & 체급 고정)
        self.discrete_proj = nn.Sequential(
            nn.Linear(discrete_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )
        
        # asof 연속 피처 MLP (gate 제거 → MLP weight가 중요도 학습)
        # cont_feats index: 0=price(item_emb용), 1~4=asof 4개
        num_asof = 4
        self.cont_mlp = ContinuousFeatureMLP(
            num_asof_feats=num_asof,
            d_model=self.d_model,
            dropout=self.dropout_rate
        ) 
        self.static_final_ln = nn.LayerNorm(self.d_model)
        # static_mlp 입력 차원
        # age(16) + chan(4) + club(4) + news(4) + fn(4) + act(4) + cont_mlp(d_model)
        #cont_out_dim = 32  # cont_mlp d_mode
        #static_input_dim = mid_dim + low_dim * 5 + cont_out_dim  # 36 + 32 = 68
        
        #self.static_mlp = nn.Sequential(
        #    nn.Linear(static_input_dim, self.d_model),
        #    nn.LayerNorm(self.d_model),
        #    nn.GELU(),
        #    nn.Dropout(self.dropout_rate)
        #)

        # ── [3] Fusion ───────────────────────────────────────────
        self.seq_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model) # 👈 추가
        )
        
        self.static_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model), # 👈 추가
            nn.Dropout(0.3)             # 👈 Static이 치트키가 되지 못하도록 강한 모래주머니 채우기
        )

        # 업그레이드된 SENet 사용
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
        for proj in [self.week_proj, self.recency_proj]:
            nn.init.xavier_normal_(proj.weight)
            nn.init.constant_(proj.bias, 0)
    
    def _init_weights(self, module):
        # 1. Linear 레이어: Kaiming Normal (GELU와 찰떡궁합)
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # 2. Embedding 레이어: Small Std Normal (SASRec/BERT 표준)
        elif isinstance(module, nn.Embedding):
            # 0.02는 임베딩 공간이 너무 팽창하거나 수축하지 않게 만드는 황금비율입니다.
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 3. LayerNorm: Identity (1.0으로 시작하여 스트림 간 균형 보장)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0) # 👈 ln_id, ln_prompt 모두 여기서 초기화

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
                padding_mask=None, training_mode=True, return_streams=False):

        device     = item_ids.device
        seq_len    = item_ids.size(1)
        batch_size = item_ids.size(0)

        # ── position / causal mask ────────────────────────────────
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

        # ====================================================================
        # Phase 2: Stream Decoupling (덧셈에서 분리로 변경!)
        # ====================================================================
        
        # 1. Prompt Stream (SimCLR Content 공간)
        # "이 아이템들의 본질적인 스타일과 메타데이터는 무엇인가?"
        comp_pretrained = self.ln_prompt(self.item_proj(pretrained_vecs))
        
        # 1. Prompt Stream
        prompt_stream = comp_pretrained + pos_embeddings + global_context
        prompt_stream = self.prompt_final_ln(prompt_stream) # 💡 분산 3.0 -> 1.0 압축!
        prompt_stream = self.emb_dropout(prompt_stream)
        
        # 2. ID Stream (CF Behavior 공간)
        # "유저들이 어떤 순서로 이 아이템들을 소비하고 유행을 따르는가?"
        comp_item_id = self.ln_id(self.item_id_emb(item_ids))
        item_price   = cont_feats[:, :, 0:1]
        comp_price   = self.ln_price(self.price_item_proj(item_price))
        
        id_stream = comp_item_id + comp_price + pos_embeddings + global_context
        id_stream = self.id_final_ln(id_stream)             # 💡 분산 4.0 -> 1.0 압축!
        id_stream = self.emb_dropout(id_stream)
        
        # ====================================================================
        # Phase 3: Decoupled Prompt-Integrated Encoding
        # ====================================================================
        hidden_id = id_stream
        hidden_prompt = prompt_stream  # 프롬프트도 살아서 같이 흘러갑니다.
        
        for layer in self.decoupled_layers:
            # 이제 두 스트림이 레이어 내부에서 어텐션 맵만 공유하고 각자의 길을 갑니다.
            hidden_id, hidden_prompt = layer(
                id_stream=hidden_id,
                prompt_stream=hidden_prompt,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask
            )
        if return_streams:
            # 패딩 제외 평균 풀링
            mask = (~padding_mask).float().unsqueeze(-1)     # (B, S, 1)
            h_id_pool  = (hidden_id     * mask).sum(1) / mask.sum(1).clamp(min=1)
            h_pr_pool  = (hidden_prompt * mask).sum(1) / mask.sum(1).clamp(min=1)
            streams = (h_id_pool, h_pr_pool)
    
        # 예측에는 "유저의 최종 행동 맥락(ID)"을 사용합니다.
        # (이 hidden_id는 이미 심층부에서 프롬프트의 스타일 가이드를 완벽히 흡수한 상태입니다)
        item_output = self.final_enc_ln_id(hidden_id) # (B, S, D)
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

        # 1. 이산형 피처 결합 및 투영
        discrete_cat = torch.cat([
            emb_age, emb_chan, emb_club,
            emb_news, emb_fn, emb_act
        ], dim=-1)                                         # (B, S, 36)
        
        discrete_vec = self.discrete_proj(discrete_cat)    # (B, S, D)

        # 2. 연속형 피처 투영
        asof_feats = cont_feats[:, :, 1:]                  # (B, S, 4)
        cont_vec   = self.cont_mlp(asof_feats)             # (B, S, D)

        # 3. 💡 [핵심 3] 동등한 체급에서 합산 (Concat 대신 Add)
        # 이제 두 피처가 동일한 D 차원 공간에서 경쟁 및 보완합니다.
        user_profile_vec = discrete_vec + cont_vec         # (B, S, D)              # (B,S,D)
        user_profile_vec = self.static_final_ln(user_profile_vec) # 👈 이 한 줄이 밸런스를 완벽하게 잠급니다.
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
                final_vec = F.normalize(final_vec, p=2, dim=-1)
            if return_streams:
                return final_vec, streams
            return final_vec
        else:
            # 추론: 마지막 스텝만 사용
            h_seq    = self.seq_proj(item_output[:, -1:, :])              # (B,1,D)
            h_static = self.static_proj(user_profile_vec[:, -1:, :])     # (B,1,D)

            fused     = self.fusion_gate(h_seq, h_static)                 # (B,1,D)
            final_vec = self.output_proj(fused).squeeze(1)                # (B,D)

            if target_week is not None:
                final_vec = final_vec + target_week_vec[:, -1, :]
                final_vec = F.normalize(final_vec, p=2, dim=-1)
            if return_streams:
                return final_vec, streams
            return final_vec

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
        import torch
import torch.nn.functional as F

def inbatch_corrected_logq_loss_with_hybrid_hard_neg(
    user_emb, seq_item_emb, target_ids, user_ids, log_q_tensor,
    sbert_embs = None,
    hn_item_emb=None, batch_hard_neg_ids=None, item_embedding_weight=None,
    flat_history_item_ids=None,
    step_weights=None,
    temperature=0.05,  # 💡 [업데이트] 0.05 디폴트
    lambda_logq=1.0,
    margin=0.00,
    T_HN=0.22,
    beta=0.25,
    T_sample=0.5, 
    boundary_ratio=0.85, 
    return_metrics=False
):
    N = user_emb.size(0)
    device = user_emb.device
    
    # 💡 [FP32 최적화 1] FP32 환경의 표준 마스킹 값 (-1e9)
    # FP16에서는 -65500 부근이 한계였지만, FP32에서는 -1e9를 써도 연산 붕괴가 일어나지 않습니다.
    SAFE_NEG_INF = -1e9

    # -----------------------------------------------------------
    # 0. 가중치 준비
    # -----------------------------------------------------------
    if step_weights is not None:
        step_weights = torch.as_tensor(step_weights, device=device, dtype=torch.float32)
        weight_sum = step_weights.sum() + 1e-8 # 💡 [FP32 최적화 2] 엡실론 값을 FP32 표준(1e-8)으로 변경
    else:
        weight_sum = None

    # -----------------------------------------------------------
    # 1. In-batch Logits 계산
    # -----------------------------------------------------------
    sim_matrix = torch.matmul(user_emb, seq_item_emb.T)  # [N, N]
    pos_sim = torch.diagonal(sim_matrix)                  # [N]
    labels = torch.arange(N, device=device)
    if sbert_embs is not None:
        # SBERT 유사도와 패널티 계산만 no_grad 안에서 수행
        with torch.no_grad():
            batch_sbert = sbert_embs[target_ids]
            sbert_sim_matrix = torch.matmul(batch_sbert, batch_sbert.T) 
            sbert_sim_matrix.fill_diagonal_(0.0)
            
            fn_threshold = 0.95 
            alpha = 0.0 # 극저온(tau=0.05) 지수 폭발을 완벽히 제압하기 위한 스케일링 팩터
            
            # 💡 [핵심] 0.85 미만은 0이 되고, 0.85 초과분만 남깁니다 (연속성 확보)
            excess_sim = torch.clamp(sbert_sim_matrix - fn_threshold, min=0.0)
            
            # 초과량에 비례하여 부드럽지만 강력하게 지수 함수를 꺾어버릴 페널티 산출
            fn_penalty = excess_sim * alpha
            
            # (선택 사항) 로깅을 위해 현재 배치에서 페널티를 받은 샘플 수 계산
            # num_penalized = (excess_sim > 0).sum().item()
            
        # 2. 모델 로짓 업데이트 (반드시 no_grad 밖에서 수행하여 미분 그래프 보존!)
        sim_matrix = sim_matrix - fn_penalty
    
    
    # 온도 스케일링 (FP32이므로 값이 커져도 Overflow 걱정 없음)
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
    
    # 가짜 오답 마스킹 (-1e9 적용)
    logits = logits.masked_fill(false_neg_mask, SAFE_NEG_INF)
    
    # 🚀 [FP32 최적화 3] torch.clamp(..., max=1e4) 완전 삭제!
    # FP16 시절 Float Overflow(Inf)를 막기 위한 억지 제한이었습니다. 
    # FP32는 3.4e38까지 버티며, PyTorch의 F.cross_entropy는 내부적으로 LogSumExp 트릭을 써서
    # 로짓 값이 아무리 커도 폭발하지 않습니다. 클램프를 지워야 순수 그래디언트가 보존됩니다.

    metrics = {}

    # -----------------------------------------------------------
    # 2. Hard Negative Mining & MNS Loss 계산
    # -----------------------------------------------------------
    num_hn_to_use = 20
    loss_hn = None

    if hn_item_emb is not None and batch_hard_neg_ids is not None:
        num_pool = 200
        # [STEP 1] Gradient 없이 HN 후보 필터링 & 선택
        with torch.no_grad():
            hn_emb_no_grad = hn_item_emb.detach()
            hn_sim_no_grad = torch.bmm(
                user_emb.unsqueeze(1),
                hn_emb_no_grad.transpose(1, 2)
            ).squeeze(1)  # [N, pool_size]

            # 히스토리 기반 FN 마스크
            final_fn_mask = torch.zeros_like(hn_sim_no_grad, dtype=torch.bool)
            if flat_history_item_ids is not None:
                final_fn_mask = (
                    batch_hard_neg_ids.unsqueeze(2) == flat_history_item_ids.unsqueeze(1)
                ).any(dim=2)

            masked_sims = hn_sim_no_grad.masked_fill(final_fn_mask, SAFE_NEG_INF)
            
            _, top_idx_pool = torch.topk(masked_sims, num_pool, dim=1)
            pool_sims = torch.gather(masked_sims, 1, top_idx_pool)

            # 확률 변환 및 추출
            sampling_weights = F.softmax(pool_sims / T_sample, dim=1)  # [N, num_pool]
            top_idx_local = torch.multinomial(
                sampling_weights,
                num_samples=num_hn_to_use,
                replacement=False
            )  # [N, num_hn_to_use]
            
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

        # [MNS] HN loss 로짓 구성
        hn_loss_input = torch.cat([
            pos_sim.unsqueeze(1) / T_HN,   # [N, 1] (정답)
            hn_sim / T_HN                  # [N, num_hn] (오답들)
        ], dim=1)  
        
        # 🚀 [FP32 최적화 4] torch.clamp(hn_loss_input, max=1e4) 완전 삭제!
        # 마찬가지로 HNM 로짓에서도 억지 제한을 해제하여 정밀한 패널티 부여.

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
def inbatch_corrected_logq_loss_with_hybrid_hard_neg_autocast(
    user_emb, seq_item_emb, target_ids, user_ids, log_q_tensor,
    hn_item_emb=None, batch_hard_neg_ids=None,item_embedding_weight=None,
    flat_history_item_ids=None,
    step_weights=None,
    temperature=0.07,
    lambda_logq=1.0,
    margin=0.00,
    T_HN=0.22,
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
    num_hn_to_use = 20
    loss_hn = None

    if hn_item_emb is not None and batch_hard_neg_ids is not None:
        num_pool = 200
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
    user_tower = SASRecUserTower_v4(cfg).to(device)
    
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
            
            
            
            

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
        





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
    total_ap10 = 0.0  # 💡 [추가] MAP@0 누적 변수
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
                
                # 💡 [추가] MAP@20 계산 로직
                if 10 in k_list:
                    ap10 = 0.0
                    hits_so_far = 0
                    pred_10 = pred_ids[i, :10] # K=20 예측 리스트
                    
                    for rank, p_id in enumerate(pred_10):
                        if p_id in actual_indices:
                            hits_so_far += 1
                            ap10 += hits_so_far / (rank + 1.0) # Precision @ rank
                    
                    # 분모: 실제 유저가 구매한 정답 개수와 20 중 더 작은 값 (정답이 20개보다 적을 수 있으므로)
                    num_relevant = min(len(actual_indices), 10)
                    if num_relevant > 0:
                        total_ap10 += ap10 / num_relevant
            
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
            results['MAP@10'] = (total_ap10 / total_valid_users) * 100
            
    print(f"\n📈 [Validation Results] Valid Users: {total_valid_users}")
    for k in k_list:
        print(f"   - Recall@{k:03d}: {results.get(f'Recall@{k}', 0):.2f}%")
    
    # 💡 [추가] 출력
    if 'MAP@10' in results:
        print(f"   - MAP@10: {results['MAP@10']:.2f}%")    
    del full_item_embeddings, norm_item_embeddings
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

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

def log_feature_contributions_v4(model, wandb, epoch=None):
    """
    SASRecUserTower_v4 전용 정밀 로깅 함수
    수정 사항: 
      - static_mlp 제거 대응 (discrete_proj + cont_mlp 이원화 반영)
      - 2-Layer cont_mlp 내부 기여도 추적
      - Concat 기반 StreamFusionGate(SENet)의 seq vs static 실제 비중 분석
    """
    log_dict = {}
    d_model = model.d_model

    with torch.no_grad():
        # ════════════════════════════════════════════════════════
        # 1. Item Stream 컴포넌트별 LayerNorm 스케일 (체급 확인)
        # ════════════════════════════════════════════════════════
        ln_modules = {
            'ItemStream/ln_prompt': model.ln_prompt,
            'ItemStream/ln_id':     model.ln_id,
            'ItemStream/ln_price':  model.ln_price,
        }
        for name, ln in ln_modules.items():
            log_dict[f"{name}_norm"] = ln.weight.norm().item()
            log_dict[f"{name}_mean"] = ln.weight.mean().item()

        # ════════════════════════════════════════════════════════
        # 2. Static 이산형 피처 Embedding 자체 에너지 (Weight Norm)
        # ════════════════════════════════════════════════════════
        emb_modules = {
            'Static_Emb/age':     model.age_emb,
            'Static_Emb/chan':    model.channel_emb,
            'Static_Emb/club':    model.club_status_emb,
            'Static_Emb/news':    model.news_freq_emb,
            'Static_Emb/fn':      model.fn_emb,
            'Static_Emb/act':     model.active_emb,
        }
        for name, emb in emb_modules.items():
            valid_weights = emb.weight[1:]  # padding_idx 제외
            row_norms = valid_weights.norm(dim=1)
            log_dict[f"{name}_norm_mean"] = row_norms.mean().item()

        # ════════════════════════════════════════════════════════
        # 3. Discrete Projection (이산형 피처가 D차원으로 갈 때의 비중)
        #    discrete_proj[0] = Linear(36, D)
        # ════════════════════════════════════════════════════════
        disc_linear = model.discrete_proj[0]
        disc_col_norms = disc_linear.weight.norm(dim=0) # (36,)
        
        disc_blocks = {
            'Stat_Weights/Disc_age':     (0, 16),
            'Stat_Weights/Disc_channel': (16, 20),
            'Stat_Weights/Disc_club':    (20, 24),
            'Stat_Weights/Disc_news':    (24, 28),
            'Stat_Weights/Disc_fn':      (28, 32),
            'Stat_Weights/Disc_active':  (32, 36),
        }
        for name, (start, end) in disc_blocks.items():
            log_dict[name] = disc_col_norms[start:end].mean().item()

        # ════════════════════════════════════════════════════════
        # 4. 리팩토링된 2-Layer Continuous MLP 기여도
        #    cont_mlp.mlp[0] = Linear(4, 32) -> 실제 피처 상호작용 지점
        # ════════════════════════════════════════════════════════
        cont_linear = model.cont_mlp.mlp[0]  # Linear(4, 32)
        cont_col_norms = cont_linear.weight.norm(dim=0) # (4,)
        
        asof_labels = ['price_std', 'last_price_diff', 'repurchase', 'weekend']
        for label, norm_val in zip(asof_labels, cont_col_norms):
            log_dict[f"Stat_Weights/Cont_{label}"] = norm_val.item()

        # ════════════════════════════════════════════════════════
        # 5. SENet Fusion Gate (Seq vs Static 최종 주도권)
        #    fusion_gate.se[0] = Linear(D*2, D//4) <- Concat 기반
        # ════════════════════════════════════════════════════════
        fusion_linear = model.fusion_gate.se[0] 
        fusion_col_norms = fusion_linear.weight.norm(dim=0) # (D*2,)
        
        seq_imp = fusion_col_norms[:d_model].mean().item()
        stat_imp = fusion_col_norms[d_model:].mean().item()
        
        log_dict['SENet/seq_importance'] = seq_imp
        log_dict['SENet/static_importance'] = stat_imp
        log_dict['SENet/static_ratio'] = stat_imp / (seq_imp + stat_imp + 1e-9)

        # ════════════════════════════════════════════════════════
        # 6. 기타 Gate 및 최종 스케일
        # ════════════════════════════════════════════════════════
        log_dict['Gate/target_week_val'] = torch.sigmoid(model.target_week_gate).item()
        log_dict['Static/final_ln_norm'] = model.static_final_ln.weight.norm().item()
        # ════════════════════════════════════════════════════════
        # [신규 수정] 7. Decoupled Transformer Layers의 Dynamic Gate 모니터링
        #  - 고정된 prompt_weight 대신, 신경망이 판단한 '실제 게이트 개방률'과 '네트워크 가중치' 추적
        # ════════════════════════════════════════════════════════
        if hasattr(model, 'decoupled_layers'):
            gate_open_ratios = []
            for i, layer in enumerate(model.decoupled_layers):
                # A. 실제 데이터가 통과할 때 게이트가 평균적으로 얼만큼 열렸는가 (0~1)
                # (주의: 에포크 마지막 배치의 평균 상태를 보여줍니다)
                if hasattr(layer, '_last_gate_mean'):
                    open_ratio = layer._last_gate_mean
                    log_dict[f'Decoupled/layer_{i}_gate_open_ratio'] = open_ratio
                    gate_open_ratios.append(open_ratio)
                
                # B. Gate 네트워크(Linear) 자체의 학습 상태 (Weight Norm & Bias)
                # 가중치 놈이 커지면 게이트가 토큰마다 매우 예민하고 극단적으로 판단하고 있다는 뜻입니다.
                gate_linear = layer.dynamic_gate[0]
                log_dict[f'Decoupled/layer_{i}_gate_weight_norm'] = gate_linear.weight.norm().item()
                log_dict[f'Decoupled/layer_{i}_gate_bias_mean'] = gate_linear.bias.mean().item()
            
            # 모델 전체의 평균 프롬프트 수용률 추적
            if gate_open_ratios:
                log_dict['Decoupled/avg_gate_open_ratio'] = sum(gate_open_ratios) / len(gate_open_ratios)
    wandb.log(log_dict)
    
def shuffle_within_session_gpu(session_ids, padding_mask):
    """
    session_ids: (B, S) 정수
    padding_mask: (B, S) bool
    returns shuffle_idx: (B, S) - view2용 재정렬 인덱스
    """
    B, S   = session_ids.shape
    device = session_ids.device

    rand_keys = torch.rand(B, S, device=device)

    # padding 위치 → -inf (argsort 시 맨 앞 = left padding 유지)
    rand_keys = rand_keys.masked_fill(padding_mask, -float('inf'))

    # session_id(정수) + rand(0~1) → 세션 경계 유지하면서 내부 셔플
    sort_key    = session_ids.float() + rand_keys    # (B, S)
    shuffle_idx = torch.argsort(sort_key, dim=1)     # (B, S)

    return shuffle_idx


def apply_shuffle(tensor, shuffle_idx):
    """shuffle_idx로 텐서 재정렬"""
    if tensor.dim() == 2:
        return torch.gather(tensor, 1, shuffle_idx)
    elif tensor.dim() == 3:
        idx = shuffle_idx.unsqueeze(-1).expand_as(tensor)
        return torch.gather(tensor, 1, idx)
    
def get_cl_lambdas(epoch, hnm_start_epoch):

    # ── 구간 1: warm-up (in-batch only) ──────────────────────
    # CL이 주요 학습 신호 역할
    if epoch < hnm_start_epoch:
        return {
            'lambda_unsup': 0.10,   # unsup 강하게
            'lambda_sup':   0.02,   # sup 약하게 (static 겹침 억제)
        }

    # ── 구간 2: HNM 초기 도입 ────────────────────────────────
    # HNM이 주요 학습 신호로 전환
    # CL은 보조 역할로 축소
    elif epoch < hnm_start_epoch + 5:
        return {
            'lambda_unsup': 0.05,
            'lambda_sup':   0.01,
        }

    # ── 구간 3: HNM 안정화 이후 ──────────────────────────────
    # CL을 더 낮춰 HNM 신호에 집중
    else:
        return {
            'lambda_unsup': 0.03,
            'lambda_sup':   0.01,
        }
def get_hnm_params(hnm_relative_epoch: int) -> dict:
    """
    hnm_relative_epoch: HNM 시작 후 몇 번째 에포크인지 (1-indexed)
    
    전략:
    - T_HN: 크게 시작(soft) → 목표값으로 감소 (gradient 충격 완화)
    - beta: 작게 시작 → 목표값으로 증가 (HN loss 비중 점진적 확대)
    """
    # 워밍업 스케줄 정의
    schedule = [
        # (T_HN, beta)
        (0.48, 0.05),   # epoch 1: 메인 온도의 4배 (매우 soft), HN 비중 최소
        (0.36, 0.10),   # epoch 2: 메인 온도의 3배
        (0.28, 0.15),   # epoch 3: 서서히 타겟에 근접
        (0.24, 0.25),   # epoch 4~: 목표값 도달 (메인 온도의 2배)
    ]
    
    idx = min(hnm_relative_epoch - 1, len(schedule) - 1)
    T_HN, beta = schedule[idx]
    
    return {"T_HN": T_HN, "beta": beta}

def alignment_loss_fn(h_id, h_prompt):
    # 방향 반전: ID는 고정, Prompt가 ID 공간으로 적응
    h_id_sg = h_id.detach()  # ID 스트림 고정
    cos = F.cosine_similarity(h_prompt, h_id_sg, dim=-1)
    return (1.0 - cos).mean()
# ================================================================
# Train 함수 (DuoRec Unsupervised + Supervised CL)
# ================================================================
def train_user_tower_session_sampler_with_intent_point_autocast(
    epoch, model, item_tower, norm_item_embeddings, log_q_tensor,
    dataloader, optimizer, scaler, cfg, device,
    hard_neg_pool_tensor, scheduler, hn_scheduler, T_sample, beta,
    hn_refresh_interval, hn_exclusion_top_k, hn_mine_k, T_HN,sbert_embs
):
    """
    SASRecUserTower_v4 전용 훈련 루프 (DuoRec 제거 및 HNM 최적화 버전)
    """
    model.train()
    total_loss_accum = 0.0
    num_batches = 0
    
    epoch_discard_ratio_sum = 0.0
    num_hn_metric_batches = 0
    accumulation_steps = 1
    
    current_hn_pool = hard_neg_pool_tensor
    current_norm_item_embs = norm_item_embeddings

    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):
        # ── [1] Hard Negative Pool 갱신 (일정 주기마다) ────────────────
        if (current_hn_pool is not None and hn_refresh_interval > 0 and 
            batch_idx > 0 and batch_idx % hn_refresh_interval == 0):
            item_tower.eval()
            with torch.no_grad():
                all_item_embs = item_tower.get_all_embeddings()
                current_norm_item_embs = F.normalize(all_item_embs, p=2, dim=1)
                current_hn_pool, _ = mine_global_hard_negatives(
                    item_embs=norm_item_embeddings,    # 다시 계산 안 하고 캐시본 사용
                    sbert_embs=sbert_embs,     # 💡 [수정 사항 2] 시맨틱 방어막 텐서 주입
                    fn_threshold=0.85,
                    fn_lower=0.50, # 💡 [수정 사항 3] 시맨틱 유사도 0.85 이상은 오답 풀에서 영구 배제
                    exclusion_top_k=hn_exclusion_top_k,      # (안전지대)
                    mine_k=hn_mine_k,
                    batch_size=2048, 
                    device=device
                )
            item_tower.train()
            print(f"🔍 Refreshing Global Hard Negatives in Loop...")

            

        # ── [2] 데이터 로딩 및 장치 이동 ────────────────────────────────
        # (v1/v2 구분을 없애고 단일 스트림으로 통합)
        item_ids = batch['item_ids'].to(device, non_blocking=True)
        target_ids = batch['target_ids'].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(device, non_blocking=True)
        session_ids = batch['session_ids'].to(device, non_blocking=True)
        interaction_dates = batch['interaction_dates'].to(device, non_blocking=True)
        
        # Pretrained Vectors
        if 'pretrained_vecs' in batch:
            pretrained_vecs = batch['pretrained_vecs'].to(device, non_blocking=True)
        else:
            pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device, non_blocking=True)

        # 공통 피처 및 기타 입력을 kwargs로 묶음
        forward_kwargs = {
            'pretrained_vecs': pretrained_vecs,
            'item_ids': item_ids,
            'time_bucket_ids': batch['time_bucket_ids'].to(device, non_blocking=True),
            'type_ids': batch['type_ids'].to(device, non_blocking=True),
            'color_ids': batch['color_ids'].to(device, non_blocking=True),
            'graphic_ids': batch['graphic_ids'].to(device, non_blocking=True),
            'section_ids': batch['section_ids'].to(device, non_blocking=True),
            'age_bucket': batch['age_bucket'].to(device, non_blocking=True),
            'price_bucket': batch['price_bucket'].to(device, non_blocking=True),
            'cnt_bucket': batch['cnt_bucket'].to(device, non_blocking=True),
            'recency_bucket': batch['recency_bucket'].to(device, non_blocking=True),
            'channel_ids': batch['channel_ids'].to(device, non_blocking=True),
            'club_status_ids': batch['club_status_ids'].to(device, non_blocking=True),
            'news_freq_ids': batch['news_freq_ids'].to(device, non_blocking=True),
            'fn_ids': batch['fn_ids'].to(device, non_blocking=True),
            'active_ids': batch['active_ids'].to(device, non_blocking=True),
            'cont_feats': batch['cont_feats'].to(device, non_blocking=True),
            'recency_offset': batch['recency_offset'].to(device, non_blocking=True),
            'current_week': batch['current_week'].to(device, non_blocking=True),
            'target_week': batch['target_week'].to(device, non_blocking=True),
            'session_ids': session_ids,
            'padding_mask': padding_mask,
            'training_mode': True
        }

        # ── [3] Forward & Loss 계산 ──────────────────────────────────
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            user_outputs, (h_id, h_pr) = model(**forward_kwargs, return_streams=True)

            
            # (A) Time-Decay Weighting
            valid_mask = ~padding_mask
            max_dates = interaction_dates.masked_fill(padding_mask, -1).max(dim=1, keepdim=True)[0]
            delta_t = (max_dates - interaction_dates).float()
            min_weight, half_life = 0.2, 21.0
            decay_rate = math.log(2) / half_life
            seq_weights = min_weight + (1.0 - min_weight) * torch.exp(-decay_rate * delta_t)

            # (B) Session Last & Random Sampling Mask
            shifted_session = torch.roll(session_ids, shifts=-1, dims=1)
            shifted_session[:, -1] = 0
            session_last_mask = (session_ids != shifted_session) & valid_mask
            
            intermediate_mask = valid_mask & ~session_last_mask
            random_sample_mask = torch.rand_like(intermediate_mask, dtype=torch.float) < 0.2
            final_loss_mask = session_last_mask | (intermediate_mask & random_sample_mask)

            # (C) Flattening & Sampling (메모리 방어)
            flat_output = user_outputs[final_loss_mask]
            flat_targets = target_ids[final_loss_mask]
            flat_weights = seq_weights[final_loss_mask]
            
            # 배치 행 인덱스 복구
            batch_row_idx = torch.arange(item_ids.size(0), device=device).unsqueeze(1).expand(-1, item_ids.size(1))
            flat_user_ids = batch_row_idx[final_loss_mask]

            if flat_output.size(0) > 8500: # Max Flat Size 제한
                _, recent_idx = torch.topk(flat_weights, k=8500)
                flat_output, flat_targets, flat_user_ids, flat_weights = \
                    flat_output[recent_idx], flat_targets[recent_idx], flat_user_ids[recent_idx], flat_weights[recent_idx]

            # (D) Main HNM Loss 계산
            if flat_output.size(0) > 0:
                flat_user_emb = F.normalize(flat_output, p=2, dim=1)
                batch_seq_item_emb = F.normalize(item_tower.item_matrix.weight[flat_targets], p=2, dim=1)
                
                batch_hn_item_emb_cached = None
                batch_hard_neg_ids = None
                if current_hn_pool is not None:
                    batch_hard_neg_ids = current_hn_pool[flat_targets]
                    batch_hn_item_emb_cached = current_norm_item_embs[batch_hard_neg_ids]

                main_loss, b_metrics = inbatch_corrected_logq_loss_with_hybrid_hard_neg(
                    user_emb=flat_user_emb, seq_item_emb=batch_seq_item_emb,
                    target_ids=flat_targets, user_ids=flat_user_ids,
                    log_q_tensor=log_q_tensor,
                    hn_item_emb=batch_hn_item_emb_cached,
                    batch_hard_neg_ids=batch_hard_neg_ids,
                    item_embedding_weight=item_tower.item_matrix.weight,
                    flat_history_item_ids=item_ids[flat_user_ids],
                    step_weights=flat_weights,
                    temperature=0.05, lambda_logq=cfg.lambda_logq,
                    T_HN=T_HN, beta=beta, T_sample=T_sample,
                    return_metrics=True
                )
            else:
                main_loss = torch.tensor(0.0, device=device)
                b_metrics = {}

            align_loss = torch.tensor(0.0, device=device)#alignment_loss_fn(h_id, h_pr)
            total_loss = (main_loss + cfg.lambda_align * align_loss) / accumulation_steps


        # ── [4] Backward & Step ──────────────────────────────────────
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if hasattr(item_tower, 'parameters'):
            torch.nn.utils.clip_grad_norm_(item_tower.parameters(), max_norm=10.0) # 임베딩 타워용 별도 기준 적용
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        # ── [5] 통계 누적 및 WandB 로깅 ───────────────────────────────
        total_loss_accum += main_loss.item()
        num_batches += 1
    

        pbar.set_postfix({'Loss': f"{main_loss.item():.4f}"})

        if batch_idx % 100 == 0:
            wandb_log = {"Train/Total_Loss": main_loss.item(),
                          "Train/Align_Loss": align_loss.item()}
            # HNM 관련 메트릭 통합 로깅
            for k in ['sim/pos', 'sim/hn_max_mean', 'loss/hn_ratio','loss/hn', 'loss/inbatch', 'hn/discarded_ratio']:
                if k in b_metrics: wandb_log[k] = b_metrics[k]
            #wandb_log["train/grad_norm"] = grad_norm.item()

            wandb.log(wandb_log)

    # ── [6] Epoch 종료 ───────────────────────────────────────────
    avg_loss = total_loss_accum / num_batches
    avg_discard = epoch_discard_ratio_sum / num_hn_metric_batches if num_hn_metric_batches > 0 else 0.0
    
    print(f"🏁 Epoch {epoch} Completed | Avg_Loss: {avg_loss:.4f} | Discard: {avg_discard:.4f}")
    
    # 우리가 만든 정교한 피처 로깅 함수 호출
    log_feature_contributions_v4(model, wandb)
    
    return avg_loss, False, avg_discard
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

def train_user_tower_session_sampler_with_intent_point(
    epoch, model, item_tower, norm_item_embeddings, log_q_tensor,
    dataloader, optimizer, cfg, device,  # 💡 [수정] scaler 제거
    hard_neg_pool_tensor, scheduler, hn_scheduler, T_sample, beta,
    hn_refresh_interval, hn_exclusion_top_k, hn_mine_k, T_HN, sbert_embs
):
    """
    SASRecUserTower_v4 전용 훈련 루프 (순수 FP32 정밀 학습 & Accumulation 제거 버전)
    - Autocast 및 Scaler 제거 -> 극소 Gradient(온도 0.05) 손실 방지
    """
    model.train()
    total_loss_accum = 0.0
    num_batches = 0
    epoch_discard_ratio_sum = 0.0
    num_hn_metric_batches = 0
    
    current_hn_pool = hard_neg_pool_tensor
    current_norm_item_embs = norm_item_embeddings

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):
        # ── [1] Hard Negative Pool 갱신 (일정 주기마다) ────────────────
        if (current_hn_pool is not None and hn_refresh_interval > 0 and 
            batch_idx > 0 and batch_idx % hn_refresh_interval == 0):
            item_tower.eval()
            with torch.no_grad():
                all_item_embs = item_tower.get_all_embeddings()
                current_norm_item_embs = F.normalize(all_item_embs, p=2, dim=1)
                current_hn_pool, _ = mine_global_hard_negatives(
                    item_embs=norm_item_embeddings,
                    sbert_embs=sbert_embs,
                    fn_threshold=0.85,
                    fn_lower=0.50,
                    exclusion_top_k=hn_exclusion_top_k,
                    mine_k=hn_mine_k,
                    batch_size=2048, 
                    device=device
                )
            item_tower.train()
            print(f"\n🔍 Refreshing Global Hard Negatives in Loop...")

        # ── [2] 데이터 로딩 및 장치 이동 ────────────────────────────────
        item_ids = batch['item_ids'].to(device, non_blocking=True)
        target_ids = batch['target_ids'].to(device, non_blocking=True)
        padding_mask = batch['padding_mask'].to(device, non_blocking=True)
        session_ids = batch['session_ids'].to(device, non_blocking=True)
        interaction_dates = batch['interaction_dates'].to(device, non_blocking=True)
        
        # Pretrained Vectors
        if 'pretrained_vecs' in batch:
            pretrained_vecs = batch['pretrained_vecs'].to(device, non_blocking=True)
        else:
            pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device, non_blocking=True)

        forward_kwargs = {
            'pretrained_vecs': pretrained_vecs,
            'item_ids': item_ids,
            'time_bucket_ids': batch['time_bucket_ids'].to(device, non_blocking=True),
            'type_ids': batch['type_ids'].to(device, non_blocking=True),
            'color_ids': batch['color_ids'].to(device, non_blocking=True),
            'graphic_ids': batch['graphic_ids'].to(device, non_blocking=True),
            'section_ids': batch['section_ids'].to(device, non_blocking=True),
            'age_bucket': batch['age_bucket'].to(device, non_blocking=True),
            'price_bucket': batch['price_bucket'].to(device, non_blocking=True),
            'cnt_bucket': batch['cnt_bucket'].to(device, non_blocking=True),
            'recency_bucket': batch['recency_bucket'].to(device, non_blocking=True),
            'channel_ids': batch['channel_ids'].to(device, non_blocking=True),
            'club_status_ids': batch['club_status_ids'].to(device, non_blocking=True),
            'news_freq_ids': batch['news_freq_ids'].to(device, non_blocking=True),
            'fn_ids': batch['fn_ids'].to(device, non_blocking=True),
            'active_ids': batch['active_ids'].to(device, non_blocking=True),
            'cont_feats': batch['cont_feats'].to(device, non_blocking=True),
            'recency_offset': batch['recency_offset'].to(device, non_blocking=True),
            'current_week': batch['current_week'].to(device, non_blocking=True),
            'target_week': batch['target_week'].to(device, non_blocking=True),
            'session_ids': session_ids,
            'padding_mask': padding_mask,
            'training_mode': True
        }

        # ── [3] Forward & Loss 계산 (FP32 순수 정밀도) ─────────────────
        # 💡 [수정] autocast 블록 제거, FP32로 미세한 그래디언트 손실 없이 연산
        user_outputs, (h_id, h_pr) = model(**forward_kwargs, return_streams=True)

        # (A) Time-Decay Weighting
        valid_mask = ~padding_mask
        max_dates = interaction_dates.masked_fill(padding_mask, -1).max(dim=1, keepdim=True)[0]
        delta_t = (max_dates - interaction_dates).float()
        min_weight, half_life = 0.2, 21.0
        decay_rate = math.log(2) / half_life
        seq_weights = min_weight + (1.0 - min_weight) * torch.exp(-decay_rate * delta_t)

        # (B) Session Last & Random Sampling Mask
        shifted_session = torch.roll(session_ids, shifts=-1, dims=1)
        shifted_session[:, -1] = 0
        session_last_mask = (session_ids != shifted_session) & valid_mask
        
        intermediate_mask = valid_mask & ~session_last_mask
        random_sample_mask = torch.rand_like(intermediate_mask, dtype=torch.float) < 0.2
        final_loss_mask = session_last_mask | (intermediate_mask & random_sample_mask)

        # (C) Flattening & Sampling
        flat_output = user_outputs[final_loss_mask]
        flat_targets = target_ids[final_loss_mask]
        flat_weights = seq_weights[final_loss_mask]
        
        batch_row_idx = torch.arange(item_ids.size(0), device=device).unsqueeze(1).expand(-1, item_ids.size(1))
        flat_user_ids = batch_row_idx[final_loss_mask]

        if flat_output.size(0) > 8500:
            _, recent_idx = torch.topk(flat_weights, k=8500)
            flat_output, flat_targets, flat_user_ids, flat_weights = \
                flat_output[recent_idx], flat_targets[recent_idx], flat_user_ids[recent_idx], flat_weights[recent_idx]

        # (D) Main HNM Loss 계산
        if flat_output.size(0) > 0:
            flat_user_emb = F.normalize(flat_output, p=2, dim=1)
            batch_seq_item_emb = F.normalize(item_tower.item_matrix.weight[flat_targets], p=2, dim=1)
            
            batch_hn_item_emb_cached = None
            batch_hard_neg_ids = None
            if current_hn_pool is not None:
                batch_hard_neg_ids = current_hn_pool[flat_targets]
                batch_hn_item_emb_cached = current_norm_item_embs[batch_hard_neg_ids]

            main_loss, b_metrics = inbatch_corrected_logq_loss_with_hybrid_hard_neg(
                user_emb=flat_user_emb, seq_item_emb=batch_seq_item_emb,
                target_ids=flat_targets, user_ids=flat_user_ids,
                log_q_tensor=log_q_tensor,
                sbert_embs = sbert_embs,
                hn_item_emb=batch_hn_item_emb_cached,
                batch_hard_neg_ids=batch_hard_neg_ids,
                item_embedding_weight=item_tower.item_matrix.weight,
                flat_history_item_ids=item_ids[flat_user_ids],
                step_weights=flat_weights,
                temperature=0.07, lambda_logq=cfg.lambda_logq,
                T_HN=T_HN, beta=beta, T_sample=T_sample,
                return_metrics=True
            )
            
            # HNM Discard Ratio 추적용
            if 'hn/discarded_ratio' in b_metrics:
                epoch_discard_ratio_sum += b_metrics['hn/discarded_ratio']
                num_hn_metric_batches += 1
        else:
            main_loss = torch.tensor(0.0, device=device)
            b_metrics = {}

        align_loss = torch.tensor(0.0, device=device)
        # 💡 [수정] accumulation_steps 나눗셈 제거 (어차피 1이었으므로 직관적으로)
        total_loss = main_loss + cfg.lambda_align * align_loss

        # ── [4] Backward & Step (순수 PyTorch 방식) ──────────────────
        # 💡 [수정] scaler 제거 후 직접 backward 호출
        total_loss.backward()
        
        # 💡 [수정] unscale 불필요. 직접 clip_grad_norm_ 적용
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if hasattr(item_tower, 'parameters'):
            torch.nn.utils.clip_grad_norm_(item_tower.parameters(), max_norm=10.0) 
        
        # 💡 [수정] scaler.step() 대신 순수 optimizer.step()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if scheduler is not None:
            scheduler.step()

        # ── [5] 통계 누적 및 WandB 로깅 ───────────────────────────────
        total_loss_accum += main_loss.item()
        num_batches += 1
    
        pbar.set_postfix({'Loss': f"{main_loss.item():.4f}"})

        if batch_idx % 100 == 0:
            wandb_log = {"Train/Total_Loss": main_loss.item(),
                         "Train/Align_Loss": align_loss.item()}
            for k in ['sim/pos', 'sim/hn_max_mean', 'loss/hn_ratio','loss/hn', 'loss/inbatch', 'hn/discarded_ratio']:
                if k in b_metrics: wandb_log[k] = b_metrics[k]
            wandb.log(wandb_log)

    # ── [6] Epoch 종료 ───────────────────────────────────────────
    avg_loss = total_loss_accum / num_batches
    avg_discard = epoch_discard_ratio_sum / num_hn_metric_batches if num_hn_metric_batches > 0 else 0.0
    
    print(f"🏁 Epoch {epoch} Completed | Avg_Loss: {avg_loss:.4f} | Discard: {avg_discard:.4f}")
    
    # 정교한 피처 로깅 함수 호출
    log_feature_contributions_v4(model, wandb)
    
    return avg_loss, False, avg_discard
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
    cfg.lr = 8e-4
    cfg.epochs = 40           # 처음부터 학습하므로 넉넉하게 40에포크 설정
    cfg.HN_K = 150             # HNM 발동 시 추출할 풀 사이즈
    cfg.EX_TOP_K = 0
    cfg.soft_penalty_weigh = 1.0
    cfg.hn_scheduled = False
    cfg.batch_size = 512
    cfg.dropout = 0.1
    cfg.lambda_align = 0.05
    FREEZE_ITEM_EPOCHS = 0
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
    
    aligned_sbert_embs =get_or_build_aligned_sbert_embeddings(processor, cfg.base_dir,device)

    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22", aligned_vecs, is_train=False)
    
    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v3",
        name=f"FromScratch_DIF-SR_lr{cfg.lr}_K{cfg.HN_K}", 
        config=cfg.__dict__ 
    )
    
    # -----------------------------------------------------------
    # 3. Models Setup (가중치 로드 없이 초기화)
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, None, log_q_tensor) 
    
    save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_from_scratch_prompt.pth")
    save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_from_scratch_prompt.pth")
    item_tower.init_from_pretrained(aligned_vecs.to(device))
    print(f"❄️ Item Tower is Frozen for the first {FREEZE_ITEM_EPOCHS} epochs.")
    item_tower.set_freeze_state(True)
    

    # 💡 [참고] 처음부터 학습할 때는 Item Tower의 LR을 낮추지 않고 동일하게 가는 것도 좋습니다.
    # 만약 Pretrained 임베딩이 많이 깨지는 것을 방지하고 싶다면 현재처럼 0.05배율을 유지하세요.
    item_finetune_lr = cfg.lr * 1
    
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
        {'params': item_tower.parameters(), 'lr': item_finetune_lr, 'weight_decay': cfg.weight_decay}
    ], fused=True, betas=(0.9, 0.98) )
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.05) 
    #scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05) 
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=0.06,
        anneal_strategy='cos',
        div_factor=10.0,      # 시작 lr = 3e-5
        final_div_factor=100.0  # 종료 lr = 3e-6
    )
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
        
        
        if epoch == 8:
            print(f"📈 [Epoch {epoch}] Increasing Item Tower LR multiplier from 0.05x to 0.35x!")
            new_item_base_lr = cfg.lr * 1
            
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
            if epoch >= 50:
                if (epoch - 10) % 1 == 0 or force_mining_next_epoch:
                    print(f"🔍 Mining Global Hard Negatives using cached embeddings...")
                    epoch_hn_pool, hn_metrics = mine_global_hard_negatives(
                        item_embs=norm_item_embeddings,    # 다시 계산 안 하고 캐시본 사용
                        sbert_embs=aligned_sbert_embs,     # 💡 [수정 사항 2] 시맨틱 방어막 텐서 주입
                        fn_threshold=0.85,
                        fn_lower=0.50, # 💡 [수정 사항 3] 시맨틱 유사도 0.85 이상은 오답 풀에서 영구 배제
                        exclusion_top_k=cfg.EX_TOP_K,      # (안전지대)
                        mine_k=160, 
                        batch_size=2048, 
                        device=device
                    )
                    wandb.log(hn_metrics)
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
        
     

        T_sample = 0.5
        hnm_rel = epoch - 18 + 1  # 1-indexed
        #hnm_params = get_hnm_params(hnm_rel)
        T_HN_val = 0#hnm_params["T_HN"]
        beta_val  = 0#hnm_params["beta"]
        print(f"  [HNM Warmup] rel_epoch={hnm_rel}, T_HN={T_HN_val:.2f}, beta={beta_val:.2f}")

        # ------------------- 훈련 -------------------
        avg_loss, force_mining_next_epoch, avg_discard_ratio = train_user_tower_session_sampler_with_intent_point(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower,
            norm_item_embeddings=norm_item_embeddings,
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=epoch_hn_pool,
            scheduler=scheduler,
            hn_scheduler=hn_scheduler,
            T_sample=0.50,        # 고정
            beta=beta_val,        # 워밍업 스케줄
            hn_refresh_interval=500,
            hn_exclusion_top_k=3,
            hn_mine_k=400,
            T_HN=T_HN_val,
            sbert_embs=aligned_sbert_embs
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
        if epoch == 13:
            print(f"🔔 [Checkpoint] Epoch {epoch} completed. Current Recall@20: {current_recall_20:.2f}%")
            base_user_pth =os.path.join(cfg.model_dir, "best_user_tower_from_scratch_base_p.pth")
            base_item_pth= os.path.join(cfg.model_dir, "best_item_tower_from_scratch_base_p.pth")
            
            torch.save(user_tower.state_dict(), base_user_pth)
            torch.save(item_tower.state_dict(), base_item_pth)
        if early_stopping.early_stop:
            print(f"\n🛑 조기 종료 발동! {early_stopping.patience} Epoch 동안 Recall@20이 개선되지 않았습니다.")
            print(f"🏆 최종 최고 Recall@20: {early_stopping.best_score:.2f}%")
            break
            
    print("\n🎉 From-Scratch Pipeline Execution Finished Successfully!")
def train_pipeline_from_checkpoint():
    """
    best_user_tower_from_scratch_base_duorec.pth /
    best_item_tower_from_scratch_base_duorec.pth 에서 로드하여
    Epoch 14부터 HNM을 적용해 재학습하는 파이프라인
    """
    print("🔄 Resuming Training from Base Checkpoint (Epoch 14+ with HNM)...")

    SEQ_LABELS = ['recency_curr', 'week_curr', 'item_id', 'item_price', 'target_week']
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
    # 💡 하이퍼파라미터 세팅 (원본과 동일하게 유지)
    # -----------------------------------------------------------
    cfg.lr = 1e-3
    cfg.epochs = 50
    cfg.HN_K = 150
    cfg.EX_TOP_K = 3
    cfg.soft_penalty_weigh = 1.0
    cfg.hn_scheduled = False
    cfg.batch_size = 512
    cfg.dropout = 0.2
    cfg.lambda_align = 0.05
    FREEZE_ITEM_EPOCHS = 2
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # 재시작 에포크 설정
    START_EPOCH = 14  # 체크포인트 저장 시점(Epoch 13) 다음부터

    # 2. Data & Metadata
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    log_q_tensor = processor.get_logq_probs(device)

    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    val_processor.i_side_arr = processor.i_side_arr

    aligned_sbert_embs = get_or_build_aligned_sbert_embeddings(processor, cfg.base_dir, device)

    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22", aligned_vecs, is_train=False)

    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2",
        name=f"ResumeFromEp13_HNM_lr{cfg.lr}_K{cfg.HN_K}",
        config=cfg.__dict__
    )

    # -----------------------------------------------------------
    # 3. Models Setup & 체크포인트 로드
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, None, log_q_tensor)

    # 체크포인트 경로
    base_user_pth = os.path.join(cfg.model_dir, "best_user_tower_from_scratch_base_p.pth")
    base_item_pth = os.path.join(cfg.model_dir, "best_item_tower_from_scratch_base_p.pth")

    # 저장 경로 (재시작 후 best 모델용)
    save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_dif-sr.pth")
    save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_dif-sr.pth")

    print(f"📂 Loading user tower from: {base_user_pth}")
    user_tower.load_state_dict(torch.load(base_user_pth, map_location=device))

    print(f"📂 Loading item tower from: {base_item_pth}")
    item_tower.load_state_dict(torch.load(base_item_pth, map_location=device))

    print("✅ Checkpoint loaded successfully!")

    # Item Tower는 이미 Epoch 13에서 unfreeze된 상태이므로 그대로 유지
    item_tower.set_freeze_state(False)

    # -----------------------------------------------------------
    # 4. Optimizer & Scheduler 세팅
    # -----------------------------------------------------------
    # Epoch 10에서 item tower LR이 0.05x → 0.15x로 올라간 상태를 반영
    item_finetune_lr = cfg.lr * 0.3
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
        {'params': item_tower.parameters(), 'lr': item_finetune_lr, 'weight_decay': 0.0}
    ], fused=True, betas=(0.9, 0.98) )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 스케줄러: 전체 스텝 기준으로 생성 후, 이미 지난 스텝만큼 fast-forward
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.05)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.06,
        anneal_strategy='cos',
        div_factor=10.0,      # 시작 lr = 3e-5
        final_div_factor=100.0  # 종료 lr = 3e-6
    )
    # Epoch 13까지 진행된 스텝 수만큼 스케줄러를 fast-forward
    steps_per_epoch = len(train_loader)
    elapsed_steps = steps_per_epoch * (START_EPOCH - 1)
    print(f"⏩ Fast-forwarding scheduler by {elapsed_steps} steps (Epoch 1~{START_EPOCH - 1})...")
    for _ in range(elapsed_steps):
        scheduler.step()

    # base_lrs 동기화 (Epoch 10 이후 item lr 변경분 반영)
    if hasattr(scheduler, 'base_lrs'):
        scheduler.base_lrs[1] = item_finetune_lr

    # EarlyStopping: Epoch 13 best score(9.58%)를 초기값으로 설정
    early_stopping = EarlyStopping(patience=7, mode='max')
    early_stopping.best_score = 7.80  # 체크포인트 시점 best 주입
    print(f"📌 EarlyStopping initialized with best_score=9.58% (from Epoch 13 checkpoint)")

    hn_scheduler = None  # cfg.hn_scheduled = False
    force_mining_next_epoch = False
    epoch_hn_pool = None
    current_beta = 0.25

    # -----------------------------------------------------------
    # 5. Training Loop (Epoch 14부터 재시작)
    # -----------------------------------------------------------
    for epoch in range(START_EPOCH, cfg.epochs + 1):

        item_tower.eval()

        with torch.no_grad():
            print(f"📦 [Epoch {epoch}] Caching All Item Embeddings for Training...")
            all_item_embs = item_tower.get_all_embeddings()
            norm_item_embeddings = F.normalize(all_item_embs, p=2, dim=1) # [50000, Dim]
            
            # 10에포크부터는 이 캐시된 임베딩을 마이닝에도 재사용하여 속도 극대화
            if epoch >= 17:
                if (epoch - 10) % 1 == 0 or force_mining_next_epoch:
                    print(f"🔍 Mining Global Hard Negatives using cached embeddings...")
                    epoch_hn_pool, hn_metrics = mine_global_hard_negatives(
                        item_embs=norm_item_embeddings,    # 다시 계산 안 하고 캐시본 사용
                        sbert_embs=aligned_sbert_embs,     # 💡 [수정 사항 2] 시맨틱 방어막 텐서 주입
                        fn_threshold=0.85,
                        fn_lower=0.50, # 💡 [수정 사항 3] 시맨틱 유사도 0.85 이상은 오답 풀에서 영구 배제
                        exclusion_top_k=cfg.EX_TOP_K,      # (안전지대)
                        mine_k=300, 
                        batch_size=2048, 
                        device=device
                    )
                    wandb.log(hn_metrics)
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

        T_sample = 0.5
        hnm_rel = epoch - 17 + 1  # 1-indexed
        hnm_params = get_hnm_params(hnm_rel)
        T_HN_val = hnm_params["T_HN"]
        beta_val  = hnm_params["beta"]
        print(f"  [HNM Warmup] rel_epoch={hnm_rel}, T_HN={T_HN_val:.2f}, beta={beta_val:.2f}")

        # ------------------- 훈련 -------------------
        avg_loss, force_mining_next_epoch, avg_discard_ratio = train_user_tower_session_sampler_with_intent_point(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower,
            norm_item_embeddings=norm_item_embeddings,
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,

            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=epoch_hn_pool,
            scheduler=scheduler,
            hn_scheduler=hn_scheduler,
            T_sample=0.50,        # 고정
            beta=beta_val,        # 워밍업 스케줄
            hn_refresh_interval=0,
            hn_exclusion_top_k=3,
            hn_mine_k=400,
            T_HN=T_HN_val,
            sbert_embs=aligned_sbert_embs
        )

        del norm_item_embeddings
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print(f"📊 [Epoch {epoch}] avg_discard_ratio={avg_discard_ratio:.3f} "
              f"→ next beta={current_beta:.2f}, T_sample={T_sample:.3f}")

        # ------------------- 평가 -------------------
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

        # ------------------- Best Model 저장 & EarlyStopping -------------------
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

if __name__ == "__main__":
    # 5에포크까지 학습했으므로 6번부터 재개
    #run_resume_pipeline(resume_epoch=16, last_best_recall=22.60)
    #run_pipeline_opt_v2()
    #import torch.multiprocessing as mp
    #mp.freeze_support()
    #mp.set_start_method('spawn', force=True)  # Windows 필수
    #analysis_model_and_vectors()
    #cfg = PipelineConfig()
    #JSON_PATH = os.path.join(cfg.base_dir, "filtered_data_reinforced.json")
    #item_dict = load_and_parse_json(JSON_PATH)
    
    # 함수 호출 시 기본 세팅된 9가지 가중치가 자동 적용됩니다.
    #ids, embs = build_aspect_item_embeddings(item_dict)
    
    #analyze_semantic_similarities(ids, embs, sample_size=5000)
    #resume_pipeline_session_weights()
    #run_resume_pipeline_v2()
    #train_pipeline_from_checkpoint()
    train_pipeline_from_scratch()