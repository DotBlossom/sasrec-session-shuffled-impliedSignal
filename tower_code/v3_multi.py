def multi_positive_supcon_logq_loss(
    last_user_emb,     
    item_tower_emb,      
    target_ids,          
    target_mask,         
    log_q_tensor,        
    hard_neg_ids=None,   
    temperature=0.1, 
    lambda_logq=1.0,         
    alpha=1.0,               
    margin=0.00,             
    return_metrics=False     
):
    B, max_T = target_ids.shape
    device = last_user_emb.device
    
    # -----------------------------------------------------------
    # 1. In-batch 유사도 연산 (NaN 방지를 위해 eps 추가)
    # -----------------------------------------------------------
    flat_targets = target_ids.reshape(-1) 
    valid_flat_mask = target_mask.reshape(-1)
    
    # [B*max_T, dim]
    batch_item_emb = item_tower_emb[flat_targets] 
    
    # 💡 [방어] 내적 전 임베딩에 아주 작은 값을 더해 Norm이 0이 되는 극단적 상황 방지
    sim_matrix = torch.matmul(last_user_emb, batch_item_emb.T) 
    
    # -----------------------------------------------------------
    # 2. Multi-Positive Mask 생성 (메모리 효율화)
    # -----------------------------------------------------------
    # [B, B * max_T]
    pos_mask = torch.eq(target_ids.unsqueeze(2), flat_targets.unsqueeze(0)).any(dim=1)
    pos_mask = pos_mask & valid_flat_mask.unsqueeze(0)
    
    # 💡 [방어] 정답이 없는 유저가 배치에 섞여 있을 경우를 위한 마스킹
    user_has_pos = pos_mask.any(dim=1) 
    if not user_has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Margin 적용
    if margin > 0:
        sim_matrix = sim_matrix - (pos_mask.float() * margin)
        
    # 💡 [방어] Temperature로 나누기 전 로짓 범위 제한 (FP16 오버플로우 방지)
    logits = sim_matrix / temperature

    # -----------------------------------------------------------
    # 3. LogQ 보정 및 마스킹
    # -----------------------------------------------------------
    if lambda_logq > 0.0:
        # 💡 [방어] LogQ Clamp 범위를 조금 더 넓혀 유연성 확보
        batch_log_q = log_q_tensor[flat_targets].clamp(min=-30.0, max=30.0)
        logits = logits - (batch_log_q.unsqueeze(0) * lambda_logq)

    # 💡 [방어] -inf 대신 충분히 작은 값을 사용하여 log_softmax의 안정성 확보
    logits.masked_fill_(~valid_flat_mask.unsqueeze(0), -1e4) 

    # -----------------------------------------------------------
    # 4. Final Loss (수치 안정화 로직 포함)
    # -----------------------------------------------------------
    # 💡 [핵심 방어] Max-Normalization 적용 (이전 코드의 F.cross_entropy 효과 재현)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # Log Softmax
    log_probs = F.log_softmax(logits, dim=1)
    
    # 정답 위치의 log_prob만 합산 (eps를 더해 log(0) 방지)
    pos_log_probs = (log_probs * pos_mask.float()).sum(dim=1)
    
    # 💡 [방어] 분모에 아주 작은 값(1e-8)을 더해 0 나누기 방지
    num_positives = pos_mask.sum(dim=1).float().clamp(min=1e-8)
    
    # 유효한 유저(정답이 있는 유저)에 대해서만 평균 로스 계산
    loss = - (pos_log_probs[user_has_pos] / num_positives[user_has_pos]).mean()
    
    if return_metrics:
        with torch.no_grad():
            metrics = {
                'sim/pos': (sim_matrix[pos_mask] + (margin if margin > 0 else 0)).mean().item() if pos_mask.any() else 0,
                'num_positives_per_user': num_positives[user_has_pos].mean().item(),
                'logits/max': logits_max.mean().item()
            }
        return loss, metrics
        
    return loss

import torch
import torch.nn.functional as F

def multi_positive_supcon_logq_loss_v2(
    last_user_emb,     
    item_tower_emb,      
    target_ids,          
    target_mask,
    input_ids,           # 💡 [New] 전략 C를 위한 인풋 아이템 시퀀스 [B, max_len]
    input_padding_mask,  # 💡 [New] 인풋의 패딩 마스크 [B, max_len]
    log_q_tensor,        
    hard_neg_ids=None,   
    temperature=0.1, 
    lambda_logq=1.0,         
    alpha=1.0,               
    margin=0.00,             
    return_metrics=False     
):
    B, max_T = target_ids.shape
    device = last_user_emb.device
    
    # -----------------------------------------------------------
    # 1. In-batch 유사도 연산
    # -----------------------------------------------------------
    flat_targets = target_ids.reshape(-1) 
    valid_flat_mask = target_mask.reshape(-1)
    
    batch_item_emb = item_tower_emb[flat_targets] 
    sim_matrix = torch.matmul(last_user_emb, batch_item_emb.T) 
    
    # -----------------------------------------------------------
    # 2. Masks 생성 (Positive & Past Interaction)
    # -----------------------------------------------------------
    # ✅ [전략 A] Positive Mask: 배치 내에 나와 정답이 같은 아이템 매칭
    pos_mask = torch.eq(target_ids.unsqueeze(2), flat_targets.unsqueeze(0)).any(dim=1)
    pos_mask = pos_mask & valid_flat_mask.unsqueeze(0)
    
    user_has_pos = pos_mask.any(dim=1) 
    if not user_has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # ✅ [전략 C] Past Interaction Mask: 인풋 시퀀스에 있는 아이템 매칭
    valid_input_mask = ~input_padding_mask # [B, max_L] (패딩이 아닌 실제 아이템)
    
    # input_ids[B, max_L, 1] == flat_targets[1, 1, B*max_T] -> [B, max_L, B*max_T]
    past_overlap = torch.eq(input_ids.unsqueeze(2), flat_targets.unsqueeze(0).unsqueeze(0))
    past_overlap = past_overlap & valid_input_mask.unsqueeze(2) # 유효한 인풋만 비교
    
    # [B, B*max_T]
    past_mask = past_overlap.any(dim=1)

    # Margin 적용 및 Temperature 스케일링
    if margin > 0:
        sim_matrix = sim_matrix - (pos_mask.float() * margin)
    logits = sim_matrix / temperature

    # -----------------------------------------------------------
    # 3. LogQ 보정
    # -----------------------------------------------------------
    if lambda_logq > 0.0:
        batch_log_q = log_q_tensor[flat_targets].clamp(min=-30.0, max=30.0)
        logits = logits - (batch_log_q.unsqueeze(0) * lambda_logq)

    # -----------------------------------------------------------
    # 4. Final Loss (전략 B: 분모 내 Positive/Past 마스킹 적용)
    # -----------------------------------------------------------
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits_stable = logits - logits_max.detach()
    
    exp_logits = torch.exp(logits_stable)
    
    # 💡 [핵심] Negative Mask: 정답도 아니고(A, B), 과거에 본 것도 아니며(C), 유효한 배치 아이템
    neg_mask = (~pos_mask) & (~past_mask) & valid_flat_mask.unsqueeze(0)
    
    # 분모 계산: 오직 "진짜 네거티브" 아이템들의 Exp 합산
    neg_exp_sum = (exp_logits * neg_mask.float()).sum(dim=1, keepdim=True)
    
    # 각 포지티브 아이템의 분모 = 자기 자신 + 네거티브 합 (다른 포지티브 배제됨)
    pos_denominators = exp_logits + neg_exp_sum
    
    # Log Softmax 수동 계산
    log_probs = logits_stable - torch.log(pos_denominators + 1e-8)
    
    # 정답 위치의 log_prob만 합산
    pos_log_probs = (log_probs * pos_mask.float()).sum(dim=1)
    num_positives = pos_mask.sum(dim=1).float().clamp(min=1e-8)
    
    # 유효한 유저에 대해서만 평균 로스 계산
    loss = - (pos_log_probs[user_has_pos] / num_positives[user_has_pos]).mean()
    
    if return_metrics:
        with torch.no_grad():
            metrics = {
                'sim/pos': (sim_matrix[pos_mask] + (margin if margin > 0 else 0)).mean().item() if pos_mask.any() else 0,
                'num_positives_per_user': num_positives[user_has_pos].mean().item(),
                'logits/max': logits_max.mean().item()
            }
        return loss, metrics
        
    return loss

import random
import numpy as np
import pandas as pd
import torch
import datetime
from torch.utils.data import DataLoader, Dataset


from v1_usertower_train import PipelineConfig, evaluate_model, load_aligned_pretrained_embeddings, load_item_metadata_hashed, load_item_tower_state_dict, prepare_features, setup_environment, setup_models


import random
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SASRecDataset_MultiPositive_sampler(Dataset):
    def __init__(self, processor, global_now_str="2020-09-22", max_len=30, max_target_len=10, is_train=True):
        self.processor = processor
        self.max_len = max_len
        self.max_target_len = max_target_len 
        self.is_train = is_train
        
        global_now_dt = pd.to_datetime(global_now_str)
        self.now_ordinal = global_now_dt.toordinal()
        self.now_week = global_now_dt.isocalendar().week
        
        self.user_ids = []
        for uid in processor.user_ids:
            if len(processor.seqs.loc[uid, 'sequence_ids']) > 1:
                self.user_ids.append(uid)
        
        print(f"✅ Filtered Dataset: {len(self.user_ids)} users remaining")
    
    def _shuffle_within_session(self, seq_raw, time_deltas_raw):
        if len(seq_raw) <= 1:
            return seq_raw, time_deltas_raw
        grouped_indices = []
        current_group = [0]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == 0:
                current_group.append(i)
            else:
                grouped_indices.append(current_group)
                current_group = [i]
        if current_group:
            grouped_indices.append(current_group)
            
        shuffled_indices = []
        for group in grouped_indices:
            if len(group) > 1 and random.random() < 0.5:
                random.shuffle(group)
            shuffled_indices.extend(group)
            
        shuffled_seq = [seq_raw[i] for i in shuffled_indices]
        shuffled_deltas = [time_deltas_raw[i] for i in shuffled_indices] 
        return shuffled_seq, shuffled_deltas

    def _build_sample(self, user_id, u_mapped_id, input_indices, target_indices, seq_raw, time_deltas_raw):
        """인덱스를 받아 실제 텐서 딕셔너리를 생성하는 헬퍼 함수"""
        input_seq_raw = [seq_raw[i] for i in input_indices]
        input_deltas = [time_deltas_raw[i] for i in input_indices]
        target_seq_raw = [seq_raw[i] for i in target_indices]
        
        if self.is_train:
            input_seq_raw, input_deltas = self._shuffle_within_session(input_seq_raw, input_deltas)
            
        bins = np.array([0, 3, 7, 14, 30, 60, 180, 330, 395])
        time_buckets = np.digitize(input_deltas, bins, right=False).tolist()
        
        input_seq = [self.processor.item2id.get(item, 0) for item in input_seq_raw]
        target_seq = [self.processor.item2id.get(item, 0) for item in target_seq_raw]

        # Input Padding
        input_seq = input_seq[-self.max_len:]
        time_buckets = time_buckets[-self.max_len:]
        pad_len = self.max_len - len(input_seq)
        
        input_padded = [0] * pad_len + input_seq
        time_padded = [0] * pad_len + time_buckets
        padding_mask = [True] * pad_len + [False] * len(input_seq)

        # Target Padding
        if self.is_train and len(target_seq) > 0:
            target_seq = target_seq[:self.max_target_len] 
            t_pad_len = self.max_target_len - len(target_seq)
            target_padded = target_seq + [0] * t_pad_len
            target_mask = [True] * len(target_seq) + [False] * t_pad_len
        else:
            target_padded = [0] * self.max_target_len
            target_mask = [False] * self.max_target_len

        # Item Side Info
        item_side_info = self.processor.i_side_arr[input_padded]
        type_ids, color_ids, graphic_ids, section_ids = item_side_info[:, 0], item_side_info[:, 1], item_side_info[:, 2], item_side_info[:, 3]

        # User Features
        u_buckets = self.processor.u_bucket_arr[u_mapped_id]
        u_cats = self.processor.u_cat_arr[u_mapped_id]
        u_conts = self.processor.u_cont_arr[u_mapped_id]
        last_ordinal = self.processor.u_last_date_arr[u_mapped_id]
    
        if last_ordinal == 0: 
            recency_offset, current_week = 365, self.now_week
        else:
            recency_offset = max(0, min(365, self.now_ordinal - last_ordinal))
            current_week = datetime.date.fromordinal(last_ordinal).isocalendar().week
            
        return {
            'user_ids': user_id,
            'item_ids': torch.tensor(input_padded, dtype=torch.long),
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.bool),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'time_bucket_ids': torch.tensor(time_padded, dtype=torch.long),
            'type_ids': torch.tensor(type_ids, dtype=torch.long),
            'color_ids': torch.tensor(color_ids, dtype=torch.long),
            'graphic_ids': torch.tensor(graphic_ids, dtype=torch.long),
            'section_ids': torch.tensor(section_ids, dtype=torch.long),
            'age_bucket': torch.tensor(u_buckets[0], dtype=torch.long),
            'price_bucket': torch.tensor(u_buckets[1], dtype=torch.long),
            'cnt_bucket': torch.tensor(u_buckets[2], dtype=torch.long),
            'recency_bucket': torch.tensor(u_buckets[3], dtype=torch.long),
            'channel_ids': torch.tensor(u_cats[0], dtype=torch.long),
            'club_status_ids': torch.tensor(u_cats[1], dtype=torch.long),
            'news_freq_ids': torch.tensor(u_cats[2], dtype=torch.long),
            'fn_ids': torch.tensor(u_cats[3], dtype=torch.long),
            'active_ids': torch.tensor(u_cats[4], dtype=torch.long),
            'cont_feats': torch.tensor(u_conts, dtype=torch.float32),
            'recency_offset': torch.tensor(recency_offset, dtype=torch.long),
            'current_week': torch.tensor(current_week, dtype=torch.long)
        }

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
        seq_raw = self.processor.seqs.loc[user_id, 'sequence_ids']
        time_deltas_raw = self.processor.seqs.loc[user_id, 'sequence_deltas']
        
        sessions = []
        current_session = [0]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == 0:
                current_session.append(i)
            else:
                sessions.append(current_session)
                current_session = [i]
        if current_session: 
            sessions.append(current_session)

        samples = []
        
        # 💡 1. Original Sample (필수)
        if self.is_train:
            if len(sessions) > 1:
                target_orig = sessions[-1]
                input_orig = [idx for s in sessions[:-1] for idx in s]
            else:
                target_orig = sessions[0][1:]
                input_orig = [sessions[0][0]]
        else:
            target_orig = []
            input_orig = list(range(len(seq_raw)))
            
        samples.append(self._build_sample(user_id, u_mapped_id, input_orig, target_orig, seq_raw, time_deltas_raw))

        # 💡 2. Random Crop Sample (Train 시에만, 세션이 충분히 길 때만 추가)
        if self.is_train:
            num_sessions = len(sessions)
            if num_sessions >= 3:
                # [S0, S1, S2, S3] 중 타겟을 S1이나 S2로 랜덤 설정
                k = random.randint(1, num_sessions - 2)
                target_crop = sessions[k]
                input_crop = [idx for s in sessions[:k] for idx in s]
                samples.append(self._build_sample(user_id, u_mapped_id, input_crop, target_crop, seq_raw, time_deltas_raw))
            elif num_sessions == 1 and len(seq_raw) >= 3:
                # 단일 세션이지만 상품이 3개 이상인 경우 (예: [A, B, C] -> Input: [A, B], Target: [C])
                split_idx = random.randint(2, len(seq_raw) - 1)
                target_crop = sessions[0][split_idx:]
                input_crop = sessions[0][:split_idx]
                samples.append(self._build_sample(user_id, u_mapped_id, input_crop, target_crop, seq_raw, time_deltas_raw))

        # ⚠️ 반환값이 Dict가 아니라 List of Dicts 입니다!
        return samples
    
from torch.utils.data._utils.collate import default_collate

def multipositive_crop_collate_fn(batch):
    """
    batch: [[sample1_orig, sample1_crop], [sample2_orig], [sample3_orig, sample3_crop], ...]
    이 이중 리스트를 단일 리스트로 풀어서 PyTorch 기본 collate에 넘겨줍니다.
    """
    flat_batch = []
    for user_samples in batch:
        # user_samples는 리스트(길이 1 또는 2)입니다.
        flat_batch.extend(user_samples)
        
    # 평탄화된 샘플들을 하나의 텐서 배치로 합쳐줌
    return default_collate(flat_batch)
#==============================================================


class SASRecDataset_MultiPositive(Dataset):
    def __init__(self, processor, global_now_str="2020-09-22", max_len=30, max_target_len=10, is_train=True):
        self.processor = processor
        self.max_len = max_len
        self.max_target_len = max_target_len # 💡 [신규] 멀티 타겟 패딩 길이
        self.is_train = is_train
        self.user_ids = processor.user_ids
        
        global_now_dt = pd.to_datetime(global_now_str)
        self.now_ordinal = global_now_dt.toordinal()
        self.now_week = global_now_dt.isocalendar().week
        self.user_ids = []
        for uid in processor.user_ids:
            if len(processor.seqs.loc[uid, 'sequence_ids']) > 1:
                self.user_ids.append(uid)
        
        print(f"✅ Filtered Dataset: {len(self.user_ids)} users remaining (Original: {len(processor.user_ids)})")
    
    def _shuffle_within_session(self, seq_raw, time_deltas_raw):
        # (기존에 작성해주신 완벽한 헬퍼 함수 그대로 사용)
        if len(seq_raw) <= 1:
            return seq_raw, time_deltas_raw
        grouped_indices = []
        current_group = [0]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == 0:
                current_group.append(i)
            else:
                grouped_indices.append(current_group)
                current_group = [i]
        if current_group:
            grouped_indices.append(current_group)
            
        shuffled_indices = []
        for group in grouped_indices:
            if len(group) > 1 and random.random() < 0.5:
                random.shuffle(group)
            shuffled_indices.extend(group)
            
        shuffled_seq = [seq_raw[i] for i in shuffled_indices]
        shuffled_deltas = [time_deltas_raw[i] for i in shuffled_indices] 
        return shuffled_seq, shuffled_deltas

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
        
        seq_raw = self.processor.seqs.loc[user_id, 'sequence_ids']
        time_deltas_raw = self.processor.seqs.loc[user_id, 'sequence_deltas']
        
        # =========================================================
        # 1. 시퀀스를 세션 단위로 그룹화 (delta=0 기준)
        # =========================================================
        sessions = []
        current_session = [0]
        for i in range(1, len(seq_raw)):
            if time_deltas_raw[i] == 0:
                current_session.append(i)
            else:
                sessions.append(current_session)
                current_session = [i]
        if current_session: 
            sessions.append(current_session)

        # =========================================================
        # 2. Next-Session Split: 마지막 세션은 Target, 나머지는 Input
        # =========================================================
        if self.is_train:
            if len(sessions) > 1:
                # [전략 B] 세션이 여러 개 -> 마지막 세션이 타겟, 나머지가 인풋
                target_indices = sessions[-1]
                input_indices = [idx for s in sessions[:-1] for idx in s]
            else:
                # [전략 A] 세션이 1개지만 상품은 n개 (init에서 1개인 유저는 걸렀으므로 무조건 2개 이상)
                # -> 첫 번째 상품이 인풋, 나머지가 타겟
                target_indices = sessions[0][1:]
                input_indices = [sessions[0][0]]
                
            input_seq_raw = [seq_raw[i] for i in input_indices]
            input_deltas = [time_deltas_raw[i] for i in input_indices]
            target_seq_raw = [seq_raw[i] for i in target_indices]
            
            # 셔플 적용
            input_seq_raw, input_deltas = self._shuffle_within_session(input_seq_raw, input_deltas)
        else:
            # 평가 시에는 전체가 인풋 (Valid Target은 별도 파일 사용)
            input_seq_raw, input_deltas = seq_raw, time_deltas_raw
            target_seq_raw = []

        # 매핑 및 타임 버킷화
        bins = np.array([0, 3, 7, 14, 30, 60, 180, 330, 395])
        time_buckets = np.digitize(input_deltas, bins, right=False).tolist()
        
        input_seq = [self.processor.item2id.get(item, 0) for item in input_seq_raw]
        target_seq = [self.processor.item2id.get(item, 0) for item in target_seq_raw]

        # =========================================================
        # 3. Input & Target Padding
        # =========================================================
        # Input: Left Padding (max_len)
        input_seq = input_seq[-self.max_len:]
        time_buckets = time_buckets[-self.max_len:]
        pad_len = self.max_len - len(input_seq)
        
        input_padded = [0] * pad_len + input_seq
        time_padded = [0] * pad_len + time_buckets
        padding_mask = [True] * pad_len + [False] * len(input_seq)

        # Target: Right Padding (max_target_len) 💡 [핵심 변경]
        if self.is_train:
            target_seq = target_seq[:self.max_target_len] # 극단적으로 긴 세션 방어
            t_pad_len = self.max_target_len - len(target_seq)
            target_padded = target_seq + [0] * t_pad_len
            target_mask = [True] * len(target_seq) + [False] * t_pad_len
        else:
            target_padded = [0] * self.max_target_len
            target_mask = [False] * self.max_target_len

        # =========================================================
        # 4. Item Side Info Lookup (Sequence)
        # =========================================================
        # padding(0)인 경우 Lookup 배열의 0번째 인덱스(0,0,0,0)를 가져옴
        item_side_info = self.processor.i_side_arr[input_padded]
        
        type_ids = item_side_info[:, 0]
        color_ids = item_side_info[:, 1]
        graphic_ids = item_side_info[:, 2]
        section_ids = item_side_info[:, 3]

        # Padding Mask (True면 Transformer에서 무시)
        padding_mask = [True] * pad_len + [False] * len(input_seq)

        # =========================================================
        # 5. User Features Lookup (Static)
        # =========================================================
        u_buckets = self.processor.u_bucket_arr[u_mapped_id]
        u_cats = self.processor.u_cat_arr[u_mapped_id]
        u_conts = self.processor.u_cont_arr[u_mapped_id]

        # ====================
        # global time 
        # =====================
        last_ordinal = self.processor.u_last_date_arr[u_mapped_id]
    
        if last_ordinal == 0: # 패딩이거나 구매 이력이 아예 없는 경우
            recency_offset = 365
            current_week = self.now_week # 예비용 기본값
        else:
            # 1. Recency Offset (기존 유지: 오늘로부터 얼마나 낡았는가?)
            recency_offset = self.now_ordinal - last_ordinal
            recency_offset = max(0, min(365, recency_offset))
            
            # 🔥 2. Current Week (핵심 수정: 유저의 마지막 구매 시점의 주차!)
            last_date = datetime.date.fromordinal(last_ordinal)
            current_week = last_date.isocalendar().week
        return {
            'user_ids': user_id,
            'item_ids': torch.tensor(input_padded, dtype=torch.long),
            # 💡 Shape이 [max_target_len]인 1D 텐서가 됩니다. 배치로 묶이면 [B, max_target_len]
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.bool), # 정답/패딩 구분용
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'time_bucket_ids': torch.tensor(time_padded, dtype=torch.long),
            
            # Item Side Info
            'type_ids': torch.tensor(type_ids, dtype=torch.long),
            'color_ids': torch.tensor(color_ids, dtype=torch.long),
            'graphic_ids': torch.tensor(graphic_ids, dtype=torch.long),
            'section_ids': torch.tensor(section_ids, dtype=torch.long),
            
            # User Buckets
            'age_bucket': torch.tensor(u_buckets[0], dtype=torch.long),
            'price_bucket': torch.tensor(u_buckets[1], dtype=torch.long),
            'cnt_bucket': torch.tensor(u_buckets[2], dtype=torch.long),
            'recency_bucket': torch.tensor(u_buckets[3], dtype=torch.long),
            
            # User Categoricals
            'channel_ids': torch.tensor(u_cats[0], dtype=torch.long),
            'club_status_ids': torch.tensor(u_cats[1], dtype=torch.long),
            'news_freq_ids': torch.tensor(u_cats[2], dtype=torch.long),
            'fn_ids': torch.tensor(u_cats[3], dtype=torch.long),
            'active_ids': torch.tensor(u_cats[4], dtype=torch.long),
            
            # User Continuous
            'cont_feats': torch.tensor(u_conts, dtype=torch.float32),
            
            # 💡 [추가] Global Context (모델의 Early Injection 입력용)
            'recency_offset': torch.tensor(recency_offset, dtype=torch.long),
            'current_week': torch.tensor(current_week, dtype=torch.long)
        }

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

def train_user_tower_multi_positive(
    epoch, model, item_tower, log_q_tensor, dataloader, optimizer, scaler, cfg, device, 
    hard_neg_pool_tensor=None, scheduler=None, seq_labels=None, static_labels=None
):
    """Next-Session Multi-Positive 학습을 위한 단일 에포크 훈련 함수"""
    model.train()
    total_loss_accum = 0.0
    
    accumulation_steps = 1
    seq_labels = seq_labels or []
    static_labels = static_labels or []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # -------------------------------------------------------
        # 1. Data Unpacking (Target 차원 변경 주의)
        # -------------------------------------------------------
        item_ids = batch['item_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        time_bucket_ids = batch['time_bucket_ids'].to(device)
        
        # 💡 [핵심] Multi-Positive 타겟 (B, max_target_len)
        target_ids = batch['target_ids'].to(device)
        target_mask = batch['target_mask'].to(device) 
        
        type_ids = batch['type_ids'].to(device)
        color_ids = batch['color_ids'].to(device)
        graphic_ids = batch['graphic_ids'].to(device)
        section_ids = batch['section_ids'].to(device)
        
        age_bucket = batch['age_bucket'].to(device)
        price_bucket = batch['price_bucket'].to(device)
        cnt_bucket = batch['cnt_bucket'].to(device)
        recency_bucket = batch['recency_bucket'].to(device)
        
        channel_ids = batch['channel_ids'].to(device)
        club_status_ids = batch['club_status_ids'].to(device)
        news_freq_ids = batch['news_freq_ids'].to(device)
        fn_ids = batch['fn_ids'].to(device)
        active_ids = batch['active_ids'].to(device)
        cont_feats = batch['cont_feats'].to(device)
        
        recency_offset = batch['recency_offset'].to(device)
        current_week = batch['current_week'].to(device)
        
        if 'pretrained_vecs' in batch:
            pretrained_vecs = batch['pretrained_vecs'].to(device)
        else:
            pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device)
            
        forward_kwargs = {
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
            'padding_mask': padding_mask,
            'training_mode': True
        }

        # -------------------------------------------------------
        # 2. Forward & Intent Vector Extraction (AMP)
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            # output shape: [B, Seq_len, d_model]
            output = model(**forward_kwargs) 
            batch_size = item_ids.size(0)
            
            # 💡 [핵심] Padding을 무시하고, 실제 정보가 들어간 "가장 마지막 타임스텝" 1개만 추출
            valid_lengths = (~padding_mask).sum(dim=1)
            last_indices = (valid_lengths - 1).clamp(min=0)
            batch_range = torch.arange(batch_size, device=device)
            
            # last_user_emb shape: [B, d_model]
            last_user_emb = output[batch_range, last_indices]
            last_user_emb = F.normalize(last_user_emb, p=2, dim=1)
            
            # 전체 아이템 임베딩 로드 [Total_Items, d_model]
            all_item_emb = item_tower.get_all_embeddings()
            norm_item_embeddings = F.normalize(all_item_emb, p=2, dim=1)
            
            # -------------------------------------------------------
            # 3. Multi-Positive SupCon Loss 계산
            # -------------------------------------------------------
            # 방어 코드: 실제 타겟이 존재하는 유효한 배치만 처리
            if last_user_emb.size(0) > 0:
                # 💡 어제 완성한 SupCon Loss 호출
                loss, b_metrics = multi_positive_supcon_logq_loss_v2(
                    last_user_emb=last_user_emb,
                    item_tower_emb=norm_item_embeddings,
                    target_ids=target_ids,
                    target_mask=target_mask,
                    # 👇 [전략 C 적용을 위해 추가된 파라미터] 👇
                    input_ids=item_ids,               # 현재 배치의 입력 시퀀스 [B, max_len]
                    input_padding_mask=padding_mask,  # 입력 시퀀스의 패딩 마스크 [B, max_len]
                    
                    log_q_tensor=log_q_tensor,
                    hard_neg_ids=None, # 필요시 하드 네거티브 텐서 연동
                    temperature=0.1, 
                    lambda_logq=cfg.lambda_logq, 
                    alpha=1.0, 
                    margin=0.00, # 초기에는 0으로 두고 모델 안정화 후 올리는 것을 권장
                    return_metrics=True
                )
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                b_metrics = {}

            scaled_loss = loss / accumulation_steps

        # -------------------------------------------------------
        # 4. Backward & Optimizer Step
        # -------------------------------------------------------
        scaler.scale(scaled_loss).backward()

        if ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()

        # -------------------------------------------------------
        # 5. Logging
        # -------------------------------------------------------
        total_loss_accum += loss.item()
        
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Pos(U)': f"{b_metrics.get('num_positives_per_user', 0):.1f}" # 평균 정답 개수 모니터링
        })
        
        if batch_idx % 100 == 0:
            wandb_log_dict = {
                "Train/Loss": loss.item()
            }
            if 'sim/pos' in b_metrics:
                wandb_log_dict["Train/Sim_Pos"] = b_metrics['sim/pos']
            wandb.log(wandb_log_dict)

    # 에포크 종료 후 처리
    avg_loss = total_loss_accum / len(dataloader)

    # Gate Weights Logging (기존과 동일)
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

    print(f"🏁 Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f}")
    return avg_loss

import numpy as np
import torch

def dataset_peek(dataset, processor, sample_n=500):
    """
    Multi-Positive Dataset 전용 통계 및 정합성 검수 함수
    - 세션 1개인 유저 비율 계산
    - 세션별 평균 길이(아이템 수) 계산
    """
    print(f"\n🧐 [Data Peek] Analyzing first {sample_n} samples for Session Statistics...")
    
    single_session_users = 0
    total_session_lengths = []
    total_session_counts = []
    
    # 통계 계산을 위한 샘플링 (에러 방지를 위해 실제 데이터셋 길이와 비교)
    num_to_sample = min(sample_n, len(dataset))
    
    for i in range(num_to_sample):
        # __getitem__ 호출 (내부적으로 셔플 및 세션 분리가 수행됨)
        # ⚠️ 만약 여기서 IndexError가 나면 전처리 단계의 문제임
        user_id = dataset.user_ids[i]
        
        # 원본 데이터(셔플 전)를 기준으로 통계를 내기 위해 processor에서 직접 참조
        seq_raw = processor.seqs.loc[user_id, 'sequence_ids']
        time_deltas_raw = processor.seqs.loc[user_id, 'sequence_deltas']
        
        # 세션 분리 로직 (delta=0 기준)
        sessions = []
        current_session = [0]
        for j in range(1, len(seq_raw)):
            if time_deltas_raw[j] == 0:
                current_session.append(j)
            else:
                sessions.append(current_session)
                current_session = [j]
        if current_session: 
            sessions.append(current_session)
            
        # 통계 누적
        num_sessions = len(sessions)
        if num_sessions == 1:
            single_session_users += 1
            
        total_session_counts.append(num_sessions)
        for s in sessions:
            total_session_lengths.append(len(s))

    # 결과 요약 출력
    avg_sessions_per_user = np.mean(total_session_counts)
    avg_items_per_session = np.mean(total_session_lengths)
    single_session_ratio = (single_session_users / num_to_sample) * 100

    print(f"\n ─── Global Statistics (Sample size: {num_to_sample}) ───")
    print(f"  • Single-Session Users: {single_session_users} ({single_session_ratio:.1f}%)")
    print(f"  • Avg Sessions per User: {avg_sessions_per_user:.2f}")
    print(f"  • Avg Items per Session: {avg_items_per_session:.2f}")

    # 개별 샘플 정합성 확인 (index 0 기준)
    print(f"\n ─── Integrity Check (Sample [0] - Augmented) ───")
    samples = dataset[0] # 이제 리스트가 반환됩니다!
    
    for idx, sample in enumerate(samples):
        sample_type = "Original" if idx == 0 else "Random Cropped"
        print(f"\n  [{sample_type} Sample]")
        
        # 텐서인 경우 리스트로 변환 (안전 처리)
        ids = sample['item_ids'].tolist() if torch.is_tensor(sample['item_ids']) else sample['item_ids']
        targets = sample['target_ids'].tolist() if torch.is_tensor(sample['target_ids']) else sample['target_ids']
        target_mask = sample['target_mask'].tolist() if torch.is_tensor(sample['target_mask']) else sample['target_mask']
        
        actual_targets = [t for t, m in zip(targets, target_mask) if m]
        
        # 패딩(0)을 제외한 실제 인풋 확인
        valid_inputs = [x for x in ids if x != 0]
        
        print(f"  • User ID: {sample['user_ids']}")
        print(f"  • Input Tail: {valid_inputs[-5:] if valid_inputs else []}")
        print(f"  • Actual Multi-Positives: {actual_targets}")
        
        if len(actual_targets) == 0:
            print("  ⚠️ Warning: No targets found for this sample.")
        else:
            print(f"  ✅ Target Session Size: {len(actual_targets)}")

    print(f"\n🧐 [Data Peek] Statistics Analysis Complete.\n")

import os
import math
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
# (미리 정의된 설정, 모델 셋업, 전처리 함수 등은 기존과 동일하게 임포트되어 있다고 가정)

def create_dataloaders(processor, cfg: PipelineConfig, global_now_str, aligned_pretrained_vecs=None, is_train=True):
    """Dataset 및 DataLoader 인스턴스화"""
    mode_str = "Train" if is_train else "Validation"
    print(f"\n📦 [Phase 3-2] Creating {mode_str} DataLoaders...")
    
    # 💡 1. is_train 파라미터 전달
    dataset = SASRecDataset_MultiPositive(processor, global_now_str = global_now_str, max_len=cfg.max_len,max_target_len=cfg.max_target_len, is_train=is_train)
    
    # Dataset 인스턴스에 정렬된 pretrained vector 룩업 테이블 주입
    dataset.pretrained_lookup = aligned_pretrained_vecs 
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        #collate_fn=multipositive_crop_collate_fn,
        shuffle=is_train, 
        num_workers=0, 
        pin_memory=True,
        drop_last=is_train 
    )
    
    print(f"✅ {mode_str} Loader Ready: {len(loader)} batches/epoch")
    return loader


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

# 💡 [새로운 스케줄러] LR을 오랫동안 높게 유지하다가 후반부에만 계단식으로 감소시킵니다.
def get_step_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, milestones=[0.6, 0.8], gamma=0.5):
    """
    웜업 후 LR을 100%로 유지하다가, 전체 학습의 60%, 80% 지점에서 LR을 gamma만큼 감소시킵니다.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step) / float(max(1, num_training_steps))
        factor = 1.0
        for milestone in milestones:
            if progress >= milestone:
                factor *= gamma
        return factor
    return LambdaLR(optimizer, lr_lambda)


def run_pipeline_multipos_v3():
    """Multi-Positive (Next-Session) 기반 훈련 파이프라인 엔트리 포인트"""
    print("🚀 Starting User Tower Training Pipeline [Multi-Positive V3]...")
    
    SEQ_LABELS = ['item_id', 'time', 'type', 'color', 'graphic', 'section', 'recency_curr', 'week_curr']
    STATIC_LABELS = [
        'age', 'price', 'cnt', 'recency',      
        'channel', 'club', 'news', 'fn', 'active', 
        'cont_price_std', 'cont_last_diff', 'cont_repurch', 'cont_weekend' 
    ] 
    
    # -----------------------------------------------------------
    # 1. Config & Env Setup
    # -----------------------------------------------------------
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    
    # 💡 [핵심 변동] Hyper-parameters 업데이트
    cfg.lr = 2e-3
    cfg.batch_size = 2048 # In-batch Negative 풀 확장을 위해 대규모 배치 적용
    cfg.max_target_len = 10 # Multi-Positive 타겟의 최대 길이 (Dataset에 전달용)
    
    # Item metadata cfg
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # -----------------------------------------------------------
    # 2. Data & DataLoader Setup
    # -----------------------------------------------------------
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    item_state_dict = load_item_tower_state_dict(cfg.model_dir, cfg.item_tower_pth_name, device)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    
    # ⚠️ [안전 체크] 내부의 create_dataloaders가 SASRecDataset_MultiPositive를 호출하도록 
    # 사전에 업데이트되어 있어야 합니다. (max_target_len=cfg.max_target_len 전달 필수)
    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22", aligned_vecs, is_train=False)
    
    #dataset_peek(train_loader.dataset, processor)
    
    # W&B Logging Init
    wandb.init(
        project="SASRec-User-Tower-MultiPositive-v3", 
        name=f"run_bs{cfg.batch_size}_lr{cfg.lr}_ep{cfg.epochs}_supcon_multiprovider", 
        config=cfg.__dict__
    )
    
    # -----------------------------------------------------------
    # 3. Models & Optimizer Setup (Epoch 1)
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, item_state_dict, log_q_tensor)
    item_tower.init_from_pretrained(aligned_vecs.to(device))
    item_tower.set_freeze_state(True)
    print(f"❄️ Epoch 1: Item Tower FROZEN! (User Tower LR: {cfg.lr})")
    
    optimizer = torch.optim.AdamW(user_tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 배치 사이즈가 2048로 커졌으므로 len(train_loader)가 줄어듭니다.
    # 스케줄러가 알아서 줄어든 step 수에 맞춰 Warmup을 완벽히 계산합니다.
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.1) 
    scheduler = get_step_schedule_with_warmup(optimizer, warmup_steps, total_steps, milestones=[0.6, 0.8], gamma=0.5)
    
    best_recall_100 = 0.0

    # -----------------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        if epoch == 2:
            print("\n🔥 [Dynamic Unfreeze] Epoch 2: Item Tower Joint Training 시작!")
            item_tower.set_freeze_state(False)
            item_finetune_lr = cfg.lr * 0.05 
            
            optimizer.add_param_group({
                'params': item_tower.parameters(), 
                'lr': item_finetune_lr 
            })
            print(f"   - User Tower LR: {cfg.lr}")
            print(f"   - Item Tower LR: {item_finetune_lr} (Fine-tuning mode)")

        # ------------------- 훈련 (Train) -------------------
        # 💡 [핵심 변동] 새로운 Multi-Positive 트레인 함수 호출
        avg_loss = train_user_tower_multi_positive(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, 
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=None,
            scheduler=scheduler,
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        
        # ------------------- 평가 (Evaluate) -------------------
        # 💡 사전에 수정한 "True Recall 계산 방식"의 평가 함수 호출
        val_metrics = evaluate_model(
            model=user_tower, 
            item_tower=item_tower, 
            dataloader=val_loader,
            target_df_path=TARGET_VAL_PATH,
            device=device,
            processor=processor,
            k_list=[20, 100, 500]
        )
        
        current_recall_100 = val_metrics.get('Recall@100', 0.0)
        
        # ------------------- Best Model 저장 -------------------
        if current_recall_100 > best_recall_100:
            print(f"🌟 [New Best!] True Recall@100 updated: {best_recall_100:.2f}% -> {current_recall_100:.2f}%")
            best_recall_100 = current_recall_100
            
            # 파일명에 멀티 포지티브 명시
            torch.save(user_tower.state_dict(), os.path.join(cfg.model_dir, "best_user_tower_v3_multipos_lossfix.pth"))
            torch.save(item_tower.state_dict(), os.path.join(cfg.model_dir, "best_item_tower_v3_multipos_lossfix.pth"))
            print("   💾 Best model weights saved to disk.")
        else:
            print(f"   - (Current Best True Recall: {best_recall_100:.2f}%)")
            
    print("\n🎉 Multi-Positive Pipeline Execution Finished Successfully!")
    
    
    
    


def run_pipeline_multipos_v3_resume():
    """Multi-Positive (Next-Session) 기반 훈련 파이프라인 엔트리 포인트 (Resume 기능 포함)"""
    print("🚀 Starting User Tower Training Pipeline [Multi-Positive V3]...")
    
    SEQ_LABELS = ['item_id', 'time', 'type', 'color', 'graphic', 'section', 'recency_curr', 'week_curr']
    STATIC_LABELS = [
        'age', 'price', 'cnt', 'recency',      
        'channel', 'club', 'news', 'fn', 'active', 
        'cont_price_std', 'cont_last_diff', 'cont_repurch', 'cont_weekend' 
    ] 
    
    # -----------------------------------------------------------
    # 1. Config & Env Setup
    # -----------------------------------------------------------
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    
    # Hyper-parameters
    cfg.lr = 4e-3
    cfg.batch_size = 1280 
    cfg.max_target_len = 10 
    
    # Item metadata cfg
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # -----------------------------------------------------------
    # 2. Data & DataLoader Setup
    # -----------------------------------------------------------
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    item_state_dict = load_item_tower_state_dict(cfg.model_dir, cfg.item_tower_pth_name, device)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    
    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22", aligned_vecs, is_train=False)
    
    dataset_peek(train_loader.dataset, processor)
    
    # -----------------------------------------------------------
    # 3. Models & Resume Logic Setup
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, item_state_dict, log_q_tensor)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    user_model_path = os.path.join(cfg.model_dir, "best_user_tower_v3_multipos.pth")
    item_model_path = os.path.join(cfg.model_dir, "best_item_tower_v3_multipos.pth")
    f_user_model_path = os.path.join(cfg.model_dir, "best_user_tower_v3_multipos_f1.pth")
    f_item_model_path = os.path.join(cfg.model_dir, "best_item_tower_v3_multipos_f1.pth")
    
    is_resuming = os.path.exists(user_model_path) and os.path.exists(item_model_path)
    item_finetune_lr = cfg.lr * 0.05 

    if is_resuming:
        print("🔄 [Resume] Previous checkpoints found! Loading weights...")
        user_tower.load_state_dict(torch.load(user_model_path, map_location=device))
        item_tower.load_state_dict(torch.load(item_model_path, map_location=device))
        
        # 💡 [핵심] Resume 시에는 이미 Item Tower가 학습에 참여한 상태이므로 바로 Unfreeze 하고 Optimizer에 등록
        item_tower.set_freeze_state(False)
        optimizer = torch.optim.AdamW([
            {'params': user_tower.parameters(), 'lr': cfg.lr},
            {'params': item_tower.parameters(), 'lr': item_finetune_lr}
        ], weight_decay=cfg.weight_decay)
        
        print(f"✅ Models loaded. Starting with Joint Training Mode right away.")
        print(f"   - User Tower LR: {cfg.lr}")
        print(f"   - Item Tower LR: {item_finetune_lr}")
    else:
        print("🆕 [New Run] No previous checkpoints found. Starting fresh.")
        item_tower.init_from_pretrained(aligned_vecs.to(device))
        item_tower.set_freeze_state(True)
        
        # 처음 시작 시에는 User Tower만 Optimizer에 등록
        optimizer = torch.optim.AdamW(user_tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        print(f"❄️ Epoch 1: Item Tower FROZEN! (User Tower LR: {cfg.lr})")

    # Scheduler 설정
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * 0.1) 
    scheduler = get_step_schedule_with_warmup(optimizer, warmup_steps, total_steps, milestones=[0.6, 0.8], gamma=0.5)
    
    best_recall_100 = 0.0

    # W&B Logging Init
    wandb.init(
        project="SASRec-User-Tower-MultiPositive-v3", 
        name=f"{'Resume_' if is_resuming else ''}run_bs{cfg.batch_size}_lr{cfg.lr}_ep{cfg.epochs}", 
        config=cfg.__dict__
    )

    # -----------------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        
        # 새롭게 학습을 시작하는 경우(Resume 아님)에만 Epoch 2에서 Item Tower Unfreeze
        if not is_resuming and epoch == 2:
            print("\n🔥 [Dynamic Unfreeze] Epoch 2: Item Tower Joint Training 시작!")
            item_tower.set_freeze_state(False)
            
            optimizer.add_param_group({
                'params': item_tower.parameters(), 
                'lr': item_finetune_lr 
            })
            print(f"   - User Tower LR: {cfg.lr}")
            print(f"   - Item Tower LR: {item_finetune_lr} (Fine-tuning mode)")

        # ------------------- 훈련 (Train) -------------------
        avg_loss = train_user_tower_multi_positive(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, 
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=None,
            scheduler=scheduler,
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        
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
        
        current_recall_100 = val_metrics.get('Recall@100', 0.0)
        
        # ------------------- Best Model 저장 -------------------
        if current_recall_100 > best_recall_100:
            print(f"🌟 [New Best!] True Recall@100 updated: {best_recall_100:.2f}% -> {current_recall_100:.2f}%")
            best_recall_100 = current_recall_100
            
            torch.save(user_tower.state_dict(), f_user_model_path)
            torch.save(item_tower.state_dict(), f_item_model_path)
            print("   💾 Best model weights saved to disk.")
        else:
            print(f"   - (Current Best True Recall: {best_recall_100:.2f}%)")
            
    print("\n🎉 Multi-Positive Pipeline Execution Finished Successfully!")
    

if __name__ == "__main__":
    run_pipeline_multipos_v3()