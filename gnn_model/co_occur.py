import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from v1_lightgcl import load_and_process_data
base_dir: str = r"D:\trainDataset\localprops"
model_dir: str = r"C:\Users\candyform\Desktop\inferenceCode\models"
user_path = os.path.join(base_dir, "features_user_w_meta.parquet") 
item_path = os.path.join(base_dir, "features_item.parquet")
seq_path = os.path.join(base_dir, "features_sequence_cleaned.parquet")


class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path,base_processor=None):
        print("🚀 Loading preprocessed features...")
        self.users = pd.read_parquet(user_path).drop_duplicates(subset=['customer_id']).set_index('customer_id')
        self.items = pd.read_parquet(item_path).drop_duplicates(subset=['article_id']).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')

        # 인덱스 타입 강제 (String)
        self.users.index = self.users.index.astype(str)
        self.items.index = self.items.index.astype(str)
        self.seqs.index = self.seqs.index.astype(str)

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
        
        # [A] User Features (유저 ID 1~N으로 바로 접근할 수 있도록 배열화)
        num_users_total = len(self.users) + 1
        self.i_side_arr = np.zeros((self.num_items + 1, 4), dtype=np.int64)
        for iid, row in self.items.iterrows():
            if iid not in self.item2id: continue
            idx = self.item2id[iid]
            # 전처리된 아이템 피처에 맞춰 컬럼명 수정 필요
            self.i_side_arr[idx] = [
                row.get('type_id', 0), row.get('color_id', 0), 
                row.get('graphic_id', 0), row.get('section_id', 0)
            ]
        
       
import os
import torch
import numpy as np
import scipy.sparse as sp
import time
from tqdm import tqdm

# 1. NPMI 계산 (Processor 인덱스 1~N 기준)
def calculate_npmi_sparse(edge_index, num_users, num_items_total):
    print(f"\n[NPMI] Calculating NPMI (Size: {num_items_total}x{num_items_total})...")
    start = time.time()
    
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    
    # 각 아이템별 구매 유저 수 (1-based 인덱스 대응)
    item_degrees = np.bincount(dst, minlength=num_items_total)
    
    data = np.ones(len(src), dtype=np.float32)
    R = sp.csr_matrix((data, (src, dst)), shape=(num_users, num_items_total))
    R.data = np.ones_like(R.data) 
    
    # 아이템-아이템 동시출현 행렬 C = R^T * R
    C = R.T.dot(R)
    C.setdiag(0)
    C.eliminate_zeros()
    
    C_coo = C.tocoo()
    row, col, co_occur_counts = C_coo.row, C_coo.col, C_coo.data
    
    # 확률 기반 NPMI 계산
    p_i = item_degrees[row] / num_users
    p_j = item_degrees[col] / num_users
    p_i_j = co_occur_counts / num_users
    
    eps = 1e-12
    pmi = np.log((p_i_j + eps) / (p_i * p_j + eps))
    npmi = pmi / (-np.log(p_i_j + eps))
    
    NPMI_matrix = sp.coo_matrix((npmi, (row, col)), shape=(num_items_total, num_items_total)).tocsr()
    print(f" -> NPMI Done! ({time.time() - start:.2f}s)")
    return C, NPMI_matrix

# 2. Hard Negative Mining (VRAM 최적화 및 0번 패딩 보호)
def mine_hard_negatives_optimized(V_embeddings, NPMI_matrix, num_items_total, top_k=50, chunk_size=512):
    print(f"\n[Mining] Starting Direct Mining (Chunk Size: {chunk_size})...")
    start = time.time()
    device = V_embeddings.device
    
    # 코사인 유사도를 위한 L2 정규화
    norm_embed = torch.nn.functional.normalize(V_embeddings, p=2, dim=1)
    hard_neg_pool = np.zeros((num_items_total, top_k), dtype=np.int64)
    
    if not sp.isspmatrix_csr(NPMI_matrix):
        NPMI_matrix = NPMI_matrix.tocsr()

    for start_idx in tqdm(range(0, num_items_total, chunk_size), desc="Mining"):
        end_idx = min(start_idx + chunk_size, num_items_total)
        current_chunk_size = end_idx - start_idx
        
        # GPU 행렬 곱 연산
        chunk_sim = torch.matmul(norm_embed[start_idx:end_idx], norm_embed.T)
        
        # NPMI > 0 인 (Positive) 아이템 마스킹
        chunk_npmi = NPMI_matrix[start_idx:end_idx].tocoo()
        if chunk_npmi.data.any():
            pos_mask = chunk_npmi.data > 1
            rows_t = torch.tensor(chunk_npmi.row[pos_mask], dtype=torch.long, device=device)
            cols_t = torch.tensor(chunk_npmi.col[pos_mask], dtype=torch.long, device=device)
            chunk_sim[rows_t, cols_t] = -2.0
            
        # 자기 자신 및 0번 패딩 마스킹
        row_idx = torch.arange(current_chunk_size, device=device)
        col_idx = torch.arange(start_idx, end_idx, device=device)
        chunk_sim[row_idx, col_idx] = -2.0
        chunk_sim[:, 0] = -2.0 # 패딩 인덱스 원천 차단
        
        # Top-K 추출
        _, top_k_idx = torch.topk(chunk_sim, k=top_k, dim=1)
        hard_neg_pool[start_idx:end_idx] = top_k_idx.cpu().numpy()
        
    print(f" -> Mining Done! ({time.time() - start:.2f}s)")
    return hard_neg_pool

# 3. 통합 실행 파이프라인 (Direct Alignment 버전)
def run_processor_direct_mining(processor):
    NEW_CONFIG = {
        'model_pth': r'C:\Users\candyform\Desktop\inferenceCode\models\best_item_tower_c.pth',
        'raw_graph_path': r'D:\trainDataset\localprops\cache\raw_graph.json',
        'cache_dir': r'D:\trainDataset\localprops\cache',
        'top_k': 50
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool_save_path = os.path.join(NEW_CONFIG['cache_dir'], 'hard_neg_pool_direct.npy')
    
    # [Step 1] 모델 가중치 로드 및 GPU 전송 (이미 검증됨)
    print(f"🔄 Loading tuned weights from {os.path.basename(NEW_CONFIG['model_pth'])}...")
    state_dict = torch.load(NEW_CONFIG['model_pth'], map_location='cpu')
    item_weights = state_dict['item_matrix.weight'].to(device) # Shape: [47063, 128]
    num_items_total = item_weights.shape[0]

    # [Step 2] 그래프 데이터 로드 및 인덱스 치환
    # raw_graph.json의 아이템 ID를 processor.item2id(1~N)로 바꿉니다.
    print(f"🔄 Re-mapping raw graph to Processor indices...")
    edge_index, num_users, _, _, raw_item2id = load_and_process_data(
        NEW_CONFIG['raw_graph_path'], NEW_CONFIG['cache_dir']
    )
    
    # raw_idx -> string_id -> processor_idx 매핑 테이블 생성
    raw_idx_to_str = {v: str(k) for k, v in raw_item2id.items()}
    
    src_raw, dst_raw = edge_index[0].numpy(), edge_index[1].numpy()
    aligned_src, aligned_dst = [], []
    
    for u, i in zip(src_raw, dst_raw):
        item_str = raw_idx_to_str[i]
        if item_str in processor.item2id:
            aligned_src.append(u)
            aligned_dst.append(processor.item2id[item_str])
            
    aligned_edge_index = torch.tensor([aligned_src, aligned_dst], dtype=torch.long)
    print(f"✅ Graph aligned. Edges: {len(aligned_src)}")

    # [Step 3] NPMI 및 마이닝 실행
    NPMI_matrix = calculate_npmi_sparse(aligned_edge_index, num_users, num_items_total)
    
    hard_neg_pool_np = mine_hard_negatives_optimized(
        V_embeddings=item_weights,
        NPMI_matrix=NPMI_matrix,
        num_items_total=num_items_total,
        top_k=NEW_CONFIG['top_k']
    )
    
    # [Step 4] 결과 저장
    np.save(pool_save_path, hard_neg_pool_np)
    print(f"🚀 [Success] Hard Negative Pool saved to {pool_save_path}")
    return hard_neg_pool_np


def mine_hard_negatives_category_aware(V_embeddings, NPMI_matrix, item_side_arr, num_items_total, top_k=50, chunk_size=512):
    """
    item_side_arr: FeatureProcessor.i_side_arr (N+1, 4) -> 3번 인덱스가 section_id라고 가정
    """
    print(f"\n[Mining] Starting Category-Aware Mining...")
    start = time.time()
    device = V_embeddings.device
    
    # 1. 카테고리 정보(section_id) 추출 및 텐서화
    # i_side_arr의 4번째 컬럼(index 3)이 section_id
    item_categories = torch.tensor(item_side_arr[:, 0], dtype=torch.long, device=device)
    
    norm_embed = torch.nn.functional.normalize(V_embeddings, p=2, dim=1)
    hard_neg_pool = np.zeros((num_items_total, top_k), dtype=np.int64)
    
    if not sp.isspmatrix_csr(NPMI_matrix):
        NPMI_matrix = NPMI_matrix.tocsr()

    for start_idx in tqdm(range(0, num_items_total, chunk_size), desc="Mining"):
        end_idx = min(start_idx + chunk_size, num_items_total)
        current_chunk_size = end_idx - start_idx
        
        # (A) 유사도 계산
        chunk_sim = torch.matmul(norm_embed[start_idx:end_idx], norm_embed.T)
        
        # (B) 카테고리 마스크 생성 [chunk_size, num_items_total]
        # 현재 청크의 아이템 카테고리와 전체 아이템 카테고리를 비교
        chunk_cats = item_categories[start_idx:end_idx].unsqueeze(1) # [512, 1]
        all_cats = item_categories.unsqueeze(0) # [1, 47063]
        same_cat_mask = (chunk_cats == all_cats) # [512, 47063] 부울 행렬
        
        # (C) 기본 마스킹 (NPMI > 1, 자기자신, 패딩)
        chunk_npmi = NPMI_matrix[start_idx:end_idx].tocoo()
        if chunk_npmi.data.any():
            pos_mask = chunk_npmi.data > 0 # 앞서 제안한 '동시출현 1회 노이즈 허용' 기준
            rows_t = torch.tensor(chunk_npmi.row[pos_mask], dtype=torch.long, device=device)
            cols_t = torch.tensor(chunk_npmi.col[pos_mask], dtype=torch.long, device=device)
            chunk_sim[rows_t, cols_t] = -2.0
            
        row_idx = torch.arange(current_chunk_size, device=device)
        col_idx = torch.arange(start_idx, end_idx, device=device)
        chunk_sim[row_idx, col_idx] = -2.0
        chunk_sim[:, 0] = -2.0
        
        # (D) 🔥 카테고리 부스팅 (핵심 로직)
        # 마스킹되지 않은(-2.0이 아닌) 아이템 중 카테고리가 같은 놈들에게 점수 +2.0 부여
        # 이 처리를 통해 카테고리가 같은 아이템들이 무조건 상위 Top-K를 점령하게 됨
        valid_mask = (chunk_sim > -1.0)
        boost_mask = same_cat_mask & valid_mask
        chunk_sim[boost_mask] += 2.0 
        
        # (E) Top-K 추출
        _, top_k_idx = torch.topk(chunk_sim, k=top_k, dim=1)
        hard_neg_pool[start_idx:end_idx] = top_k_idx.cpu().numpy()
        
    print(f" -> Category-Aware Mining Done! ({time.time() - start:.2f}s)")
    return hard_neg_pool


def analyze_category_aware_pool(pool_path, model_path, processor, top_n_items=5, top_k=10):
    print(f"\n[Analysis] Category-Aware Mining vs Simple KNN")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 데이터 로드
    hard_neg_pool = np.load(pool_path)
    state_dict = torch.load(model_path, map_location=device)
    weights = state_dict['item_matrix.weight']
    norm_weights = F.normalize(weights, p=2, dim=1)
    
    # 2. 카테고리 정보 준비 (Type ID: index 0)
    item_side_info = processor.i_side_arr
    type_ids = item_side_info[:, 0] 

    for target_idx in range(1, top_n_items + 1):
        target_vec = norm_weights[target_idx].unsqueeze(0)
        target_type = type_ids[target_idx] 
        
        # (A) 단순 KNN (필터링 없음)
        all_sims = torch.matmul(target_vec, norm_weights.T).squeeze(0)
        all_sims[target_idx] = -2.0 
        all_sims[0] = -2.0
        simple_vals, simple_indices = torch.topk(all_sims, k=top_k)
        
        # (B) 하드 네거티브 (마이닝된 풀)
        hard_indices = hard_neg_pool[target_idx][:top_k]
        hard_vals = torch.matmul(target_vec, norm_weights[hard_indices].T).squeeze(0)

        print(f"\nTarget Index: {target_idx} | Type ID: {target_type}")
        print(f"{'Rank':<4} | {'Simple KNN Top-K':<26} | {'Category-Aware Hard Neg':<26}")
        print(f"{'-'*4}:|:{'-'*26}:|:{'-'*26}")
        
        for r in range(top_k):
            s_idx = simple_indices[r].item()
            s_sim = simple_vals[r].item()
            s_type = type_ids[s_idx]
            s_match = "★" if s_type == target_type else "  " 
            
            h_idx = hard_indices[r]
            h_sim = hard_vals[r].item()
            h_type = type_ids[h_idx]
            h_match = "★" if h_type == target_type else "  "
            
            s_str = f"ID:{s_idx:<5} ({s_sim:.3f}) Type:{s_type}{s_match}"
            h_str = f"ID:{h_idx:<5} ({h_sim:.3f}) Type:{h_type}{h_match}"
            
            print(f"{r+1:<4} | {s_str:<26} | {h_str:<26}")
        print("-" * 65)
def aware_pipeline():
    train_proc = FeatureProcessor(user_path, item_path, seq_path)
    #run_processor_direct_mining(train_proc)
    #verify_processor_vs_model(train_proc, r'C:\Users\candyform\Desktop\inferenceCode\models\best_item_tower_c.pth')
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_side_info = train_proc.i_side_arr 
    
    NEW_CONFIG = {
        'model_pth': r'C:\Users\candyform\Desktop\inferenceCode\models\best_item_tower_c.pth',
        'raw_graph_path': r'D:\trainDataset\localprops\cache\raw_graph.json',
        'cache_dir': r'D:\trainDataset\localprops\cache',
        'top_k': 50
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool_save_path = os.path.join(NEW_CONFIG['cache_dir'], 'hard_neg_pool_direct.npy')
    
    # [Step 1] 모델 가중치 로드 및 GPU 전송 (이미 검증됨)
    print(f"🔄 Loading tuned weights from {os.path.basename(NEW_CONFIG['model_pth'])}...")
    state_dict = torch.load(NEW_CONFIG['model_pth'], map_location='cpu')
    item_weights = state_dict['item_matrix.weight'].to(device) # Shape: [47063, 128]
    num_items_total = item_weights.shape[0]

    # [Step 2] 그래프 데이터 로드 및 인덱스 치환
    # raw_graph.json의 아이템 ID를 processor.item2id(1~N)로 바꿉니다.
    print(f"🔄 Re-mapping raw graph to Processor indices...")
    edge_index, num_users, _, _, raw_item2id = load_and_process_data(
        NEW_CONFIG['raw_graph_path'], NEW_CONFIG['cache_dir']
    )
    
    # raw_idx -> string_id -> processor_idx 매핑 테이블 생성
    raw_idx_to_str = {v: str(k) for k, v in raw_item2id.items()}
    
    src_raw, dst_raw = edge_index[0].numpy(), edge_index[1].numpy()
    aligned_src, aligned_dst = [], []
    
    for u, i in zip(src_raw, dst_raw):
        item_str = raw_idx_to_str[i]
        if item_str in train_proc.item2id:
            aligned_src.append(u)
            aligned_dst.append(train_proc.item2id[item_str])
            
    aligned_edge_index = torch.tensor([aligned_src, aligned_dst], dtype=torch.long)
    print(f"✅ Graph aligned. Edges: {len(aligned_src)}")

    # [Step 3] NPMI 및 마이닝 실행
    NPMI_matrix = calculate_npmi_sparse(aligned_edge_index, num_users, num_items_total)
    
    # 3. 함수 실행
    hard_neg_pool_cat_aware = mine_hard_negatives_category_aware(
        V_embeddings=item_weights,   # 모델의 최신 아이템 임베딩
        NPMI_matrix=NPMI_matrix,               # 행동 기반 관계망
        item_side_arr=item_side_info,          # 카테고리(Section) 정보
        num_items_total=train_proc.num_items + 1, # 47063 (Padding 포함)
        top_k=50,                              # 타겟당 하드 네거티브 50개 추출
        chunk_size=512     
        
        # VRAM 8GB 최적화 크기
    )
    
    output_path = r'D:\trainDataset\localprops\cache\hard_neg_pool_category_aware.npy'
    np.save(output_path, hard_neg_pool_cat_aware)
    print(f"🚀 Category-Aware Hard Negatives saved to {output_path}")
def analyze_hn_with_popularity(processor, pool_path, model_path, raw_graph_path, cache_dir):
    print(f"\n[Analysis] Checking Sales Volume of Hard Negatives...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 판매량(Popularity) 집계
    # NPMI 계산 시 사용했던 것과 동일한 방식으로 엣지를 로드합니다.
    from co_occur import load_and_process_data # 기존에 정의한 함수 임포트
    edge_index, _, _, _, raw_item2id = load_and_process_data(raw_graph_path, cache_dir)
    
    raw_idx_to_str = {v: str(k) for k, v in raw_item2id.items()}
    dst_raw = edge_index[1].numpy()
    
    # processor 인덱스(1~N) 기준 판매량 배열 생성
    num_items_total = processor.num_items + 1
    item_sales_counts = np.zeros(num_items_total, dtype=np.int64)
    
    for i in dst_raw:
        item_str = raw_idx_to_str.get(i)
        if item_str in processor.item2id:
            p_idx = processor.item2id[item_str]
            item_sales_counts[p_idx] += 1
            
    print(f" -> Sales counting done. (Max sales: {item_sales_counts.max()})")

    # 2. 하드 네거티브 및 모델 가중치 로드
    hard_neg_pool = np.load(pool_path)
    state_dict = torch.load(model_path, map_location=device)
    weights = state_dict['item_matrix.weight']
    norm_weights = F.normalize(weights, p=2, dim=1)
    
    # i_side_arr의 0번 컬럼 (type_id)
    type_ids = processor.i_side_arr[:, 0]

    # 3. 상위 5개 타겟 분석 출력
    for target_idx in range(1, 6):
        target_sales = item_sales_counts[target_idx]
        target_type = type_ids[target_idx]
        target_vec = norm_weights[target_idx]
        
        hard_indices = hard_neg_pool[target_idx][:10]
        
        print(f"\nTarget ID: {target_idx} | Type: {target_type} | Target Sales: {target_sales}")
        print(f"{'Rank':<4} | {'Neg Index':<10} | {'Type':<6} | {'Sales':<10} | {'Sim'}")
        print(f"{'-'*4}:|:{'-'*10}:|:{'-'*6}:|:{'-'*10}:|:{'-'*5}")
        
        for r in range(10):
            h_idx = hard_indices[r]
            h_sales = item_sales_counts[h_idx]
            h_type = type_ids[h_idx]
            h_sim = torch.dot(target_vec, norm_weights[h_idx]).item()
            
            print(f"{r+1:<4} | {h_idx:<10} | {h_type:<6} | {h_sales:<10} | {h_sim:.3f}")
        print("-" * 65)

import time
import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

def mine_hard_negatives_ultimate(V_embeddings, NPMI_matrix, item_side_arr, item_sales_counts, num_items_total, min_sales=10, top_k=50, chunk_size=512):
    """
    - V_embeddings: 모델의 아이템 임베딩 텐서 [num_items, dim]
    - NPMI_matrix: 앞서 계산한 NPMI 희소 행렬 [num_items, num_items]
    - item_side_arr: Type ID 등 사이드 정보 (index 0이 Type ID)
    - item_sales_counts: 각 아이템의 총 판매 횟수 배열 [num_items]
    """
    print(f"\n[Mining] Starting Ultimate Hard Negative Mining...")
    print(f" -> Config: Route 2 (NPMI > 0 Masking) | Min Sales: {min_sales} | Top-K: {top_k}")
    
    start = time.time()
    device = V_embeddings.device
    
    # 1. 텐서화 및 정규화
    norm_embed = torch.nn.functional.normalize(V_embeddings, p=2, dim=1)
    hard_neg_pool = np.zeros((num_items_total, top_k), dtype=np.int64)
    
    # Type ID (0번 인덱스)
    item_categories = torch.tensor(item_side_arr[:, 0], dtype=torch.long, device=device)
    # 판매량 정보
    sales_counts = torch.tensor(item_sales_counts, dtype=torch.long, device=device)
    
    # [필터 1] 판매량 미달 마스크 (이 아이템들은 오답으로 쓰지 않음)
    unpopular_mask = (sales_counts < min_sales) # Boolean Tensor
    
    if not sp.isspmatrix_csr(NPMI_matrix):
        NPMI_matrix = NPMI_matrix.tocsr()

    # VRAM 이슈 방지를 위한 Chunk 진행
    for start_idx in tqdm(range(0, num_items_total, chunk_size), desc="Mining Chunks"):
        end_idx = min(start_idx + chunk_size, num_items_total)
        current_chunk_size = end_idx - start_idx
        
        # (A) 전체 대상 코사인 유사도 연산 [chunk_size, num_items_total]
        # 타겟 512개 각각에 대해 전체 47k 아이템과의 유사도를 한 번에 구함
        chunk_sim = torch.matmul(norm_embed[start_idx:end_idx], norm_embed.T)
        
        # (B) 마스킹 1: 판매량 미달 유령 상품 영구 제거 (-10000점 부여)
        chunk_sim[:, unpopular_mask] = -10000.0
        
        # (C) 마스킹 2: 루트 2 (NPMI > 0) 진짜 정답 제거
        chunk_npmi_sparse = NPMI_matrix[start_idx:end_idx].tocoo()
        if len(chunk_npmi_sparse.data) > 0:
            npmi_threshold = 0.0 
            
            # NPMI가 임계값보다 큰 경우만 '진짜 의미 있는 동시 출현'으로 간주하여 마스킹
            # 0.0 < NPMI <= 0.01 구간의 아이템은 아주 미세한 우연적 겹침으로 보고 오답 후보로 허용!
            pos_mask = chunk_npmi_sparse.data > npmi_threshold 
            
            rows_t = torch.tensor(chunk_npmi_sparse.row[pos_mask], dtype=torch.long, device=device)
            cols_t = torch.tensor(chunk_npmi_sparse.col[pos_mask], dtype=torch.long, device=device)
            chunk_sim[rows_t, cols_t] = -10000.0
        # (D) 마스킹 3: 자기 자신 및 패딩(0번) 제거
        row_idx = torch.arange(current_chunk_size, device=device)
        col_idx = torch.arange(start_idx, end_idx, device=device)
        chunk_sim[row_idx, col_idx] = -10000.0 # 자기 자신
        chunk_sim[:, 0] = -10000.0           # 0번 패딩
        
        # (E) Type ID 부스팅 (살아남은 후보들 대상)
        # 현재 타겟의 Type과 전체 아이템의 Type 비교
        chunk_cats = item_categories[start_idx:end_idx].unsqueeze(1) 
        all_cats = item_categories.unsqueeze(0) 
        same_cat_mask = (chunk_cats == all_cats) 
        
        # -10000점으로 날아가지 않은(유효한) 아이템 중 타입이 같으면 점수 폭등(+2.0)
        valid_mask = (chunk_sim > -5000.0)
        boost_mask = same_cat_mask & valid_mask
        chunk_sim[boost_mask] += 0.03
        
        # 💡 수정 2: Top-K가 너무 고정되는 것을 막기 위해 미세한 Gumbel Noise 추가 (0.01 수준)
        # 이렇게 하면 같은 카테고리 내에서도 다양한 상품이 Hard Negative로 올라옵니다.
        noise = torch.empty_like(chunk_sim).uniform_(0, 1)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-20)) * 0.01
        chunk_sim += gumbel_noise
        # (F) 안전하게 최종 Top-K 추출
        # 이미 부적격자들은 -10000점 밑바닥에 있으므로,
        # 조건을 만족하는 아이템 중에서만 상위 50개가 깔끔하게 뽑힘.
        _, top_k_idx = torch.topk(chunk_sim, k=top_k, dim=1)
        hard_neg_pool[start_idx:end_idx] = top_k_idx.cpu().numpy()
        
    print(f" -> Mining Complete! Valid Hard Negatives extracted. ({time.time() - start:.2f}s)")
    return hard_neg_pool
def ttt():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_side_info = train_proc.i_side_arr 
    
    NEW_CONFIG = {
        'model_pth': r'C:\Users\candyform\Desktop\inferenceCode\models\best_item_tower_c.pth',
        'raw_graph_path': r'D:\trainDataset\localprops\cache\raw_graph.json',
        'cache_dir': r'D:\trainDataset\localprops\cache',
        'top_k': 50
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool_save_path = os.path.join(NEW_CONFIG['cache_dir'], 'hard_neg_pool_direct.npy')
    
    # [Step 1] 모델 가중치 로드 및 GPU 전송 (이미 검증됨)
    print(f"🔄 Loading tuned weights from {os.path.basename(NEW_CONFIG['model_pth'])}...")
    state_dict = torch.load(NEW_CONFIG['model_pth'], map_location='cpu')
    item_weights = state_dict['item_matrix.weight'].to(device) # Shape: [47063, 128]
    num_items_total = item_weights.shape[0]

    # [Step 2] 그래프 데이터 로드 및 인덱스 치환
    # raw_graph.json의 아이템 ID를 processor.item2id(1~N)로 바꿉니다.
    print(f"🔄 Re-mapping raw graph to Processor indices...")
    edge_index, num_users, _, _, raw_item2id = load_and_process_data(
        NEW_CONFIG['raw_graph_path'], NEW_CONFIG['cache_dir']
    )
    
    # raw_idx -> string_id -> processor_idx 매핑 테이블 생성
    raw_idx_to_str = {v: str(k) for k, v in raw_item2id.items()}
    
    src_raw, dst_raw = edge_index[0].numpy(), edge_index[1].numpy()
    aligned_src, aligned_dst = [], []
    
    for u, i in zip(src_raw, dst_raw):
        item_str = raw_idx_to_str[i]
        if item_str in train_proc.item2id:
            aligned_src.append(u)
            aligned_dst.append(train_proc.item2id[item_str])
            
    aligned_edge_index = torch.tensor([aligned_src, aligned_dst], dtype=torch.long)
    print(f"✅ Graph aligned. Edges: {len(aligned_src)}")

    # [Step 3] NPMI 및 마이닝 실행
    NPMI_matrix = calculate_npmi_sparse(aligned_edge_index, num_users, num_items_total)
    
    cache_dir = r'D:\trainDataset\localprops\cache'
    ultimate_pool_save_path = os.path.join(cache_dir, 'hard_neg_pool_ultimate.npy')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_sales_counts = np.bincount(
        aligned_edge_index[1].numpy(), 
        minlength=train_proc.num_items + 1
    )
    # 2. Ultimate 하드 네거티브 마이닝 실행
    # (weights, NPMI_matrix, processor, item_sales_counts가 이미 로드되어 있다고 가정)
    hard_neg_pool_ultimate = mine_hard_negatives_ultimate(
        V_embeddings=item_weights,           # 모델 아이템 가중치
        NPMI_matrix=NPMI_matrix,                   # NPMI 희소 행렬
        item_side_arr=train_proc.i_side_arr,        # 아이템 사이드 정보 (Type ID 포함)
        item_sales_counts=item_sales_counts,       # 아이템별 판매량 배열
        num_items_total=train_proc.num_items + 1,   # 전체 아이템 수 (0번 패딩 포함)
        min_sales=10,                              # [핵심] 판매량 10개 미만 유령 상품 필터링
        top_k=50,                                  # 타겟당 50개 추출
        chunk_size=512                             # 8GB VRAM 최적화 청크 사이즈
    )
    np.save(ultimate_pool_save_path, hard_neg_pool_ultimate)
    print(f"🚀 [Success] Ultimate Hard Negative Pool saved to:\n{ultimate_pool_save_path}")
def inspect_hard_neg_pool_quality(npy_path):
    print(f"\n🔍 [Hard Negative Pool Inspection]")
    pool = np.load(npy_path)
    num_items, top_k = pool.shape
    
    # 0번 인덱스(패딩)가 아닌 유효한 네거티브 개수 카운트
    valid_counts = (pool != 0).sum(axis=1)
    
    full_count = (valid_counts == top_k).sum()
    zero_count = (valid_counts == 0).sum()
    avg_valid = valid_counts.mean()
    
    print(f" - Total Items in Pool: {num_items:,}")
    print(f" - Target Top-K: {top_k}")
    print(f" - Average Valid HNs per item: {avg_valid:.2f} / {top_k}")
    print(f" - Items with full {top_k} HNs: {full_count:,} ({full_count/num_items*100:.1f}%)")
    print(f" - Items with ZERO HNs: {zero_count:,} ({zero_count/num_items*100:.1f}%) -> (Will fallback to In-batch natively)")
    
    # 히스토그램 출력 (간단하게 10구간)
    bins = [0, 10, 20, 30, 40, 50]
    hist, _ = np.histogram(valid_counts, bins=bins)
    print("\n📊 Valid HN Distribution:")
    for i in range(len(hist)):
        print(f"   [{bins[i]:02d}~{bins[i+1]:02d}]: {hist[i]:,} items")
    print("-" * 40)
    
    
import seaborn as sns
    
def visualize_item_stats(C_matrix, NPMI_matrix, n_items_to_show=50):
    """
    C_matrix: 아이템-아이템 동시출현 행렬 (R.T.dot(R))
    NPMI_matrix: 계산된 NPMI 행렬
    n_items_to_show: 시각화할 상위 아이템 개수 (0~n)
    """
    
    # 1. 공동구매 빈도 히트맵 (Co-occurrence Count)
    # R^T * R 결과인 C 행렬의 일부를 시각화합니다.
    plt.figure(figsize=(12, 10))
    # sparse matrix일 경우 .toarray()로 변환
    subset_c = C_matrix[:n_items_to_show, :n_items_to_show].toarray()
    
    sns.heatmap(subset_c, cmap="YlGnBu", annot=False)
    plt.title(f"Item Co-occurrence Heatmap (Top {n_items_to_show} items)")
    plt.xlabel("Item ID")
    plt.ylabel("Item ID")
    plt.savefig("item_cooccurrence_heatmap.png")
    
    # 2. NPMI 점수 히트맵 (Normalized PMI)
    # -1(전혀 같이 안 나타남) ~ 1(항상 같이 나타남) 사이의 값을 가집니다.
    plt.figure(figsize=(12, 10))
    subset_npmi = NPMI_matrix[:n_items_to_show, :n_items_to_show].toarray()
    
    # NPMI는 0을 기준으로 양수(연관), 음수(비연관)을 보기 위해 RdYlBu_r 맵 사용
    sns.heatmap(subset_npmi, cmap="RdYlBu_r", center=0, annot=False)
    plt.title(f"Item NPMI Heatmap (Top {n_items_to_show} items)")
    plt.xlabel("Item ID")
    plt.ylabel("Item ID")
    plt.savefig("item_npmi_heatmap.png")
    
    # 3. 공동구매 빈도 분포 (Distribution)
    # 얼마나 많은 아이템 쌍들이 강하게 결합되어 있는지 확인합니다.
    plt.figure(figsize=(10, 6))
    counts = C_matrix.data # 0이 아닌 값들만 추출
    plt.hist(counts, bins=50, color='skyblue', edgecolor='black', log=True)
    plt.title("Distribution of Co-occurrence Counts (Log Scale)")
    plt.xlabel("Number of Shared Users")
    plt.ylabel("Frequency (Log)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("cooccurrence_distribution.png")
if __name__ == "__main__":
    train_proc = FeatureProcessor(user_path, item_path, seq_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_side_info = train_proc.i_side_arr 
    
    NEW_CONFIG = {
        'model_pth': r'C:\Users\candyform\Desktop\inferenceCode\models\best_item_tower_c.pth',
        'raw_graph_path': r'D:\trainDataset\localprops\cache\raw_graph.json',
        'cache_dir': r'D:\trainDataset\localprops\cache',
        'top_k': 50
    }
    

    # [Step 2] 그래프 데이터 로드 및 인덱스 치환
    # raw_graph.json의 아이템 ID를 processor.item2id(1~N)로 바꿉니다.
    print(f"🔄 Re-mapping raw graph to Processor indices...")
    edge_index, num_users, num_items_total, _, raw_item2id = load_and_process_data(
        NEW_CONFIG['raw_graph_path'], NEW_CONFIG['cache_dir']
    )
    

    C, NPMI_matrix = calculate_npmi_sparse(edge_index, num_users, num_items_total)
    visualize_item_stats(C, NPMI_matrix, n_items_to_show=50)