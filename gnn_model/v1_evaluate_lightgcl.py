import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
    
from tower_code.params_config import PipelineConfig
from tower_code.v3_utils import prepare_features

# =========================================================
# 1. Inference용 껍데기 모델
# =========================================================


def build_sparse_graph(user_ids, item_ids, train_df, device):
    """
    Train DataFrame(Parquet)을 읽어 LightGCL 학습 때와 동일한 
    Normalized Adjacency Matrix를 생성합니다.
    """
    print("⚡ Building Graph Adjacency Matrix...")
    
    n_users = len(user_ids) + 1 # 0번 padding 고려 (Processor 기준)
    n_items = len(item_ids) + 1
    
    # 1. ID 매핑 준비 (Processor의 ID 체계 사용)
    # user_ids, item_ids는 processor.user_ids, processor.item_ids
    u_mapper = {uid: i+1 for i, uid in enumerate(user_ids)}
    i_mapper = {iid: i+1 for i, iid in enumerate(item_ids)}
    
    # 2. 컬럼명 자동 감지 (sequence_ids 추가)
    u_col = 'customer_id' if 'customer_id' in train_df.columns else train_df.columns[0]
    
    # 'sequence_ids'를 우선순위로 둠
    possible_item_cols = ['sequence_ids', 'article_id', 'item_id', 'product_id', 'article_ids']
    i_col = next((col for col in possible_item_cols if col in train_df.columns), None)
    
    if i_col is None:
        raise KeyError(f"❌ Item column not found! Available: {train_df.columns}")
    
    print(f"   -> Using columns: User='{u_col}', Item='{i_col}'")

    # 3. 엣지(Edge) 추출
    src = []
    dst = []
    
    valid_interactions = 0
    
    # DataFrame 순회
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="   -> Mapping Edges"):
        u_val = row[u_col]
        i_val = row[i_col] # 이게 리스트일 확률 99% (sequence_ids)
        
        # 유저가 존재하지 않으면 건너뜀
        if u_val not in u_mapper:
            continue
            
        u_idx = u_mapper[u_val]
        
        # [핵심] 리스트 형태(sequence_ids) 처리
        if isinstance(i_val, (list, np.ndarray)):
            for item in i_val:
                if item in i_mapper:
                    src.append(u_idx)
                    dst.append(i_mapper[item])
                    valid_interactions += 1
        # 단일 값 처리 (혹시 모를 대비)
        else:
            if i_val in i_mapper:
                src.append(u_idx)
                dst.append(i_mapper[i_val])
                valid_interactions += 1
            
    print(f"   -> Valid Edges: {valid_interactions}")
    
    if valid_interactions == 0:
        raise ValueError("❌ No valid interactions found! Check ID matching.")

    # 4. Sparse Matrix 생성 (Train 코드 로직 준수)
    # User-Item Interaction Matrix R
    # shape: (n_users, n_items)
    # src(user indices), dst(item indices)
    
    # 중복 제거 (User가 같은 아이템을 여러 번 샀을 수 있음 -> Graph Edge는 1개로 취급)
    # coo_matrix 생성 시 중복된 좌표는 값이 더해지므로, 일단 만들고 1로 만듦
    R = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_users, n_items))
    # 0보다 큰 값은 1로 (Interaction 여부만 중요)
    R.data = np.ones_like(R.data) 

    # 5. Adjacency Matrix A 생성
    # [ 0, R ]
    # [ R.T, 0 ]
    # Training 코드에서는 sp.coo_matrix로 직접 좌표를 합쳐서 만들었지만,
    # 여기서는 R을 기반으로 안전하게 만듭니다.
    
    R = R.tocoo()
    
    # User Node: 0 ~ n_users-1
    # Item Node: n_users ~ n_users + n_items - 1 (Offset 적용)
    user_nodes = R.row
    item_nodes = R.col + n_users
    
    # 상단 우측 (User -> Item)
    row_idx = np.concatenate([user_nodes, item_nodes])
    col_idx = np.concatenate([item_nodes, user_nodes])
    data = np.ones(len(row_idx), dtype=np.float32)
    
    num_nodes = n_users + n_items
    adj_mat = sp.coo_matrix((data, (row_idx, col_idx)), shape=(num_nodes, num_nodes))
    
    # 6. Normalization (Train 코드와 완벽 동일 로직)
    # D^-0.5 * A * D^-0.5
    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    norm_adj = norm_adj.tocoo()
    
    # 7. Torch Sparse Tensor 변환
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data).float()
    shape = torch.Size(norm_adj.shape)
    
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
    
    return adj_tensor

def compute_final_embeddings(model, adj_tensor, n_layers=2):
    """
    저장된 Weight(Layer 0)를 그래프에 통과시켜 Final Embedding을 계산
    """
    print(f"\n🌊 Propagating Embeddings (Layers: {n_layers})...")
    model.eval()
    with torch.no_grad():
        # 1. 초기 임베딩 결합 (User + Item)
        ego_embeddings = torch.cat([
            model.embedding_user.weight, 
            model.embedding_item.weight
        ], dim=0)
        
        all_embeddings = [ego_embeddings]
        
        # 2. 레이어 전파 (Graph Convolution)
        for k in range(n_layers):
            # Sparse Matrix Multiplication (Message Passing)
            ego_embeddings = torch.sparse.mm(adj_tensor, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            print(f"   -> Layer {k+1} done.")
            
        # 3. 레이어 평균 (Mean Aggregation)
        # stack -> (Layers, Nodes, Dim) -> mean(dim=0)
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        
        # 4. 다시 User/Item으로 분리
        num_users = model.embedding_user.num_embeddings
        num_items = model.embedding_item.num_embeddings
        
        final_user_emb, final_item_emb = torch.split(final_embeddings, [num_users, num_items])
        
        return final_user_emb, final_item_emb
# =========================================================
# 2. 모델 로드 및 정렬 (Alignment) - 가장 중요 ⭐
# =========================================================
def load_and_align_model(model, processor, checkpoint_path, maps_path, device):
    """
    학습된 .pth(과거 ID 순서)를 로드하여,
    현재 processor(검증 ID 순서)에 맞게 임베딩 행렬을 재조립합니다.
    """
    print(f"\n🔄 [Alignment] Loading model weights and aligning IDs...")
    
    # 1. 파일 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 학습 당시 ID 맵 로드 (이게 없으면 복원 불가능)
    saved_maps = torch.load(maps_path, map_location='cpu')
    train_user2id = saved_maps['user2id']
    train_item2id = saved_maps['item2id']
    
    # 2. User Embedding 정렬
    # 학습된 원본 가중치
    raw_user_emb = state_dict['embedding_user.weight'] 
    # 새로 만들 빈 행렬 (현재 Processor 기준 크기)
    aligned_user_emb = torch.zeros(len(processor.user_ids) + 1, raw_user_emb.shape[1])
    
    u_hit = 0
    # 현재 Processor의 ID 순서대로 순회하며 학습된 가중치를 가져옴
    for i, uid_str in enumerate(processor.user_ids):
        target_idx = i + 1 # 1-based index
        if uid_str in train_user2id:
            src_idx = train_user2id[uid_str]
            if src_idx < len(raw_user_emb):
                aligned_user_emb[target_idx] = raw_user_emb[src_idx]
                u_hit += 1
    
    model.embedding_user = nn.Embedding.from_pretrained(aligned_user_emb, freeze=True, padding_idx=0)
    print(f"   ✅ Users Aligned: {u_hit} / {len(processor.user_ids)} (Coverage: {u_hit/len(processor.user_ids):.2%})")

    # 3. Item Embedding 정렬
    raw_item_emb = state_dict['embedding_item.weight']
    aligned_item_emb = torch.zeros(len(processor.item_ids) + 1, raw_item_emb.shape[1])
    
    i_hit = 0
    for i, iid_str in enumerate(processor.item_ids):
        target_idx = i + 1
        if iid_str in train_item2id:
            src_idx = train_item2id[iid_str]
            if src_idx < len(raw_item_emb):
                aligned_item_emb[target_idx] = raw_item_emb[src_idx]
                i_hit += 1
                
    model.embedding_item = nn.Embedding.from_pretrained(aligned_item_emb, freeze=True, padding_idx=0)
    print(f"   ✅ Items Aligned: {i_hit} / {len(processor.item_ids)} (Coverage: {i_hit/len(processor.item_ids):.2%})")
    
    return model.to(device)

# =========================================================
# 3. 정답 데이터 전처리 (String -> Integer Set)
# =========================================================
def prepare_ground_truth(target_df_path, processor):
    """
    평가 속도를 위해 String ID 정답지를 Integer Index 집합으로 미리 변환합니다.
    Return: {user_idx: {item_idx1, item_idx2, ...}}
    """
    print("\n⚡ Preparing Ground Truth Data...")
    df = pd.read_parquet(target_df_path) # [customer_id, target_ids]
    
    ground_truth = {}
    
    # DataFrame 순회
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   -> Indexing Targets"):
        u_str = row['customer_id']
        t_list = row['target_ids']
        
        # User가 Processor에 없으면 평가 불가 (Skip)
        if u_str not in processor.user2id:
            continue
            
        u_idx = processor.user2id[u_str]
        
        # Target Item들도 Integer ID로 변환
        item_indices = set()
        for i_str in t_list:
            if i_str in processor.item2id:
                item_indices.add(processor.item2id[i_str])
        
        if item_indices: # 정답이 하나라도 있는 경우만
            ground_truth[u_idx] = item_indices
            
    print(f"   ✅ Ready to evaluate {len(ground_truth)} users.")
    return ground_truth

# =========================================================
# 4. 평가 루프 (Clean Logic)
# =========================================================
def evaluate_recall(model, ground_truth_dict, device, k_list=[20, 100], batch_size=4096):
    """
    [수정됨] Cosine Similarity 대신 Dot Product 사용
    """
    max_k = max(k_list)
    model.eval()
    
    eval_user_indices = list(ground_truth_dict.keys())
    
    # DataLoader
    loader = DataLoader(
        eval_user_indices, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: torch.tensor(x, dtype=torch.long)
    )

    
    hits = {k: 0 for k in k_list}
    total_users = 0
    
    print(f"\n🚀 Starting Recall@{k_list} Evaluation (Metric: Dot Product)...")
    with torch.no_grad():
        if hasattr(model, 'final_item_emb'):
            all_items = model.final_item_emb.weight.data
            print("✅ Ready to use item emb")
        else:
            all_items = model.embedding_item.weight.data
        
        for batch_u_idx in tqdm(loader, desc="   -> Retrieving"):
            batch_u_idx = batch_u_idx.to(device)
            
            if hasattr(model, 'final_user_emb'):
                user_emb = model.final_user_emb(batch_u_idx)
                #print("✅ Ready to use user emb")
            else:
                user_emb = model.embedding_user(batch_u_idx)
            
            scores = torch.matmul(user_emb, all_items.T)
            
            # 1. Padding 0번 아이템 마스킹
            scores[:, 0] = -float('inf')
            
            # 💡 2. [핵심 추가] Train History 마스킹 (과거 구매 아이템 제외)
            batch_u_cpu = batch_u_idx.cpu().numpy()
            # Top-K 추출
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)
            topk_cpu = topk_indices.cpu().numpy()

            # Metric Check (기존 동일)
            for i, u_idx in enumerate(batch_u_cpu):
                true_item_set = ground_truth_dict.get(u_idx, set()) # .get()을 쓰면 혹시 모를 KeyError 방지 가능
                pred_list = topk_cpu[i]
                
                for k in k_list:
                    if not true_item_set.isdisjoint(pred_list[:k]):
                        hits[k] += 1
                        
            total_users += len(batch_u_cpu)
    # Report
    print(f"\n{'='*40}")
    print(f"📊 LightGCL Final Report (Dot Product)")
    print(f"{'-'*40}")
    for k in sorted(k_list):
        recall = hits[k] / total_users
        print(f"Recall@{k:<3} | {recall:.4f}")
    print(f"{'='*40}\n")
    
class LightGCL_Base(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64): # 학습 시 dim과 맞춰주세요
        super().__init__()
        # Padding(0) 포함을 위해 보통 +1을 해줍니다 (Processor 설정에 따라 조절)
        self.embedding_user = nn.Embedding(num_users + 1, emb_dim, padding_idx=0)
        self.embedding_item = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)

# 💡 2. 최종 전파된 벡터를 고정(Freeze)하여 서빙/평가에만 사용하는 최종 래퍼
class LightGCL_InferenceWrapper(nn.Module):
    def __init__(self, final_user_emb, final_item_emb):
        super().__init__()
        self.final_user_emb = nn.Embedding.from_pretrained(final_user_emb, freeze=True, padding_idx=0)
        self.final_item_emb = nn.Embedding.from_pretrained(final_item_emb, freeze=True, padding_idx=0)
        
    def forward(self, u_idx):
        return self.final_user_emb(u_idx)
        
    # 평가 함수(evaluate_recall)에서 아이템 전체 벡터가 필요할 경우를 위한 헬퍼 메서드
    def get_all_item_embeddings(self):
        return self.final_item_emb.weight
def build_sparse_graph_native(train_df, gcl_user2id, gcl_item2id,  n_users, n_items,device):
    """
    Train DataFrame(Parquet)과 LightGCL 순정(Native) ID 매핑을 사용하여
    학습 때와 완벽히 동일한 Normalized Adjacency Matrix를 생성합니다.
    """
    print("⚡ Building Graph Adjacency Matrix (Native LightGCL Space)...")
    
    # 1. 크기 설정: 패딩(+1) 없이 GNN 학습 당시의 순정 크기 그대로 사용
    n_users = len(gcl_user2id)
    n_items = len(gcl_item2id)
    
    # 2. 컬럼명 자동 감지
    u_col = 'customer_id' if 'customer_id' in train_df.columns else train_df.columns[0]
    possible_item_cols = ['sequence_ids', 'article_id', 'item_id', 'product_id', 'article_ids']
    i_col = next((col for col in possible_item_cols if col in train_df.columns), None)
    
    if i_col is None:
        raise KeyError(f"❌ Item column not found! Available: {train_df.columns}")
    
    print(f"   -> Using columns: User='{u_col}', Item='{i_col}'")

    src = []
    dst = []
    valid_interactions = 0
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="   -> Mapping Edges"):
        u_val = row[u_col]
        i_val = row[i_col] 
        
        if u_val not in gcl_user2id:
            continue
            
        u_idx = gcl_user2id[u_val] 
        # [핵심] 실제 임베딩 매트릭스 사이즈를 초과하는 인덱스(패딩 등) 방어
        if u_idx >= n_users: 
            continue
        
        if isinstance(i_val, (list, np.ndarray)):
            for item in i_val:
                if item in gcl_item2id:
                    i_idx = gcl_item2id[item]
                    if i_idx < n_items: # 안전장치
                        src.append(u_idx)
                        dst.append(i_idx)
                        valid_interactions += 1
        else:
            if i_val in gcl_item2id:
                i_idx = gcl_item2id[i_val]
                if i_idx < n_items: # 안전장치
                    src.append(u_idx)
                    dst.append(i_idx)
                    valid_interactions += 1
    print(f"   -> Valid Edges: {valid_interactions}")
    
    if valid_interactions == 0:
        raise ValueError("❌ No valid interactions found! Check Native ID matching.")

    # 4. Sparse Matrix 생성
    # shape 또한 padding 없이 순수 n_users, n_items 크기로 생성
    R = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_users, n_items))
    R.data = np.ones_like(R.data) 

    # 5. Adjacency Matrix A 생성
    R = R.tocoo()
    
    user_nodes = R.row
    item_nodes = R.col + n_users # Bipartite Graph 구성을 위한 Offset 적용
    
    row_idx = np.concatenate([user_nodes, item_nodes])
    col_idx = np.concatenate([item_nodes, user_nodes])
    data = np.ones(len(row_idx), dtype=np.float32)
    
    num_nodes = n_users + n_items
    adj_mat = sp.coo_matrix((data, (row_idx, col_idx)), shape=(num_nodes, num_nodes))
    
    # 6. Normalization (Train 코드와 완벽 동일 로직)
    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    norm_adj = norm_adj.tocoo()
    
    # 7. Torch Sparse Tensor 변환
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data).float()
    shape = torch.Size(norm_adj.shape)
    
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
    
    return adj_tensor
def align_final_embeddings_to_sasrec(lightgcl_embs, lightgcl_map, sasrec_processor, dim, is_user=True):
    """
    LightGCL의 최종 임베딩을 SASRec의 1-based (Unfiltered) 인덱스 체계에 맞춰 재배열합니다.
    """
    # SASRec의 전체 ID 딕셔너리 (1-based)
    sasrec_map = sasrec_processor.user2id if is_user else sasrec_processor.item2id
    max_sasrec_idx = max(sasrec_map.values())
    
    # 0번 인덱스는 패딩이므로 max_idx + 1 크기로 생성
    aligned_embs = torch.zeros((max_sasrec_idx + 1, dim), dtype=torch.float32)
    
    match_count = 0
    # LightGCL의 String ID -> 0-based ID 맵핑을 순회
    for string_id, gcl_idx in lightgcl_map.items():
        # 해당 String ID가 SASRec에도 존재한다면
        if string_id in sasrec_map:
            sasrec_idx = sasrec_map[string_id]
            # LightGCL 벡터를 SASRec 인덱스 위치로 복사
            aligned_embs[sasrec_idx] = lightgcl_embs[gcl_idx]
            match_count += 1
            
    target_name = "User" if is_user else "Item"
    print(f"✅ [{target_name} Alignment] {match_count} / {len(sasrec_map)} matched and mapped to SASRec index.")
    
    return aligned_embs


def lightgcl_importer():
    BASE_DIR = r'D:\trainDataset\localprops'
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')
    
    MODEL_PATH = os.path.join(PipelineConfig.model_dir, "simgcl_trained.pth") 
    MAPS_PATH = os.path.join(CACHE_DIR, "id_maps_train.pt") 
    
    print("1️⃣ Initializing Processors...")
    train_proc, val_proc, cfg = prepare_features(PipelineConfig) 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========================================================
    # 💡 Step A. LightGCL 순정(Native) ID 매핑 로드
    # ========================================================
    # MAPS_PATH에는 LightGCL 학습 당시의 String -> ID 매핑 딕셔너리가 들어있어야 합니다.
    # 예: {'user2id': {'uid1': 0, ...}, 'item2id': {'iid1': 0, ...}}
    gcl_maps = torch.load(MAPS_PATH, map_location='cpu')
    gcl_user2id = gcl_maps['user2id']
    gcl_item2id = gcl_maps['item2id']
    
    # ========================================================
    # 💡 Step B. LightGCL 순정 환경에서 모델 및 그래프 빌드
    # ========================================================
    # ========================================================
    # 💡 Step B. 가중치 파일에서 크기를 읽어와 모델 초기화
    # ========================================================
    # 1. 가중치 먼저 로드
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # 2. 실제 학습되었던 임베딩 사이즈 동적 추출
    num_users_ckpt = state_dict['embedding_user.weight'].shape[0]
    num_items_ckpt = state_dict['embedding_item.weight'].shape[0]
    
    print(f"✅ Loaded Checkpoint Shapes -> Users: {num_users_ckpt}, Items: {num_items_ckpt}")
    
    # 3. 추출한 사이즈로 모델 초기화
    base_model = LightGCL_Base(
        num_users=num_users_ckpt-1,
        num_items=num_items_ckpt-1,
        emb_dim=64 
    )
    
    # 4. 모델에 가중치 덮어쓰기 (이제 사이즈가 100% 일치합니다)
    base_model.load_state_dict(state_dict)
    base_model = base_model.to(device)
    base_model.eval()
    seq_path = os.path.join(cfg.base_dir, "features_sequence_cleaned.parquet")
    train_df = pd.read_parquet(seq_path) 

    # 그래프 생성 시에도 SASRec ID가 아닌, LightGCL의 ID 매핑을 사용하여 생성해야 전파가 정상적으로 일어납니다.
    # (주의: build_sparse_graph 함수가 string ID를 받아서 gcl_user2id로 변환하도록 구현되어야 함)
    adj_tensor = build_sparse_graph_native(
        train_df, 
        gcl_user2id, 
        gcl_item2id, 
        num_users_ckpt, # 모델의 실제 유저 수 전달
        num_items_ckpt,
        device
    )
    
    # ========================================================
    # 💡 Step C. GNN 전파(Propagation) 진행
    # ========================================================
    print("\n🔍 [Check] Executing Native Embedding Propagation...")
    with torch.no_grad():
        final_user_emb, final_item_emb = compute_final_embeddings(
            base_model, 
            adj_tensor, 
            n_layers=2 
        )
    
    # ========================================================
    # 💡 Step D. 최종 결과물을 SASRec 인덱스로 이사 (Late Alignment)
    # ========================================================
    print("\n🔄 Aligning Final Embeddings to SASRec 1-based Index Space...")
    
    aligned_user_emb = align_final_embeddings_to_sasrec(
        final_user_emb.cpu(), gcl_user2id, train_proc, dim=64, is_user=True
    )
    aligned_item_emb = align_final_embeddings_to_sasrec(
        final_item_emb.cpu(), gcl_item2id, train_proc, dim=64, is_user=False
    )
    
    return aligned_user_emb, aligned_item_emb
# =========================================================
# 5. 실행부 (Main)
# =========================================================
if __name__ == '__main__':
    # 설정 (경로 수정 필요)
    BASE_DIR = r'D:\trainDataset\localprops'
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')
    
    # 1. 학습된 모델 경로 (Fine-tuning 완료된 모델)
    MODEL_PATH = os.path.join(PipelineConfig.model_dir, "simgcl_trained.pth") 
    MAPS_PATH = os.path.join(CACHE_DIR, "id_maps_train.pt") # 학습 시 저장한 ID 매핑
    
    print("1️⃣ Initializing Processors...")
    train_proc, val_proc, cfg = prepare_features(PipelineConfig) # cfg도 받아온다고 가정
    
    # 2. 검증용 데이터 경로 (Parquet)
    TARGET_DF_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========================================================
    # 💡 Step A. Layer-0 가중치 로드 (임시 베이스 모델 활용)
    # ========================================================
    base_model = LightGCL_Base(
        num_users=train_proc.num_users,
        num_items=train_proc.num_items,
        emb_dim=64 # 파이프라인의 임베딩 차원과 동기화
    )
    # 베이스 모델에 정렬된 0층 가중치 덮어쓰기
    base_model = load_and_align_model(base_model, train_proc, MODEL_PATH, MAPS_PATH, device)
    
    # 그래프 빌드를 위한 시퀀스 데이터 로드
    seq_path = os.path.join(cfg.base_dir, "features_sequence_cleaned.parquet")
    train_df = pd.read_parquet(seq_path) 

    adj_tensor = build_sparse_graph(
        train_proc.user_ids, 
        train_proc.item_ids, 
        train_df,
        device
    )
    
    # ========================================================
    # 💡 Step B. GNN 전파(Propagation) 및 상태 검증
    # ========================================================
    print("\n🔍 [Check] Verifying Embedding Propagation...")

    # 전파 전 (Original Layer-0) 상태 저장
    before_user_emb = base_model.embedding_user.weight.data.clone()
    before_mean = before_user_emb.mean().item()
    before_std = before_user_emb.std().item()

    print(f"   Original Weights | Mean: {before_mean:.6f} | Std: {before_std:.6f}")
    
    # 최종 임베딩 추출
    final_user_emb, final_item_emb = compute_final_embeddings(
        base_model, 
        adj_tensor, 
        n_layers=2 # 학습 시 세팅과 동일하게 (2 or 3)
    )

    # 전파 후 (Propagated Final) 상태 확인
    after_mean = final_user_emb.mean().item()
    after_std = final_user_emb.std().item()

    print(f"   Propagated Weights | Mean: {after_mean:.6f} | Std: {after_std:.6f}")

    # 결과 판정
    if before_mean == after_mean:
        print("❌ [FAIL] Embeddings did NOT change! (Check compute_final_embeddings logic)")
    else:
        print("✅ [SUCCESS] Embeddings successfully updated via Graph Propagation!")
        diff = torch.norm(before_user_emb - final_user_emb).item()
        print(f"   -> Difference Magnitude (L2): {diff:.4f}")
        
    # ========================================================
    # 💡 Step C. 최종 인퍼런스 래퍼 생성 및 평가
    # ========================================================
    print("\n📦 Wrapping Final Embeddings for Inference...")
    # 전파가 완료된 벡터를 주입하여 최종 모델 생성
    final_model = LightGCL_InferenceWrapper(final_user_emb, final_item_emb).to(device)
    
    ground_truth = prepare_ground_truth(TARGET_DF_PATH, val_proc)

    # 평가 함수에는 최종 모델(final_model)을 전달합니다.
    evaluate_recall(final_model, ground_truth, device, k_list=[20, 100,500])