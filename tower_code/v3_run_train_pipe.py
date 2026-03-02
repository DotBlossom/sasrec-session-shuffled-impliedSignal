import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import wandb
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)
    
from gnn_model.v1_evaluate_lightgcl import lightgcl_importer
from params_config import PipelineConfig
from sheduler import EarlyStopping, get_cosine_schedule_with_warmup
from v3_lightgcl_util import load_aligned_lightgcl_user_embeddings
from v3_model_usertower import create_category_mapping_tensor, mine_category_constrained_hard_negatives
from v3_train_usertower import evaluate_model, train_user_tower_all_time, train_user_tower_all_time_gcl_dil, verify_id_alignment
from v3_utils import create_dataloaders, load_aligned_pretrained_embeddings, load_item_metadata_hashed, prepare_features, setup_environment, setup_models
import torch.nn.functional as F

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
    processor, val_processor, cfg = prepare_features(cfg)
    
    # -----------------------------------------------------------
    # 💡 하드 네거티브 및 하이퍼파라미터 세팅
    # -----------------------------------------------------------
    cfg.lr = 1.8e-3               # 기존 학습률 유지
    cfg.epochs = 24             # 이미 11에포크를 돌았으므로, 추가로 30에포크면 충분
    HN_K = 100                   # 20개 추출 후 내부 0.95 제약으로 필터링
    
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # 2. Data & Metadata 가져오기
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.base_dir, cfg.pretrained_dim)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    val_processor.i_side_arr = processor.i_side_arr
    
    train_loader = create_dataloaders(processor, cfg, "2020-09-16", aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, "2020-09-22", aligned_vecs, is_train=False)
    
    json_path = os.path.join(cfg.base_dir, "filtered_data_reinforced.json")
    item_category_ids = create_category_mapping_tensor(json_path, processor, device)

    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2",
        name=f"Resume_SessionWeight_lr{cfg.lr}_K{HN_K}", 
        config=cfg.__dict__ 
    )
    
    # -----------------------------------------------------------
    # 3. Models Setup & 💡 Epoch 11 베이스라인 가중치 로드
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, None, log_q_tensor) 
    
    # 💡 [요청 사항 1] 모델 경로 지정 및 로드
    base_user_pth = os.path.join(cfg.base_dir, "best_user_tower_hn_v3_hnm_e16.pth")
    base_item_pth = os.path.join(cfg.base_dir, "best_item_tower_hn_v3_hnm_e16.pth")
    
    print(f"📥 Loading Baseline Models from Epoch 11...")
    user_tower.load_state_dict(torch.load(base_user_pth, map_location=device))
    item_tower.load_state_dict(torch.load(base_item_pth, map_location=device))
    print(f"✅ Baseline Models loaded successfully.")

    # 💡 [요청 사항 2] 처음부터 Item Tower 해동 (Unfreeze) 및 비대칭 LR 적용
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
    warmup_steps = int(total_steps * 0.05) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01) 
    
    early_stopping = EarlyStopping(patience=7, mode='max')

    # -----------------------------------------------------------
    # 4. Training Loop (💡 자연스러운 출력을 위해 에포크 10부터 시작)
    # -----------------------------------------------------------
    start_epoch = 16
    end_epoch = start_epoch + cfg.epochs
    
    epoch_hn_pool = None

    for epoch in range(start_epoch, end_epoch):
        
        # 💡 분기 삭제됨: 이미 위에서 Item Tower를 해동했으므로 epoch 5 로직 제거
        
        # 💡 재학습 시작(Epoch 12)과 동시에 마이닝 즉시 발동
        if (epoch - 10) % 2 == 0:
            print(f"\n🔍 [Epoch {epoch} Start] Mining Category-Constrained Hard Negatives (K={HN_K})...")
            item_tower.eval() 
            with torch.no_grad():
                all_item_embs = item_tower.get_all_embeddings()
                norm_item_embs = F.normalize(all_item_embs, p=2, dim=1)
                
                epoch_hn_pool = mine_category_constrained_hard_negatives(
                    norm_item_embs, item_category_ids, k=HN_K, device=device
                )
                
                del all_item_embs, norm_item_embs
                torch.cuda.empty_cache()
                
            item_tower.train()

        else: 
            print(f"\n♻️ [Epoch {epoch} Start] Using Cached Hard Negative Pool (No Mining Overheads)")
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}]  - Current LR: {current_lr:.8f}")  
        
        # ------------------- 훈련 (Train) -------------------
        avg_loss = train_user_tower_all_time(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, 
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=epoch_hn_pool, # 💡 방금 뽑은 최신 HN 풀 전달
            scheduler=scheduler, 
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
        )
        
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
            
            # 💡 [요청 사항 3] 저장 이름 변경 (session_weights 명시)
            save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_hn_v3_hnm_alpha.pth")
            save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_hn_v3_hnm_alpha.pth")
            
            torch.save(user_tower.state_dict(), save_user_pth)
            torch.save(item_tower.state_dict(), save_item_pth)
            print(f"   💾 Best model weights saved to: {save_user_pth}")
            
        if early_stopping.early_stop:
            print(f"\n🛑 조기 종료 발동! {early_stopping.patience} Epoch 동안 Recall@20이 개선되지 않았습니다.")
            print(f"🏆 최종 최고 Recall@20: {early_stopping.best_score:.2f}%")
            break
            
    print("\n🎉 Resume Pipeline Execution Finished Successfully!")
    
def align_teacher_embeddings(processor, lightgcl_user_embs, gcl_user2id, device):
    """
    SASRec의 인덱스(1~N)에 맞춰 LightGCL의 임베딩을 재배치합니다.
    """
    num_sas_users = processor.num_users + 1 # 0번 패딩 포함
    emb_dim = lightgcl_user_embs.size(1)
    
    # 1. SASRec 인덱스 체계에 맞는 빈 텐서 생성
    aligned_embs = torch.zeros((num_sas_users, emb_dim), device=device)
    
    print(f"🔄 Re-aligning Teacher Embeddings to SASRec Index...")
    match_count = 0
    
    # 2. SASRec에 있는 유저들을 순회하며 LightGCL의 벡터를 찾아 채워넣음
    for uid, sas_idx in processor.user2id.items():
        if uid in gcl_user2id:
            gcl_idx = gcl_user2id[uid]
            aligned_embs[sas_idx] = lightgcl_user_embs[gcl_idx]
            match_count += 1
            
    print(f"✅ Alignment Complete: {match_count}/{len(processor.user2id)} users matched.")
    return aligned_embs 
def resume_pipeline_session_weights_gcl():
    print("🚀 Starting User Tower Resume Pipeline (Session Weights, HNM & Latent KD)...")


    
    
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
    # 💡 하드 네거티브 및 하이퍼파라미터 세팅
    # -----------------------------------------------------------

    cfg.lr = 1.8e-3              # 💡 Fine-tuning이므로 기존 LR의 1/10 수준으로 대폭 축소
    cfg.epochs = 20                    # 💡 이미 11~16에포크 수렴했으므로 10에포크면 Alignment 충분
    HN_K = 100 
    LAMBDA_ALIGN = 0.05                # 💡 KD Alignment 비중 (너무 크면 시퀀스 정보가 망가짐)
    
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
    
    json_path = os.path.join(cfg.base_dir, "filtered_data_reinforced.json")
    item_category_ids = create_category_mapping_tensor(json_path, processor, device)

    wandb.init(
        project="SASRec-User-Tower-causality-Optimization-v2",
        name=f"gcl_kd_lr{cfg.lr}_K{HN_K}", 
        config=cfg.__dict__ 
    )
    
    CACHE_DIR = os.path.join(cfg.base_dir, 'cache')
    MAPS_PATH = os.path.join(CACHE_DIR, "id_maps_train.pt") # 학습 시 저장한 ID 매핑    

    # ===========================================================
    # 💡 [신규] LightGCL User Embedding 로드 (Teacher)
    # ===========================================================
    lightgcl_model_path = os.path.join(cfg.model_dir, "simgcl_trained.pth")
    lightgcl_cache_dir = os.path.join(cfg.base_dir, "cache")
    lightgcl_maps = torch.load(os.path.join(CACHE_DIR, "id_maps_train.pt"))
    gcl_user2id = lightgcl_maps['user2id']
    # LightGCL User 임베딩 정렬하여 로드 (D=cfg.embed_dim)
    aligned_lightgcl_user_embs = lightgcl_importer()
    # ===========================================================
    aligned_lightgcl_user_embs= align_teacher_embeddings(processor,aligned_lightgcl_user_embs,gcl_user2id,device)
    
    
    # 3. Models Setup & Epoch 16 베이스라인 로드
    user_tower, item_tower = setup_models(cfg, device, None, log_q_tensor) 
    
    base_user_pth = os.path.join(cfg.base_dir, "best_user_tower_hn_v3_hnm_e16.pth")
    base_item_pth = os.path.join(cfg.base_dir, "best_item_tower_hn_v3_hnm_e16.pth")
    user_tower.load_state_dict(torch.load(base_user_pth, map_location=device), strict=False)
    item_tower.load_state_dict(torch.load(base_item_pth, map_location=device))

    item_tower.set_freeze_state(False)
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

    start_epoch = 16 # 에포크 16 이후부터 재개
    end_epoch = start_epoch + cfg.epochs
    epoch_hn_pool = None


    # 2. 검증 함수 실행
    #gcl_user2id = verify_id_alignment(processor, MAPS_PATH )
    #del gcl_user2id
    for epoch in range(start_epoch, end_epoch):

        if (epoch - 10) % 2 == 0:
            print(f"\n🔍 [Epoch {epoch} Start] Mining Category-Constrained Hard Negatives (K={HN_K})...")
            item_tower.eval() 
            with torch.no_grad():
                all_item_embs = item_tower.get_all_embeddings()
                norm_item_embs = F.normalize(all_item_embs, p=2, dim=1)
                
                epoch_hn_pool = mine_category_constrained_hard_negatives(
                    norm_item_embs, item_category_ids, k=HN_K, device=device
                )
                
                del all_item_embs, norm_item_embs
                torch.cuda.empty_cache()
                
            item_tower.train()

        else: 
            print(f"\n♻️ [Epoch {epoch} Start] Using Cached Hard Negative Pool (No Mining Overheads)")
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch}]  - Current LR: {current_lr:.8f}") 
        # ------------------- 훈련 (Train) -------------------
        avg_loss = train_user_tower_all_time_gcl_dil(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, 
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            hard_neg_pool_tensor=epoch_hn_pool, 
            scheduler=scheduler, 
            processor=processor,
            seq_labels=SEQ_LABELS,
            static_labels=STATIC_LABELS,
            aligned_lightgcl_user_embs=aligned_lightgcl_user_embs, # 💡 Teacher 주입
            lambda_align=LAMBDA_ALIGN                              # 💡 Alignment 강도 주입
        )
        
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
            
            # 💡 [요청 사항 3] 저장 이름 변경 (session_weights 명시)
            save_user_pth = os.path.join(cfg.model_dir, "best_user_tower_hn_v3_hnm_gcl.pth")
            save_item_pth = os.path.join(cfg.model_dir, "best_item_tower_hn_v3_hnm_gcl.pth")
            
            torch.save(user_tower.state_dict(), save_user_pth)
            torch.save(item_tower.state_dict(), save_item_pth)
            print(f"   💾 Best model weights saved to: {save_user_pth}")
            
        if early_stopping.early_stop:
            print(f"\n🛑 조기 종료 발동! {early_stopping.patience} Epoch 동안 Recall@20이 개선되지 않았습니다.")
            print(f"🏆 최종 최고 Recall@20: {early_stopping.best_score:.2f}%")
            break
            
    print("\n🎉 Resume Pipeline Execution Finished Successfully!")
    

def evaluate_weighted_score_ensemble_recall_safe(
    model, item_tower, dataloader, target_df_path, 
    gnn_user_matrix, gnn_item_matrix, device, processor, 
    k_list=[10, 20, 200, 500], alpha_step=0.1, candidate_pool_size=1000
):
    max_k = max(k_list)
    pool_k = max(candidate_pool_size, max_k * 2)
    
    print(f"\n🚀 Starting Safe Weighted Score Ensemble (Pool K: {pool_k})...")
    
    model.eval()
    item_tower.eval()
    
    # ---------------------------------------------------------
    # 1. Target Data Load
    # ---------------------------------------------------------
    print(f"🎯 Loading targets from: {target_df_path}")
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    del target_df

    # ---------------------------------------------------------
    # 2. Pre-computation (0-Based Alignment)
    # ---------------------------------------------------------
    print("⚡ Pre-computing Item Vectors and Aligning Indices...")
    with torch.no_grad():
        full_seq_item_embs = item_tower.get_all_embeddings()
        all_seq_item_vecs = F.normalize(full_seq_item_embs, p=2, dim=1)
        
        # 0번 인덱스 패딩 추가 (SASRec 1-based 매칭용)
 
        aligned_gnn_item_matrix = gnn_item_matrix.to(device)
        all_gnn_item_vecs = F.normalize(aligned_gnn_item_matrix, p=2, dim=1)
        
        pad_user = torch.zeros(1, gnn_user_matrix.size(1), dtype=gnn_user_matrix.dtype)
        aligned_gnn_user_matrix = torch.cat([pad_user, gnn_user_matrix.cpu()]).to(device)

    alphas = [round(x, 1) for x in np.arange(1.0, -0.01, -alpha_step)]
    results = {a: {k: 0.0 for k in k_list} for a in alphas}
    total_valid_users = 0

    # ---------------------------------------------------------
    # 3. Main Loop
    # ---------------------------------------------------------
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="   -> Weighted Score Fusion")):
            user_ids = batch['user_ids']
            item_ids = batch['item_ids'].to(device, non_blocking=True)
            padding_mask = batch['padding_mask'].to(device, non_blocking=True)
            
            # =========================================================
            # 💡 [핵심 수정] 정답지 + GNN 인덱스 에러 방지용 이중 필터링
            # =========================================================
            valid_idx_list = []
            valid_user_ids = []
            
            for i, uid in enumerate(user_ids):
                # 1) 정답지가 없는 유저 스킵 (Numpy Array 호환)
                if uid not in target_dict or len(target_dict[uid]) == 0:
                    continue
                
                # 2) processor에 아예 없거나, GNN Matrix 크기를 초과하는 인덱스는 스킵 (IndexError 방지)
                sasrec_idx = processor.user2id.get(uid, -1)
                if sasrec_idx == -1 or sasrec_idx >= aligned_gnn_user_matrix.size(0):
                    continue 
                
                valid_idx_list.append(i)
                valid_user_ids.append(uid)

            if not valid_idx_list: 
                continue
                
            v_idx = torch.tensor(valid_idx_list, device=device)
            current_batch_size = len(valid_user_ids)
            
            # =========================================================
            # [Step A] User Vector Generation
            # =========================================================
            if 'pretrained_vecs' in batch:
                pretrained_vecs = batch['pretrained_vecs'].to(device, non_blocking=True)
            else:
                pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device)

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
                'target_week': batch['target_week'].to(device),
                'padding_mask': padding_mask,
                'training_mode': False
            }

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = model(**forward_kwargs)
                
            if output.dim() == 3:
                lengths = (~padding_mask).sum(dim=1)
                last_indices = (lengths - 1).clamp(min=0)
                batch_range = torch.arange(output.size(0), device=device)
                last_user_emb = output[batch_range, last_indices]
            else:
                last_user_emb = output
                
            user_seq_vecs = F.normalize(last_user_emb[v_idx], p=2, dim=1)

            # 안전이 보장된 인덱스로 GNN User Vector 추출
            gnn_u_indices = torch.tensor([processor.user2id[uid] for uid in valid_user_ids], device=device)
            user_gnn_vecs = F.normalize(aligned_gnn_user_matrix[gnn_u_indices], p=2, dim=1)

            # =========================================================
            # [Step B & C] Candidate Pool & Score Calculation
            # =========================================================
            scores_seq_all = torch.matmul(user_seq_vecs, all_seq_item_vecs.T)
            _, indices_seq_top = torch.topk(scores_seq_all, k=pool_k, dim=1)

            scores_gnn_all = torch.matmul(user_gnn_vecs, all_gnn_item_vecs.T)
            _, indices_gnn_top = torch.topk(scores_gnn_all, k=pool_k, dim=1)

            combined_indices = torch.cat([indices_seq_top, indices_gnn_top], dim=1)
            flat_indices = combined_indices.view(-1)
            
            batch_seq_items = all_seq_item_vecs[flat_indices].view(current_batch_size, -1, all_seq_item_vecs.shape[1])
            batch_gnn_items = all_gnn_item_vecs[flat_indices].view(current_batch_size, -1, all_gnn_item_vecs.shape[1])
            
            s_seq = (user_seq_vecs.unsqueeze(1) * batch_seq_items).sum(dim=-1)
            s_gnn = (user_gnn_vecs.unsqueeze(1) * batch_gnn_items).sum(dim=-1)

            # =========================================================
            # [Step D] Normalization & Target Prep
            # =========================================================
            def min_max_norm(tensor):
                min_val = tensor.min(dim=1, keepdim=True)[0]
                max_val = tensor.max(dim=1, keepdim=True)[0]
                return (tensor - min_val) / (max_val - min_val + 1e-9)
            
            s_seq_norm = min_max_norm(s_seq)
            s_gnn_norm = min_max_norm(s_gnn)
            
            batch_targets = []
            for uid in valid_user_ids:
                raw_targets = target_dict[uid]
                if isinstance(raw_targets, str) or not hasattr(raw_targets, '__iter__'):
                    raw_targets = [raw_targets]
                
                # 정답 아이템 중 GNN 매트릭스의 아이템 차원을 넘어서는 오류도 함께 방지합니다.
                actual_indices = set(
                    processor.item2id[iid] for iid in raw_targets 
                    if iid in processor.item2id and processor.item2id[iid] < aligned_gnn_item_matrix.size(0)
                )
                batch_targets.append(actual_indices)
                
            total_valid_users += len(valid_user_ids)
            combined_indices_cpu = combined_indices.cpu().numpy()

            # =========================================================
            # [Step E] Alpha Sweep & Recall@K
            # =========================================================
            for alpha in alphas:
                final_scores = alpha * s_gnn_norm + (1.0 - alpha) * s_seq_norm
                
                _, local_topk_indices = torch.topk(final_scores, k=max_k + 50, dim=1)
                local_topk_indices = local_topk_indices.cpu().numpy()
                
                for i, actual_indices in enumerate(batch_targets):
                    if not actual_indices: 
                        continue
                    
                    pred_global_ids = combined_indices_cpu[i][local_topk_indices[i]]
                    _, unique_idx = np.unique(pred_global_ids, return_index=True)
                    pred_unique = pred_global_ids[np.sort(unique_idx)]
                    
                    for k in k_list:
                        top_k_items = set(pred_unique[:k])
                        hit_items = actual_indices.intersection(top_k_items)
                        user_recall = len(hit_items) / len(actual_indices)
                        
                        results[alpha][k] += user_recall

    # ---------------------------------------------------------
    # 4. Report
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"📊 Safe Weighted Score Fusion Report (Recall@K)")
    print(f"   -> Valid Users Evaluated: {total_valid_users:,}")
    print(f"{'-'*80}")
    
    header = f"{'Alpha(GNN)':<12} | {'GNN:Seq Ratio':<15}"
    for k in k_list:
        header += f" | {f'Recall@{k}':<10}"
    print(header)
    print(f"{'-'*80}")
    
    best_alpha = -1
    best_score = -1
    
    for alpha in sorted(results.keys(), reverse=True):
        scores = {}
        row_str = f"{alpha:<12.1f} | {f'{int(alpha*10)} : {int((1-alpha)*10)}':<15}"
        
        for k in k_list:
            avg_recall = (results[alpha][k] / total_valid_users) * 100 if total_valid_users > 0 else 0
            scores[k] = avg_recall
            row_str += f" | {avg_recall:>7.2f}%"
        print(row_str)
        
        if len(k_list) > 1 and scores[k_list[1]] > best_score:
            best_score = scores[k_list[1]]
            best_alpha = alpha
            
    print(f"{'='*80}")
    if len(k_list) > 1:
        print(f"🏆 Best Weighted Alpha (Recall@{k_list[1]} 기준): {best_alpha}")

    del full_seq_item_embs, aligned_gnn_user_matrix, aligned_gnn_item_matrix
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    # 5에포크까지 학습했으므로 6번부터 재개
    #run_resume_pipeline(resume_epoch=16, last_best_recall=22.60)
    #run_pipeline_opt_v2()
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    
    # -----------------------------------------------------------
    # 💡 하드 네거티브 및 하이퍼파라미터 세팅
    # -----------------------------------------------------------
    cfg.lr = 1.8e-3               # 기존 학습률 유지
    cfg.epochs = 24             # 이미 11에포크를 돌았으므로, 추가로 30에포크면 충분
    HN_K = 100                   # 20개 추출 후 내부 0.95 제약으로 필터링
    
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
    
    json_path = os.path.join(cfg.base_dir, "filtered_data_reinforced.json")
    item_category_ids = create_category_mapping_tensor(json_path, processor, device)

    
    # -----------------------------------------------------------
    # 3. Models Setup & 💡 Epoch 11 베이스라인 가중치 로드
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, None, log_q_tensor) 
    
    # 💡 [요청 사항 1] 모델 경로 지정 및 로드
    base_user_pth = os.path.join(cfg.base_dir, "best_user_tower_hn_v3_hnm_alpha (1).pth")
    base_item_pth = os.path.join(cfg.base_dir, "best_item_tower_hn_v3_hnm_alpha (1).pth")
    
    print(f"📥 Loading Baseline Models from Epoch 11...")
    user_tower.load_state_dict(torch.load(base_user_pth, map_location=device))
    item_tower.load_state_dict(torch.load(base_item_pth, map_location=device))
    print(f"✅ Baseline Models loaded successfully.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # =====================================================================
    # 2. GNN (LightGCL) 임베딩 로드
    # =====================================================================
    # 정렬 로직을 거치지 않은 순수 LightGCL 임베딩(0-based)을 그대로 불러옵니다.
    print("📥 Loading LightGCL Embeddings...")
    gnn_user_matrix,gnn_item_matrix  = lightgcl_importer()

    # =====================================================================
    # 3. 앙상블 평가 함수 실행
    # =====================================================================
    print("🚀 Launching Ensemble Evaluation...")
    ensemble_results = evaluate_weighted_score_ensemble_recall_safe(
        model=user_tower,                        # SASRec User Tower
        item_tower=item_tower,              # SASRec Item Tower
        dataloader=val_loader,              # Validation DataLoader
        target_df_path=TARGET_VAL_PATH,    # Parquet 정답지 경로
        gnn_user_matrix=gnn_user_matrix,    # 로드한 LightGCL 유저 임베딩 (0-based)
        gnn_item_matrix=gnn_item_matrix,    # 로드한 LightGCL 아이템 임베딩 (0-based)
        device=device,                      # GPU Device
        processor=processor,                # FeatureProcessor_v3 인스턴스
        k_list=[10, 20, 200, 500],          # 측정할 Top-K 리스트
        alpha_step=0.1,                     # 1.0부터 0.0까지 0.1 단위로 탐색 (총 11개 스텝)
        candidate_pool_size=1000            # 각 모델에서 추출할 후보군 크기 (max_k인 500의 2배로 설정)
    )
    #run_resume_pipeline_v2()