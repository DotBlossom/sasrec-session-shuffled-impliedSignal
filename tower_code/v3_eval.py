import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

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
        pad_item = torch.zeros(1, gnn_item_matrix.size(1), dtype=gnn_item_matrix.dtype)
        aligned_gnn_item_matrix = torch.cat([pad_item, gnn_item_matrix.cpu()]).to(device)
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
                # 1) 정답지가 없는 유저 스킵
                if uid not in target_dict or not target_dict[uid]:
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
    evaluate_weighted_score_ensemble_recall_safe()