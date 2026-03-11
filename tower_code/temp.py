def inbatch_corrected_logq_loss_with_hybrid_hard_neg(
    user_emb, seq_item_emb, target_ids, user_ids, log_q_tensor,
    hn_item_emb=None, batch_hard_neg_ids=None,
    flat_history_item_ids=None,
    step_weights=None,
    # final_idx=None, 💡 [삭제] 더 이상 부분 연산이 필요 없습니다.
    temperature=0.07, # 💡 [수정] 낮춘 온도 적용
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
    else:
        weight_sum = None

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
    # 2. Hard Negative Processing (💡 전체 N개에 대해 동시 다발적 연산!)
    # -----------------------------------------------------------
    num_hn_to_use = 30
    
    # 이제 batch_hard_neg_ids는 전체 N개에 대한 HNM 후보군을 모두 담고 들어옵니다.
    if hn_item_emb is not None and batch_hard_neg_ids is not None:
        
        pool_multiplier = 3
        num_pool = num_hn_to_use * pool_multiplier  
        skip_top_k = 10

        # [STEP 1] 기울기(Gradient) 추적 없이 후보군 필터링
        with torch.no_grad():
            hn_emb_no_grad = hn_item_emb.detach() 
            # 💡 [핵심] user_emb [N, D]와 hn_emb_no_grad [N, Pool, D]의 bmm 연산
            hn_sim_no_grad = torch.bmm(user_emb.unsqueeze(1), hn_emb_no_grad.transpose(1, 2)).squeeze(1)
            
            absolute_fn_mask = torch.zeros_like(hn_sim_no_grad, dtype=torch.bool, device=device)
            if flat_history_item_ids is not None:
                absolute_fn_mask = (batch_hard_neg_ids.unsqueeze(2) == flat_history_item_ids.unsqueeze(1)).any(dim=2)
            
            # 💡 [최적화] 15% 불완전한 스텝을 위해 경계를 0.80 -> 0.85로 관대하게 변경
            boundary_ratio = 0.85
            dynamic_boundary = pos_sim.unsqueeze(1) * boundary_ratio 
            dynamic_fn_mask = hn_sim_no_grad >= dynamic_boundary
            
            final_fn_mask = absolute_fn_mask | dynamic_fn_mask
            masked_sims = hn_sim_no_grad.masked_fill(final_fn_mask, -1e4)
            
            _, top_idx_all = torch.topk(masked_sims, num_pool + skip_top_k, dim=1)
            top_idx_pool = top_idx_all[:, skip_top_k:]

            rand_idx = torch.rand(N, num_pool, device=device).argsort(dim=1)[:, :num_hn_to_use]
            top_idx = torch.gather(top_idx_pool, 1, rand_idx)

        # [STEP 2] 본 학습용 임베딩 조립
        batch_hn_ids_final = torch.gather(batch_hard_neg_ids, 1, top_idx)
        top_idx_expanded = top_idx.unsqueeze(-1).expand(-1, -1, hn_item_emb.size(2))
        hn_emb_final = torch.gather(hn_item_emb, 1, top_idx_expanded).detach()

        # [STEP 3] 본 연산 (Gradient 흐름)
        hn_sim = torch.bmm(user_emb.unsqueeze(1), hn_emb_final.transpose(1, 2)).squeeze(1) 
        hn_logits = (hn_sim / temperature) * alpha
        
        if lambda_logq > 0.0:
            hn_log_q = log_q_tensor[batch_hn_ids_final] 
            hn_logits = hn_logits - (hn_log_q * lambda_logq)
        
        final_safety_mask = hn_sim >= (pos_sim.unsqueeze(1) * boundary_ratio)
        hn_logits = hn_logits.masked_fill(final_safety_mask, SAFE_NEG_INF)

        # 💡 [핵심 최적화] 퍼즐 조립 삭제. 바로 Cat!
        logits = torch.cat([logits, hn_logits], dim=1)
        
        if return_metrics:
            metrics['hn/discarded_ratio'] = dynamic_fn_mask.float().mean().item()
            valid_hn_sim = hn_sim[~final_safety_mask]
            metrics['sim/hn_true_hard'] = valid_hn_sim.mean().item() if valid_hn_sim.numel() > 0 else 0.0
            
    # -----------------------------------------------------------
    # 3. Loss 계산 및 Metrics 등 (기존 로직과 완전히 동일하여 생략)
    # -----------------------------------------------------------
    # ... (기존 Cross Entropy 연산 부분 동일) ...
    
    
# HNM 타겟 준비 (이제 남은 N개 전체에 대해 진행)
batch_hn_item_emb = None
batch_hard_neg_ids = None

if hard_neg_pool_tensor is not None:
    # flat_targets는 이미 Session-last + 15% 마스크가 적용된 상태입니다.
    batch_hard_neg_ids = hard_neg_pool_tensor[flat_targets] 
    batch_hn_item_emb = norm_item_embeddings[batch_hard_neg_ids]

main_loss, b_metrics = inbatch_corrected_logq_loss_with_hybrid_hard_neg(
    user_emb=flat_user_emb, seq_item_emb=batch_seq_item_emb,
    target_ids=flat_targets, user_ids=flat_user_ids,
    log_q_tensor=log_q_tensor, 
    hn_item_emb=batch_hn_item_emb, batch_hard_neg_ids=batch_hard_neg_ids, 
    flat_history_item_ids=flat_history_item_ids, 
    step_weights=flat_weights, # Loss 비중(Weight) 용도로만 사용됨
    # final_idx 삭제
    temperature=0.07, lambda_logq=cfg.lambda_logq, alpha=1.0,
    soft_penalty_weight=cfg.soft_penalty_weigh, margin=0.0, return_metrics=True
)