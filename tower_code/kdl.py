import torch.nn.functional as F
'''
gcl_user_emb = torch.load('lightgcl_user_emb.pth').to(device) # (Num_users, 64)
gcl_item_emb = torch.load('lightgcl_item_emb.pth').to(device) # (Num_items, 64)

# 2. 그래디언트 업데이트 방지 (동결)
gcl_user_emb.requires_grad = False
gcl_item_emb.requires_grad = False
# -------------------------------------------------------------------
# [1] Teacher Logits 계산 (동결된 LightGCL 벡터 사용)
# -------------------------------------------------------------------
teacher_user_vecs = gcl_user_emb[flat_user_ids] 
teacher_item_vecs = gcl_item_emb[flat_targets]  

# Teacher Raw Logits (크기 보정 없음, LogQ 없음)
teacher_logits_raw = torch.matmul(teacher_user_vecs, teacher_item_vecs.T)

# -------------------------------------------------------------------
# [2] Student Logits 계산 (KD 전용: LogQ 적용 전 순수 내적)
# -------------------------------------------------------------------
# 현재 SASRec은 L2 정규화가 되어 있으므로 코사인 유사도가 나옴 (-1 ~ 1)
student_logits_raw = torch.matmul(flat_user_emb, norm_item_embeddings.T)

# -------------------------------------------------------------------
# [3] Scale 정렬 (Z-Score Standardization per User)
# -------------------------------------------------------------------
# 두 모델의 점수 스케일이 다르므로, 행(유저) 단위로 평균 0, 표준편차 1로 맞춰줌
t_mean = teacher_logits_raw.mean(dim=-1, keepdim=True)
t_std = teacher_logits_raw.std(dim=-1, keepdim=True) + 1e-8
teacher_logits_norm = (teacher_logits_raw - t_mean) / t_std

s_mean = student_logits_raw.mean(dim=-1, keepdim=True)
s_std = student_logits_raw.std(dim=-1, keepdim=True) + 1e-8
student_logits_norm = (student_logits_raw - s_mean) / s_std

# -------------------------------------------------------------------
# [4] KL-Divergence Loss 계산
# -------------------------------------------------------------------
# Z-score로 정렬되었으므로 T=1.0 혹은 약간만 부드럽게 T=2.0 정도를 씁니다.
kd_temp = 2.0 

# Teacher는 확률 분포로 (Softmax, 정답지 역할)
teacher_probs = F.softmax(teacher_logits_norm / kd_temp, dim=-1)

# Student는 Log 확률 분포로 (LogSoftmax, 예측지 역할)
student_log_probs = F.log_softmax(student_logits_norm / kd_temp, dim=-1)

# KD Loss 계산 (batchmean 평균)
kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
kd_loss = kd_loss * (kd_temp ** 2)

# -------------------------------------------------------------------
# [5] 최종 Loss 합산
# -------------------------------------------------------------------
# main_loss는 기존 개발자님의 inbatch_corrected_logq_loss_with_hard_neg_margin 결과값
alpha = 0.2 # GCL 지식 반영 비율
total_loss = main_loss + (alpha * kd_loss)
'''