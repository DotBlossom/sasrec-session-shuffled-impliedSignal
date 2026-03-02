import math
import os
import torch
from torch.optim.lr_scheduler import LambdaLR
import wandb

# =====================================================================
# 1. 스케줄러 & 조기 종료 클래스 정의 (기존 코드 상단이나 utils 모듈에 추가)
# =====================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.01):
    """최소 학습률 하한선(min_lr_ratio)이 보장되는 코사인 스케줄러"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 0.0으로 떨어지지 않고 초기 LR의 n% 수준(min_lr_ratio)으로 수렴
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
            
class AdaptiveHNScheduler:
    def __init__(self, initial_ex_top_k=75, threshold=0.04, window_size=3, step_size=25, max_ex_top_k=200):
        self.ex_top_k = initial_ex_top_k
        self.threshold = threshold
        self.window_size = window_size
        self.step_size = step_size
        self.max_ex_top_k = max_ex_top_k
        self.danger_history = []

    def step(self, current_danger_ratio):
        """
        매 에폭마다 danger_zone_ratio를 입력받아 상태를 업데이트합니다.
        return: (업데이트 여부(bool), 현재 ex_top_k, 이동평균치)
        """
        self.danger_history.append(current_danger_ratio)
        
        # 윈도우 사이즈 유지
        if len(self.danger_history) > self.window_size:
            self.danger_history.pop(0)

        moving_avg = None
        updated = False

        # 최근 3에폭의 데이터가 모두 모였을 때만 판별
        if len(self.danger_history) == self.window_size:
            moving_avg = sum(self.danger_history) / self.window_size
            
            # 이동평균이 임계치(4%)를 초과하면 확장!
            if moving_avg > self.threshold:
                if self.ex_top_k < self.max_ex_top_k:
                    self.ex_top_k = min(self.ex_top_k + self.step_size, self.max_ex_top_k)
                    updated = True
                    # 연속 확장 방지 - 근데 잘모르겠음, 늘려도 터질려하면? 최소한 이전거는 저장해야지.
                    self.danger_history = [] 

        return updated, self.ex_top_k, moving_avg
    
    
'''
if current_danger_ratio >= 0.06:
    ex_top_k += 5  # 숨통만 미세하게 트여줌
    self.danger_history = []  # 쿨타임 리셋!
    return True, ex_top_k


'''