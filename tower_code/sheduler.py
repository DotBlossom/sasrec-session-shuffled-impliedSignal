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
def get_warmup_hold_decay_schedule(optimizer, num_warmup_steps, num_training_steps, hold_ratio=0.6, min_lr_ratio=0.1):
    """
    동적 HNM 파인튜닝에 최적화된 사다리꼴(Trapezoidal) 스케줄러
    1. Warmup: 0 -> Max LR (부드러운 출발)
    2. Hold: Max LR 유지 (지속적인 HNM 충격 흡수 및 공간 복구)
    3. Decay: Linear 감소 (극후반부 수렴)
    """
    def lr_lambda(current_step):
        # 1. Warmup Phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 2. Hold Phase (가장 중요한 구간: 평탄하게 유지)
        hold_steps = int(num_training_steps * hold_ratio)
        if current_step < num_warmup_steps + hold_steps:
            return 1.0  # 설정한 Max LR을 그대로 유지
        
        # 3. Decay Phase (나머지 구간 동안 서서히 감소)
        decay_steps = num_training_steps - num_warmup_steps - hold_steps
        decay_current = current_step - num_warmup_steps - hold_steps
        progress = float(decay_current) / float(max(1, decay_steps))
        
        # 1.0에서 min_lr_ratio까지 선형적으로 감소
        return 1.0 - progress * (1.0 - min_lr_ratio)
        
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

class TrendBasedHNScheduler:
    def __init__(self, initial_ex_top_k=50, margin_drop_ratio=0.95, penalty_rise_ratio=1.05, window_size=3, step_size=25, max_ex_top_k=200, cooldown_epochs=2):
        self.ex_top_k = initial_ex_top_k
        
        # 💡 [핵심] 절대값이 아닌 '비율'을 임계치로 사용합니다.
        # margin_drop_ratio = 0.95 : 이전 윈도우 평균 대비 마진이 5% 이상 하락하면 위험!
        self.margin_drop_ratio = margin_drop_ratio 
        
        # penalty_rise_ratio = 1.05 : 이전 윈도우 평균 대비 페널티 비율이 5% 이상 상승하면 위험!
        self.penalty_rise_ratio = penalty_rise_ratio 
        
        self.window_size = window_size
        self.step_size = step_size
        self.max_ex_top_k = max_ex_top_k
        self.cooldown_epochs = cooldown_epochs
        self.cooldown_counter = 0

        self.margin_history = []
        self.penalized_history = []

    def step(self, sim_pos, sim_hn, penalized_ratio):
        # 1. 현재 마진 계산 및 히스토리 저장
        current_margin = sim_pos - sim_hn
        self.margin_history.append(current_margin)
        self.penalized_history.append(penalized_ratio)

        # 윈도우 사이즈의 2배수(예: 6개)를 초과하는 오래된 기록은 삭제
        if len(self.margin_history) > self.window_size * 2:
            self.margin_history.pop(0)
            self.penalized_history.pop(0)

        # ⏳ 쿨다운(휴식기) 체크
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, self.ex_top_k, "Cooldown Active"

        # 데이터가 충분히(최소 6에포크 분량) 모이지 않았으면 대기
        if len(self.margin_history) < self.window_size * 2:
            return False, self.ex_top_k, "Gathering Data"

        # 2. 과거 윈도우(앞 3개) vs 최근 윈도우(뒤 3개) 평균 계산
        prev_margin_avg = sum(self.margin_history[:self.window_size]) / self.window_size
        recent_margin_avg = sum(self.margin_history[self.window_size:]) / self.window_size
        
        prev_penalized_avg = sum(self.penalized_history[:self.window_size]) / self.window_size
        recent_penalized_avg = sum(self.penalized_history[self.window_size:]) / self.window_size

        updated = False
        reason = ""

        # ---------------------------------------------------------
        # 🎯 Trigger 1: 마진이 과거 대비 상대적으로 축소되었는가?
        # ---------------------------------------------------------
        if recent_margin_avg < prev_margin_avg * self.margin_drop_ratio:
            updated = True
            reason = f"Margin Drop (Prev: {prev_margin_avg:.4f} -> Recent: {recent_margin_avg:.4f})"

        # ---------------------------------------------------------
        # 🎯 Trigger 2: 페널티 비율이 과거 대비 상승(U-Turn)했는가?
        # ---------------------------------------------------------
        elif recent_penalized_avg > prev_penalized_avg * self.penalty_rise_ratio:
            updated = True
            reason = f"Penalty U-Turn (Prev: {prev_penalized_avg:.4f} -> Recent: {recent_penalized_avg:.4f})"

        # 🚀 조건 충족 시 난이도 완화 (EX_TOP_K 확장)
        if updated and self.ex_top_k < self.max_ex_top_k:
            self.ex_top_k = min(self.ex_top_k + self.step_size, self.max_ex_top_k)
            self.cooldown_counter = self.cooldown_epochs # 확장 후 쿨다운 진입
            return True, self.ex_top_k, reason

        return False, self.ex_top_k, "Stable"
    
    
class BidirectionalHNScheduler:
    def __init__(self, initial_ex_top_k=50, 
                 margin_drop_ratio=0.95, penalty_rise_ratio=1.05,   # 🚨 응급 후퇴(난이도 하락) 기준
                 margin_growth_ratio=1.05, penalty_drop_ratio=0.95, # 📈 레벨업(난이도 상승) 기준
                 window_size=2, step_size=10,                       # 💡 추천 세팅 반영
                 min_ex_top_k=20, max_ex_top_k=100, cooldown_epochs=1):
        
        self.ex_top_k = initial_ex_top_k
        self.margin_drop_ratio = margin_drop_ratio
        self.penalty_rise_ratio = penalty_rise_ratio
        self.margin_growth_ratio = margin_growth_ratio
        self.penalty_drop_ratio = penalty_drop_ratio
        
        self.window_size = window_size
        self.step_size = step_size
        self.min_ex_top_k = min_ex_top_k
        self.max_ex_top_k = max_ex_top_k
        self.cooldown_epochs = cooldown_epochs
        self.cooldown_counter = 0

        self.margin_history = []
        self.penalized_history = []

    def step(self, sim_pos, sim_hn, penalized_ratio):
        current_margin = sim_pos - sim_hn
        self.margin_history.append(current_margin)
        self.penalized_history.append(penalized_ratio)

        # 윈도우 사이즈의 2배수 초과 기록 삭제
        if len(self.margin_history) > self.window_size * 2:
            self.margin_history.pop(0)
            self.penalized_history.pop(0)

        # ⏳ 쿨다운 체크
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False, self.ex_top_k, "Cooldown Active"

        # 데이터 수집 대기
        if len(self.margin_history) < self.window_size * 2:
            return False, self.ex_top_k, "Gathering Data"

        # 과거 윈도우 vs 최근 윈도우 평균 계산
        prev_margin_avg = sum(self.margin_history[:self.window_size]) / self.window_size
        recent_margin_avg = sum(self.margin_history[self.window_size:]) / self.window_size
        
        prev_penalized_avg = sum(self.penalized_history[:self.window_size]) / self.window_size
        recent_penalized_avg = sum(self.penalized_history[self.window_size:]) / self.window_size

        updated = False
        reason = ""

        # =========================================================
        # 📈 Trigger A: 레벨업 (Level Up - 난이도 상승)
        # 조건: 마진이 늘어나고(여유) AND 페널티 비율이 떨어질 때(정확도 상승)
        # =========================================================
        if (recent_margin_avg > prev_margin_avg * self.margin_growth_ratio) and \
           (recent_penalized_avg < prev_penalized_avg * self.penalty_drop_ratio):
            
            if self.ex_top_k > self.min_ex_top_k:
                self.ex_top_k = max(self.ex_top_k - self.step_size, self.min_ex_top_k)
                updated = True
                reason = f"Level Up! Margin ⬆️ & Penalty ⬇️ (New K: {self.ex_top_k})"

        # =========================================================
        # 🚨 Trigger B: 응급 후퇴 (Emergency - 난이도 하락)
        # 조건: 마진이 좁아지거나(붕괴) OR 페널티 비율이 반등할 때(한계 도달)
        # =========================================================
        elif (recent_margin_avg < prev_margin_avg * self.margin_drop_ratio) or \
             (recent_penalized_avg > prev_penalized_avg * self.penalty_rise_ratio):
            
            if self.ex_top_k < self.max_ex_top_k:
                self.ex_top_k = min(self.ex_top_k + self.step_size, self.max_ex_top_k)
                updated = True
                reason = f"Emergency! Margin ⬇️ or Penalty ⬆️ (New K: {self.ex_top_k})"

        # 액션이 취해졌다면 쿨다운 진입
        if updated:
            self.cooldown_counter = self.cooldown_epochs
            return True, self.ex_top_k, reason

        return False, self.ex_top_k, "Stable"