"""
A²-BMS 학습 데이터 생성기 (소전류 + 대전류 통합)
==================================================
방안 B: 환경별 적응형 AI를 위한 두 가지 데이터셋 동시 생성

[소전류 데이터셋] - 학부 환경 (0.1~0.5A)
  - bms_pack_data_low.csv  (분류기)
  - bms_cell_data_low.csv  (회귀기)
  - 임계값: MOSFET 50°C, 배터리 45°C, ΔV 0.02~0.08V

[대전류 데이터셋] - 가상 환경 (10~30A 시뮬)
  - bms_pack_data_high.csv (분류기)
  - bms_cell_data_high.csv (회귀기)
  - 임계값: MOSFET 80°C, 배터리 50°C, ΔV 0.02~0.20V
  - 발열·전류 크게 → 더 보수적인 derating

확률적 라벨링: 시그모이드 기반 부드러운 모드 전환
Thermal Throttling: 온도 높을 때 출력값 자동 감쇄
"""
import csv
import random
import math

# ============================================================
# 공통 설정
# ============================================================
NUM_PACK_SAMPLES = 6000
NUM_CELL_SAMPLES = 8000


def sigmoid(x, center, sharpness=50):
    """경계 근처에서 부드럽게 전환되는 확률 함수"""
    return 1.0 / (1.0 + math.exp(-sharpness * (x - center)))


# ============================================================
# [소전류 환경] 임계값 및 데이터 분포
# ============================================================
class LowCurrentParams:
    """학부 실험 환경 (0.1~0.5A) - 기존 시스템"""
    # 임계값
    MOSFET_T_HIGH = 50.0
    MOSFET_T_STOP = 70.0
    BATTERY_T_HIGH = 45.0
    BATTERY_T_STOP = 55.0
    DV_LOW = 0.02
    DV_HIGH = 0.08
    
    # 데이터 분포
    MAX_MOSFET_T_RANGE = (25, 65)
    MAX_BATTERY_T_RANGE = (25, 50)
    PACK_DV_RANGE = (0, 0.15)
    
    CELL_MOSFET_T_RANGE = (25, 70)
    CELL_BATTERY_T_RANGE = (25, 55)
    CELL_V_RANGE = (3.0, 4.2)
    DELTA_RANGE = (0, 0.30)
    
    # Derating 계수
    DAC_MOSFET_DERATE = 0.04
    DAC_BATTERY_DERATE = 0.06
    PWM_MOSFET_DERATE = 0.025
    PWM_BATTERY_DERATE = 0.04
    
    # 파일명
    PACK_FILE = "bms_pack_data_low.csv"
    CELL_FILE = "bms_cell_data_low.csv"


# ============================================================
# [대전류 환경] 임계값 및 데이터 분포
# ============================================================
class HighCurrentParams:
    """가상 대전류 환경 (10~30A) - HILS 검증용"""
    # 임계값 (대전류 환경에선 발열 큼 → 더 높은 임계값)
    MOSFET_T_HIGH = 80.0
    MOSFET_T_STOP = 100.0
    BATTERY_T_HIGH = 50.0
    BATTERY_T_STOP = 60.0
    DV_LOW = 0.02
    DV_HIGH = 0.10  # 대전류는 더 큰 편차도 정상
    
    # 데이터 분포 (대전류 환경 - 발열·전압강하 큼)
    MAX_MOSFET_T_RANGE = (25, 110)   # 무방열판 시뮬 가능
    MAX_BATTERY_T_RANGE = (25, 65)
    PACK_DV_RANGE = (0, 0.25)        # IR drop으로 인한 큰 편차
    
    CELL_MOSFET_T_RANGE = (25, 120)
    CELL_BATTERY_T_RANGE = (25, 70)
    CELL_V_RANGE = (2.8, 4.2)        # 대전류 IR drop으로 낮은 전압도
    DELTA_RANGE = (0, 0.40)
    
    # Derating 계수 (대전류는 더 강하게 깎아야 안전)
    DAC_MOSFET_DERATE = 0.025         # 80°C부터 시작이라 더 완만하게
    DAC_BATTERY_DERATE = 0.05
    PWM_MOSFET_DERATE = 0.015
    PWM_BATTERY_DERATE = 0.03
    
    # 파일명
    PACK_FILE = "bms_pack_data_high.csv"
    CELL_FILE = "bms_cell_data_high.csv"


# ============================================================
# 팩 단위 데이터 생성 (분류기용)
# ============================================================
def get_pack_dac_probability(max_m_t, max_b_t, pack_dv, P):
    """팩 상태 -> DAC 모드일 확률
    P: LowCurrentParams or HighCurrentParams
    """
    # MOSFET 온도 기반 (임계 근처에서 전환)
    p_mosfet = sigmoid(max_m_t, P.MOSFET_T_HIGH, sharpness=0.6)
    # 배터리 온도 기반
    p_battery = sigmoid(max_b_t, P.BATTERY_T_HIGH, sharpness=0.6)
    # 편차 기반 (작을수록 DAC)
    center_dv = (P.DV_LOW + P.DV_HIGH) / 2
    p_dv = 1.0 - sigmoid(pack_dv, center_dv, sharpness=120)

    # 어느 하나라도 DAC 강하게 원하면 DAC (max 결합)
    p_dac = max(p_mosfet, p_battery, p_dv)
    p_dac += random.uniform(-0.05, 0.05)
    return max(0.02, min(0.98, p_dac))


def generate_pack_sample(P):
    max_m_t = round(random.uniform(*P.MAX_MOSFET_T_RANGE), 1)
    max_b_t = round(random.uniform(*P.MAX_BATTERY_T_RANGE), 1)
    pack_dv = round(random.uniform(*P.PACK_DV_RANGE), 4)

    p_dac = get_pack_dac_probability(max_m_t, max_b_t, pack_dv, P)
    label = "DAC" if random.random() < p_dac else "PWM"
    return max_m_t, max_b_t, pack_dv, label, round(p_dac, 3)


def make_pack_csv(P, env_name):
    """P: 파라미터 클래스, env_name: '소전류' 또는 '대전류'"""
    dac_list, pwm_list = [], []
    target = NUM_PACK_SAMPLES // 2
    attempts = 0
    while (len(dac_list) < target or len(pwm_list) < target) and attempts < NUM_PACK_SAMPLES * 10:
        s = generate_pack_sample(P)
        if s[3] == "DAC" and len(dac_list) < target:
            dac_list.append(s)
        elif s[3] == "PWM" and len(pwm_list) < target:
            pwm_list.append(s)
        attempts += 1

    all_data = dac_list + pwm_list
    random.shuffle(all_data)
    with open(P.PACK_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Max_Mosfet_T", "Max_Battery_T", "Pack_Delta_V", "Label", "P_DAC"])
        w.writerows(all_data)
    print(f"  [{env_name} PACK] {len(all_data)}개 (PWM:{len(pwm_list)} / DAC:{len(dac_list)}) -> {P.PACK_FILE}")


# ============================================================
# 셀 단위 데이터 생성 (회귀기용)
# ============================================================
def get_dac_value(cell_m_t, cell_b_t, cell_v, delta, P):
    """DAC 출력값 (0~4095)"""
    val = (delta / 0.15) * 4095

    # MOSFET 온도 derating
    if cell_m_t >= P.MOSFET_T_HIGH:
        penalty = 1.0 - (cell_m_t - P.MOSFET_T_HIGH) * P.DAC_MOSFET_DERATE
        val *= max(0.1, penalty)

    # 배터리 온도 derating
    if cell_b_t >= P.BATTERY_T_HIGH:
        penalty = 1.0 - (cell_b_t - P.BATTERY_T_HIGH) * P.DAC_BATTERY_DERATE
        val *= max(0.1, penalty)

    # 셀 위험 영역 (3.3V 미만)
    if cell_v < 3.3:
        val *= 0.4

    val += random.uniform(-50, 50)
    return int(min(4095, max(0, val)))


def get_pwm_duty(cell_m_t, cell_b_t, cell_v, delta, P):
    """PWM 듀티비 (0~255)"""
    val = (delta / 0.15) * 255

    if cell_m_t >= P.MOSFET_T_HIGH:
        penalty = 1.0 - (cell_m_t - P.MOSFET_T_HIGH) * P.PWM_MOSFET_DERATE
        val *= max(0.2, penalty)

    if cell_b_t >= P.BATTERY_T_HIGH:
        penalty = 1.0 - (cell_b_t - P.BATTERY_T_HIGH) * P.PWM_BATTERY_DERATE
        val *= max(0.2, penalty)

    if cell_v < 3.3:
        val *= 0.5

    val += random.uniform(-5, 5)
    return int(min(255, max(0, val)))


def generate_cell_sample(P):
    cell_m_t = round(random.uniform(*P.CELL_MOSFET_T_RANGE), 1)
    cell_b_t = round(random.uniform(*P.CELL_BATTERY_T_RANGE), 1)
    cell_v = round(random.uniform(*P.CELL_V_RANGE), 3)
    delta = round(random.uniform(*P.DELTA_RANGE), 4)

    dac_val = get_dac_value(cell_m_t, cell_b_t, cell_v, delta, P)
    pwm_duty = get_pwm_duty(cell_m_t, cell_b_t, cell_v, delta, P)

    return cell_m_t, cell_b_t, cell_v, delta, dac_val, pwm_duty


def make_cell_csv(P, env_name):
    samples = [generate_cell_sample(P) for _ in range(NUM_CELL_SAMPLES)]
    with open(P.CELL_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Mosfet_T", "Battery_T", "Cell_V", "Delta_From_Min", "DAC_Val", "PWM_Duty"])
        w.writerows(samples)
    print(f"  [{env_name} CELL] {len(samples)}개 -> {P.CELL_FILE}")


# ============================================================
# 실행
# ============================================================
def main():
    print("=" * 60)
    print("A²-BMS 학습 데이터 생성 (소전류 + 대전류 통합)")
    print("=" * 60)
    
    print("\n[1/2] 소전류 환경 데이터 생성 (학부 실험)")
    print("-" * 60)
    make_pack_csv(LowCurrentParams, "소전류")
    make_cell_csv(LowCurrentParams, "소전류")
    
    print("\n[2/2] 대전류 환경 데이터 생성 (HILS 가상)")
    print("-" * 60)
    make_pack_csv(HighCurrentParams, "대전류")
    make_cell_csv(HighCurrentParams, "대전류")
    
    print("\n" + "=" * 60)
    print("생성 완료. 다음 단계: python train_system.py")
    print("=" * 60)
    
    # 미리보기
    print("\n[소전류 PACK 샘플 3개]")
    with open(LowCurrentParams.PACK_FILE) as f:
        for i, line in enumerate(f):
            if i > 3: break
            print("  " + line.strip())
    
    print("\n[대전류 PACK 샘플 3개]")
    with open(HighCurrentParams.PACK_FILE) as f:
        for i, line in enumerate(f):
            if i > 3: break
            print("  " + line.strip())


if __name__ == "__main__":
    main()
