"""
A²-BMS 학습 데이터 생성기 (인과 관계 반영판)
[변경] 변수 간 물리적 인과 관계 반영
- 80% 인과 데이터 + 20% 독립 random (overfitting 방지)
[인과 사슬] ΔV → MOSFET T → 배터리 T (열전도), 셀 V↓ → 발열↑
"""
import csv, random, math

NUM_PACK_SAMPLES = 6000
NUM_CELL_SAMPLES = 8000
INDEPENDENT_RATIO = 0.20
AMBIENT_T = 25.0

def sigmoid(x, center, sharpness=50):
    return 1.0 / (1.0 + math.exp(-sharpness * (x - center)))


class LowCurrentParams:
    MOSFET_T_HIGH = 50.0; MOSFET_T_STOP = 70.0
    BATTERY_T_HIGH = 45.0; BATTERY_T_STOP = 55.0
    DV_LOW = 0.02; DV_HIGH = 0.08
    MAX_MOSFET_T_RANGE = (25, 65); MAX_BATTERY_T_RANGE = (25, 50)
    PACK_DV_RANGE = (0, 0.15)
    CELL_MOSFET_T_RANGE = (25, 70); CELL_BATTERY_T_RANGE = (25, 55)
    CELL_V_RANGE = (3.0, 4.2); DELTA_RANGE = (0, 0.30)
    MOSFET_HEAT_COEF_PACK = (150, 280)
    MOSFET_HEAT_COEF_CELL = (100, 220)
    HEAT_CONDUCTION_RATIO = (0.30, 0.45)
    LOW_V_PENALTY = (2, 6)
    DAC_MOSFET_DERATE = 0.04; DAC_BATTERY_DERATE = 0.06
    PWM_MOSFET_DERATE = 0.025; PWM_BATTERY_DERATE = 0.04
    PACK_FILE = "bms_pack_data_low.csv"
    CELL_FILE = "bms_cell_data_low.csv"


class HighCurrentParams:
    MOSFET_T_HIGH = 80.0; MOSFET_T_STOP = 100.0
    BATTERY_T_HIGH = 50.0; BATTERY_T_STOP = 60.0
    DV_LOW = 0.02; DV_HIGH = 0.10
    MAX_MOSFET_T_RANGE = (25, 110); MAX_BATTERY_T_RANGE = (25, 65)
    PACK_DV_RANGE = (0, 0.25)
    CELL_MOSFET_T_RANGE = (25, 120); CELL_BATTERY_T_RANGE = (25, 70)
    CELL_V_RANGE = (2.8, 4.2); DELTA_RANGE = (0, 0.40)
    MOSFET_HEAT_COEF_PACK = (400, 700)
    MOSFET_HEAT_COEF_CELL = (350, 600)
    HEAT_CONDUCTION_RATIO = (0.25, 0.40)
    LOW_V_PENALTY = (5, 12)
    DAC_MOSFET_DERATE = 0.025; DAC_BATTERY_DERATE = 0.05
    PWM_MOSFET_DERATE = 0.015; PWM_BATTERY_DERATE = 0.03
    PACK_FILE = "bms_pack_data_high.csv"
    CELL_FILE = "bms_cell_data_high.csv"


def get_pack_dac_probability(max_m_t, max_b_t, pack_dv, P):
    p_mosfet = sigmoid(max_m_t, P.MOSFET_T_HIGH, sharpness=0.6)
    p_battery = sigmoid(max_b_t, P.BATTERY_T_HIGH, sharpness=0.6)
    center_dv = (P.DV_LOW + P.DV_HIGH) / 2
    p_dv = 1.0 - sigmoid(pack_dv, center_dv, sharpness=120)
    p_dac = max(p_mosfet, p_battery, p_dv)
    p_dac += random.uniform(-0.05, 0.05)
    return max(0.02, min(0.98, p_dac))


def get_dac_value(cell_m_t, cell_b_t, cell_v, delta, P):
    val = (delta / 0.15) * 4095
    if cell_m_t >= P.MOSFET_T_HIGH:
        val *= max(0.1, 1.0 - (cell_m_t - P.MOSFET_T_HIGH) * P.DAC_MOSFET_DERATE)
    if cell_b_t >= P.BATTERY_T_HIGH:
        val *= max(0.1, 1.0 - (cell_b_t - P.BATTERY_T_HIGH) * P.DAC_BATTERY_DERATE)
    if cell_v < 3.3:
        val *= 0.4
    val += random.uniform(-50, 50)
    return int(min(4095, max(0, val)))


def get_pwm_duty(cell_m_t, cell_b_t, cell_v, delta, P):
    val = (delta / 0.15) * 255
    if cell_m_t >= P.MOSFET_T_HIGH:
        val *= max(0.2, 1.0 - (cell_m_t - P.MOSFET_T_HIGH) * P.PWM_MOSFET_DERATE)
    if cell_b_t >= P.BATTERY_T_HIGH:
        val *= max(0.2, 1.0 - (cell_b_t - P.BATTERY_T_HIGH) * P.PWM_BATTERY_DERATE)
    if cell_v < 3.3:
        val *= 0.5
    val += random.uniform(-5, 5)
    return int(min(255, max(0, val)))


# ============================================================
# 인과 기반 (80%)
# ============================================================
def generate_pack_sample_causal(P):
    pack_dv = round(random.uniform(*P.PACK_DV_RANGE), 4)
    heat = random.uniform(*P.MOSFET_HEAT_COEF_PACK)
    max_m_t = AMBIENT_T + pack_dv * heat + random.uniform(-5, 5)
    max_m_t = max(AMBIENT_T, min(P.MAX_MOSFET_T_RANGE[1], max_m_t))
    cond = random.uniform(*P.HEAT_CONDUCTION_RATIO)
    max_b_t = AMBIENT_T + (max_m_t - AMBIENT_T) * cond + random.uniform(-2, 2)
    max_b_t = max(AMBIENT_T, min(P.MAX_BATTERY_T_RANGE[1], max_b_t))
    p_dac = get_pack_dac_probability(max_m_t, max_b_t, pack_dv, P)
    label = "DAC" if random.random() < p_dac else "PWM"
    return round(max_m_t, 1), round(max_b_t, 1), pack_dv, label, round(p_dac, 3)


def generate_cell_sample_causal(P):
    delta = round(random.uniform(*P.DELTA_RANGE), 4)
    # 셀 V (방전 진행도 반영)
    base_v = random.uniform(3.5, 4.2)
    cell_v = base_v - delta * random.uniform(0.3, 0.7)
    cell_v = round(max(P.CELL_V_RANGE[0], min(P.CELL_V_RANGE[1], cell_v)), 3)
    # MOSFET T (ΔV 누적 발열)
    heat = random.uniform(*P.MOSFET_HEAT_COEF_CELL)
    cell_m_t = AMBIENT_T + delta * heat + random.uniform(-3, 3)
    # [2순위] 셀 V 낮으면 내부저항 ↑ → 추가 발열
    if cell_v < 3.3:
        cell_m_t += random.uniform(*P.LOW_V_PENALTY)
    cell_m_t = max(AMBIENT_T, min(P.CELL_MOSFET_T_RANGE[1], cell_m_t))
    cell_m_t = round(cell_m_t, 1)
    # 배터리 T (열전도)
    cond = random.uniform(*P.HEAT_CONDUCTION_RATIO)
    cell_b_t = AMBIENT_T + (cell_m_t - AMBIENT_T) * cond + random.uniform(-2, 2)
    cell_b_t = max(AMBIENT_T, min(P.CELL_BATTERY_T_RANGE[1], cell_b_t))
    cell_b_t = round(cell_b_t, 1)
    
    dac_val = get_dac_value(cell_m_t, cell_b_t, cell_v, delta, P)
    pwm_duty = get_pwm_duty(cell_m_t, cell_b_t, cell_v, delta, P)
    return cell_m_t, cell_b_t, cell_v, delta, dac_val, pwm_duty


# ============================================================
# 독립 random (20%)
# ============================================================
def generate_pack_sample_independent(P):
    max_m_t = round(random.uniform(*P.MAX_MOSFET_T_RANGE), 1)
    max_b_t = round(random.uniform(*P.MAX_BATTERY_T_RANGE), 1)
    pack_dv = round(random.uniform(*P.PACK_DV_RANGE), 4)
    p_dac = get_pack_dac_probability(max_m_t, max_b_t, pack_dv, P)
    label = "DAC" if random.random() < p_dac else "PWM"
    return max_m_t, max_b_t, pack_dv, label, round(p_dac, 3)


def generate_cell_sample_independent(P):
    cell_m_t = round(random.uniform(*P.CELL_MOSFET_T_RANGE), 1)
    cell_b_t = round(random.uniform(*P.CELL_BATTERY_T_RANGE), 1)
    cell_v = round(random.uniform(*P.CELL_V_RANGE), 3)
    delta = round(random.uniform(*P.DELTA_RANGE), 4)
    dac_val = get_dac_value(cell_m_t, cell_b_t, cell_v, delta, P)
    pwm_duty = get_pwm_duty(cell_m_t, cell_b_t, cell_v, delta, P)
    return cell_m_t, cell_b_t, cell_v, delta, dac_val, pwm_duty


def generate_pack_sample(P):
    if random.random() < INDEPENDENT_RATIO:
        return generate_pack_sample_independent(P)
    return generate_pack_sample_causal(P)


def generate_cell_sample(P):
    if random.random() < INDEPENDENT_RATIO:
        return generate_cell_sample_independent(P)
    return generate_cell_sample_causal(P)


def make_pack_csv(P, env_name):
    dac_list, pwm_list = [], []
    target = NUM_PACK_SAMPLES // 2
    attempts = 0
    while (len(dac_list) < target or len(pwm_list) < target) and attempts < NUM_PACK_SAMPLES * 15:
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


def make_cell_csv(P, env_name):
    samples = [generate_cell_sample(P) for _ in range(NUM_CELL_SAMPLES)]
    with open(P.CELL_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Mosfet_T", "Battery_T", "Cell_V", "Delta_From_Min", "DAC_Val", "PWM_Duty"])
        w.writerows(samples)
    print(f"  [{env_name} CELL] {len(samples)}개 -> {P.CELL_FILE}")


def main():
    print("=" * 60)
    print("A²-BMS 학습 데이터 생성 (인과 관계 반영)")
    print(f"  인과: {int((1-INDEPENDENT_RATIO)*100)}% / 독립: {int(INDEPENDENT_RATIO*100)}%")
    print("=" * 60)
    print("\n[1/2] 소전류 환경"); print("-"*60)
    make_pack_csv(LowCurrentParams, "소전류")
    make_cell_csv(LowCurrentParams, "소전류")
    print("\n[2/2] 대전류 환경"); print("-"*60)
    make_pack_csv(HighCurrentParams, "대전류")
    make_cell_csv(HighCurrentParams, "대전류")
    print("\n" + "="*60)
    print("완료. 다음: python train_system.py")
    print("="*60)


if __name__ == "__main__":
    main()
