"""
BMS 대전류 환경 HILS 시뮬레이션 서버
==================================================
[대전류 전용 버전 - High-Current HILS]

핵심 개념: HILS (Hardware-in-the-Loop Simulation)
- 실제 HW: 안전한 소전류 (0.1~0.5A) 운용
- Plant Model: 데이터시트 기반 물리식 (상수)으로 대전류 환경 계산
- Controller: AI는 가상 대전류 상태만 입력받아 판단

[원본 main_server.py와의 차이]
1. PhysicsConstants 클래스 추가 (데이터시트 기반 상수)
2. HighCurrentSimulator 클래스 추가 (Plant Model)
3. AI 입력 = 실측이 아닌 "스케일링된 가상 대전류 상태"
4. 대전류 전용 데이터 채널 추가 (i_scaled, v_drop, p_loss)
5. 별도 포트(8001)로 실행 (원본 서버와 독립)

[검증 정당성 - 발표 멘트]
"Plant Model(물리식)과 Controller(AI)의 분리로 순환 논증 회피,
 객관적 ground truth에 기반한 대전류 환경 AI 검증 프레임워크"

원본 호환: 동일한 .pkl 모델 사용, 동일한 RandomForest 추론
포트 분리: 원본 8000 / 대전류 8001 (동시 실행 가능)
"""
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import asyncio
import json
import random
import os
import csv
import time
from datetime import datetime
import joblib
import pandas as pd

# UART 통신용 (실제 모드에서만 사용, 시뮬 모드에서는 import 실패해도 OK)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial 미설치 - 시뮬 모드 전용 (실제 모드 시 'pip install pyserial' 필요)")

# ============================================================
# UART 설정 (실제 하드웨어 연결 시 수정)
# ============================================================
UART_PORT = '/dev/ttyUSB0'   # 라즈베리파이에 ESP32가 USB로 연결된 경우
                              # 또는 '/dev/ttyACM0' (ESP32 native USB)
                              # GPIO UART면 '/dev/serial0'
UART_BAUDRATE = 115200        # ESP32 표준 baud rate
UART_TIMEOUT = 0.5            # 수신 타임아웃 (초)

# ============================================================
# [1번 대응] 실측 데이터 자동 로깅
# ============================================================
# is_real_mode=True 일 때만 활성화 (시뮬 데이터는 안 쌓음)
# 매 tick의 입력+AI출력+결과(다음 tick 측정값)을 CSV로 저장
# 추후 train_system.py로 재학습 가능
LOGGING_ENABLED = True              # 로깅 on/off
LOG_DIR = 'logs'                    # 로그 폴더
LOG_FILE_PREFIX = 'real_data'       # 파일명 prefix
LOG_ROTATE_HOURS = 24               # N시간마다 새 파일

# ============================================================
# 설정
# ============================================================
PACK_FEATURES = ['Max_Mosfet_T', 'Max_Battery_T', 'Pack_Delta_V']
CELL_FEATURES = ['Mosfet_T', 'Battery_T', 'Cell_V', 'Delta_From_Min']

# 발열 모델 (현실 P = I × V_DS_drop 근사)
HEAT_DAC_MOSFET = 1.0    # 2.0 → 1.0 (선형 모델로 바꾸면서 계수 절반)
HEAT_PWM_MOSFET = 0.5    # 그대로 (PWM은 평균 전류 기반이라 변경 없음)

# 자연 방열 (실내 + 약방열판)
COOL_K_MOSFET = 0.014    # 0.020 → 0.014 (통풍 약함)
COOL_K_BATTERY = 0.004   # 0.006 → 0.004

# 안전 임계
MOSFET_T_STOP = 65.0     # 70 → 65 (Op-Amp 70°C 한계 보호)
BATTERY_T_STOP = 55.0

TEMP_AMBIENT = 28.0       
HEAT_BATTERY_CONDUCT = 0.02     
HEAT_BATTERY_INTERNAL = 0.1     

# ============================================================
# [4번 수정] PCB 열 결합 계수
# 이웃 셀 MOSFET이 뜨거우면 PCB 통해 이쪽도 따뜻해짐
# 0 = 완전 독립 (이전 가정), 1 = 완전 결합 (현실)
# 0.05 = 약한 결합 (현실적 PCB 가정)
# ============================================================
PCB_HEAT_COUPLING = 0.015   # 약한 결합 (현실적 PCB)

# ============================================================
# [5번 수정] 자가방전: 단조 감소 모델
# 셀마다 다른 속도 (개체 차이)
# 시간당 1~3mV 떨어짐 (실제 18650 자가방전률)
# ============================================================
# 셀별 자가방전 속도 (V/sec), 한 번만 초기화
SELF_DISCHARGE_RATES = [
    random.uniform(0.0000003, 0.0000008)  # 약 1~3 mV/hour
    for _ in range(4)
]
# 측정 노이즈 (자가방전과 분리)
MEASUREMENT_NOISE = 0.0005  # ±0.5mV 측정 노이즈

# ============================================================
# [대전류 HILS] 물리 상수 (데이터시트 기반)
# 이 값들은 학계/산업계 검증된 객관적 ground truth
# AI 학습 대상이 아닌 "Plant Model"의 결정론적 파라미터
# ============================================================
class PhysicsConstants:
    """데이터시트 + 학계 검증 상수 - AI 학습 대상 아님"""
    # ===== 18650 셀 =====
    R_CELL_INTERNAL = 0.05       # 18650 NMC 내부저항 (Ω) - 산업 표준
    C_CELL_THERMAL = 800.0       # 셀 열용량 (J/°C) - 셀 무게 × 비열
    R_THETA_CELL = 8.0           # 셀 열저항 (°C/W) - 자연방열
    
    # ===== IRLZ44N MOSFET =====
    R_DS_ON = 0.022              # PWM 모드 ON 저항 (Ω) - 데이터시트
    V_DS_SAT = 2.5               # DAC 모드 V_DS drop (V) - saturation 동작점
    R_THETA_MOSFET = 62.0        # TO-220 무방열판 (°C/W) - 데이터시트
    R_THETA_MOSFET_HEATSINK = 25.0  # TO-220 + 소형 방열판 (°C/W)
    C_MOSFET_THERMAL = 5.0       # MOSFET 열용량 (J/°C)
    T_J_MAX = 175.0              # IRLZ44N junction 최대 (°C)
    
    # ===== 회로 부품 =====
    R_BAL = 10.0                 # 밸런싱 저항 (Ω)
    R_SENSE = 0.1                # 전류 측정 저항 (Ω)
    
    # ===== 안전 임계 (대전류 환경) =====
    T_MOSFET_DERATE = 60.0       # 이 온도부터 출력 derating
    T_MOSFET_STOP = 80.0         # 즉시 STOP (대전류 환경에선 65→80 마진 확대)
    T_CELL_STOP = 60.0           # 셀 STOP 임계 (대전류 시 셀 자체 발열 큼)


# ============================================================
# [대전류 HILS] Plant Model
# 실측 소전류 → 가상 대전류 환경 시뮬레이션
# 모든 계산은 물리식 (AI 절대 사용 안 함)
# ============================================================
class HighCurrentSimulator:
    """
    Plant Model: 실제 0.5A 측정값을 50A 가상 환경으로 스케일링
    
    동작 원리:
    1. ESP32에서 실측 전류/전압/온도 수신
    2. 스케일 팩터 적용 → 가상 대전류 상태 계산 (물리식)
    3. AI는 "가상 상태"만 입력받아 판단
    4. AI 출력은 다시 ÷SCALE 해서 실제 회로엔 안전한 소전류로 적용
    """
    def __init__(self, scale_factor=100):
        self.SCALE = scale_factor
        self.C = PhysicsConstants()
        # 가상 환경 상태 (누적)
        self.virtual_mosfet_t = [28.0] * 4
        self.virtual_cell_t = [28.0] * 4
    
    def reset(self):
        self.virtual_mosfet_t = [28.0] * 4
        self.virtual_cell_t = [28.0] * 4
    
    def scale_current(self, i_real_list):
        """실측 전류 → 가상 대전류"""
        return [i * self.SCALE for i in i_real_list]
    
    def virtual_cell_voltage(self, v_open_list, i_scaled_list):
        """대전류 시 IR drop 적용 (옴의 법칙)
        대전류 인입 시 셀 단자전압이 내부저항 만큼 강하"""
        return [v - i * self.C.R_CELL_INTERNAL 
                for v, i in zip(v_open_list, i_scaled_list)]
    
    def mosfet_power_loss(self, i_scaled, mode):
        """MOSFET 발열 (W) - 모드별 물리식"""
        if mode == "DAC":
            # Linear: P = V_DS × I (saturation 영역, V_DS 거의 일정)
            return self.C.V_DS_SAT * i_scaled
        elif mode == "PWM":
            # Switching: P = I² × R_DS(on) (triode 영역, ON일 때만)
            # PWM duty 평균 고려 (스위칭 손실은 무시)
            return i_scaled ** 2 * self.C.R_DS_ON
        else:
            return 0.0
    
    def cell_power_loss(self, i_scaled):
        """셀 내부 발열 (W) - 줄 가열 P = I²R"""
        return i_scaled ** 2 * self.C.R_CELL_INTERNAL
    
    def update_virtual_temperatures(self, i_scaled_list, mode, dt=1.0, t_amb=28.0,
                                     with_heatsink=True):
        """가상 대전류 환경 온도 업데이트 (뉴턴 냉각 법칙)
        with_heatsink=True : 방열판 사용 (현실적 EV/ESS 환경)
        반환: (mosfet_t_list, cell_t_list, p_total_list)"""
        # 방열판 사용 시 열저항 감소
        r_theta_mosfet = (self.C.R_THETA_MOSFET_HEATSINK if with_heatsink 
                         else self.C.R_THETA_MOSFET)
        p_totals = []
        for i in range(4):
            i_s = i_scaled_list[i]
            
            # MOSFET 발열
            p_mosfet = self.mosfet_power_loss(i_s, mode)
            # MOSFET 온도 변화 (뉴턴 냉각)
            dt_mosfet = (p_mosfet * dt) / self.C.C_MOSFET_THERMAL
            dt_cool = (self.virtual_mosfet_t[i] - t_amb) * dt / (r_theta_mosfet * self.C.C_MOSFET_THERMAL)
            self.virtual_mosfet_t[i] += dt_mosfet - dt_cool
            self.virtual_mosfet_t[i] = max(t_amb, self.virtual_mosfet_t[i])
            
            # 셀 발열 (내부저항)
            p_cell = self.cell_power_loss(i_s)
            dt_cell = (p_cell * dt) / self.C.C_CELL_THERMAL
            dt_cell_cool = (self.virtual_cell_t[i] - t_amb) * dt / (self.C.R_THETA_CELL * self.C.C_CELL_THERMAL)
            self.virtual_cell_t[i] += dt_cell - dt_cell_cool
            self.virtual_cell_t[i] = max(t_amb, self.virtual_cell_t[i])
            
            p_totals.append(p_mosfet + p_cell)
        
        return list(self.virtual_mosfet_t), list(self.virtual_cell_t), p_totals
    
    def is_critical(self, t_mosfet_max, t_cell_max):
        """대전류 환경 위험 판단"""
        if t_mosfet_max >= self.C.T_MOSFET_STOP:
            return True, f"MOSFET 임계 ({t_mosfet_max:.1f}°C ≥ {self.C.T_MOSFET_STOP}°C)"
        if t_cell_max >= self.C.T_CELL_STOP:
            return True, f"셀 임계 ({t_cell_max:.1f}°C ≥ {self.C.T_CELL_STOP}°C)"
        return False, ""
    
    def derating_factor(self, t_mosfet):
        """온도에 따른 출력 감소 계수 (1.0=풀파워, 0.0=정지)
        대전류 환경의 thermal-aware 제어"""
        if t_mosfet < self.C.T_MOSFET_DERATE:
            return 1.0
        if t_mosfet >= self.C.T_MOSFET_STOP:
            return 0.0
        # 60~80°C 사이 선형 감소
        return 1.0 - (t_mosfet - self.C.T_MOSFET_DERATE) / (self.C.T_MOSFET_STOP - self.C.T_MOSFET_DERATE)


# 글로벌 인스턴스 (서버 시작 시 1회 생성)
# 
# Scale 결정 근거:
# - 실측 0.5A × 10 = 가상 5A (전기차 1C 충전 정도 - 현실적)
# - 실측 0.5A × 50 = 가상 25A (급속충전 환경)
# - 실측 0.5A × 100 = 가상 50A (초급속 - MOSFET 폭발 가능)
#
# 학부 발표용 권장: ×10 ~ ×20 (현실적, 안전 마진 검증 가능)
hc_sim = HighCurrentSimulator(scale_factor=20)  # 0.5A 실측 → 10A 가상 (1C 급속충전급)

app = FastAPI()

# ============================================================
# AI 모델 3개 로드
# ============================================================
try:
    clf = joblib.load('bms_mode_ai.pkl')
    reg_dac = joblib.load('bms_dac_ai.pkl')
    reg_pwm = joblib.load('bms_pwm_ai.pkl')
    AI_LOADED = True
    print("[AI] 모델 3개 로드 완료")
except Exception as e:
    clf = reg_dac = reg_pwm = None
    AI_LOADED = False
    print(f"[WARN] 모델 파일 없음: {e}")

# ============================================================
# UART 통신 (ESP32 ↔ 라즈베리파이)
# ============================================================
uart_conn = None  # 시리얼 객체 (지연 초기화)


def uart_connect():
    """
    UART 포트 연결. 실패 시 None 반환 (시뮬 모드로 fallback).
    실제 모드 첫 진입 시 호출.
    """
    global uart_conn
    if not SERIAL_AVAILABLE:
        return None
    if uart_conn is not None and uart_conn.is_open:
        return uart_conn
    try:
        uart_conn = serial.Serial(UART_PORT, UART_BAUDRATE, timeout=UART_TIMEOUT)
        print(f"[UART] 연결 성공: {UART_PORT} @ {UART_BAUDRATE}")
        return uart_conn
    except Exception as e:
        print(f"[UART] 연결 실패: {e}")
        uart_conn = None
        return None


def uart_read_sensor():
    """
    ESP32에서 센서 데이터 1줄 읽기 (JSON 한 줄).
    형식: {"v":[...], "mt":[...], "bt":[...], "i":[...]}
    실패 시 None 반환.
    """
    if uart_conn is None or not uart_conn.is_open:
        return None
    try:
        # in_waiting > 0이면 데이터 있음. 없으면 기다리지 않고 None 반환 (non-blocking)
        if uart_conn.in_waiting == 0:
            return None
        line = uart_conn.readline().decode('utf-8', errors='ignore').strip()
        if not line or not line.startswith('{'):
            return None
        return json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[UART] 데이터 파싱 실패 (무시): {e}")
        return None
    except Exception as e:
        print(f"[UART] 수신 에러: {e}")
        return None


def uart_send_command(mode, dac_vals, pwm_duty):
    """
    ESP32로 제어 명령 전송 (JSON 한 줄).
    형식: {"mode":"DAC", "dac":[1500,800,400,0], "pwm":[0,0,0,0]}
    """
    if uart_conn is None or not uart_conn.is_open:
        return False
    try:
        cmd = {
            "mode": mode,
            "dac": list(dac_vals),
            "pwm": list(pwm_duty),
        }
        msg = json.dumps(cmd) + "\n"
        uart_conn.write(msg.encode('utf-8'))
        uart_conn.flush()
        return True
    except Exception as e:
        print(f"[UART] 송신 에러: {e}")
        return False


# ============================================================
# 시뮬레이션 상태
# ============================================================
is_real_mode = False

sim_state = {
    "v": [4.20, 4.10, 4.05, 3.90],          
    "mosfet_t": [30.0, 30.0, 30.0, 30.0],   
    "battery_t": [30.0, 30.0, 30.0, 30.0],  
    "i": [0.0, 0.0, 0.0, 0.0],              

    "dac_vals": [0, 0, 0, 0],     
    "pwm_duty": [0, 0, 0, 0],     

    "pwm_v": [4.20, 4.10, 4.05, 3.90],
    "pwm_i": [0.0, 0.0, 0.0, 0.0],
    "pwm_mosfet_t": [30.0, 30.0, 30.0, 30.0],
    "pwm_battery_t": [30.0, 30.0, 30.0, 30.0],

    "dac_v": [4.20, 4.10, 4.05, 3.90],
    "dac_i": [0.0, 0.0, 0.0, 0.0],
    "dac_mosfet_t": [30.0, 30.0, 30.0, 30.0],
    "dac_battery_t": [30.0, 30.0, 30.0, 30.0],
}

# ============================================================
# [1번 대응] 실측 데이터 로깅 시스템
# ============================================================
# 매 tick 데이터를 CSV로 저장 → 실측 데이터 축적 → 재학습
# 헤더: 시간 + 입력(V/MT/BT) + AI출력(mode/DAC/PWM) + 결과지표(다음 tick ΔV 변화량)
LOG_HEADER = [
    'timestamp', 'tick',
    # 입력 (현재 상태)
    'v1', 'v2', 'v3', 'v4',
    'mt1', 'mt2', 'mt3', 'mt4',
    'bt1', 'bt2', 'bt3', 'bt4',
    'pack_delta_v',
    # AI 출력 (현재 결정)
    'ai_mode', 'p_dac',
    'dac1', 'dac2', 'dac3', 'dac4',
    'pwm1', 'pwm2', 'pwm3', 'pwm4',
    # 결과 (직전 tick 대비 ΔV 변화 — 이게 진짜 "효과")
    'delta_v_change',     # 양수 = ΔV 증가 (나빠짐), 음수 = ΔV 감소 (좋아짐)
    'mt_max_change',      # MOSFET 최고 온도 변화
    'bt_max_change',      # 배터리 최고 온도 변화
    'data_source',        # 'real' or 'sim'
]

logger_state = {
    'file': None,
    'writer': None,
    'tick': 0,
    'start_time': None,
    'prev_pdv': None,
    'prev_mt_max': None,
    'prev_bt_max': None,
}

def init_logger():
    """로그 폴더/파일 생성"""
    if not LOGGING_ENABLED:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{LOG_DIR}/{LOG_FILE_PREFIX}_{timestamp}.csv"
    logger_state['file'] = open(filename, 'w', newline='', encoding='utf-8')
    logger_state['writer'] = csv.writer(logger_state['file'])
    logger_state['writer'].writerow(LOG_HEADER)
    logger_state['file'].flush()
    logger_state['start_time'] = time.time()
    print(f"[LOG] 로깅 시작 → {filename}")

def log_tick(state, mode, p_dac, dac_vals, pwm_duty, is_real):
    """매 tick 데이터 1줄 기록"""
    if not LOGGING_ENABLED or logger_state['writer'] is None:
        return
    
    pdv = max(state['v']) - min(state['v'])
    mt_max = max(state['mosfet_t'])
    bt_max = max(state['battery_t'])
    
    # 이전 tick과 비교 (결과 지표)
    dv_change = (pdv - logger_state['prev_pdv']) if logger_state['prev_pdv'] is not None else 0
    mt_change = (mt_max - logger_state['prev_mt_max']) if logger_state['prev_mt_max'] is not None else 0
    bt_change = (bt_max - logger_state['prev_bt_max']) if logger_state['prev_bt_max'] is not None else 0
    
    row = [
        datetime.now().isoformat(),
        logger_state['tick'],
        # 입력
        round(state['v'][0], 4), round(state['v'][1], 4), round(state['v'][2], 4), round(state['v'][3], 4),
        round(state['mosfet_t'][0], 2), round(state['mosfet_t'][1], 2), round(state['mosfet_t'][2], 2), round(state['mosfet_t'][3], 2),
        round(state['battery_t'][0], 2), round(state['battery_t'][1], 2), round(state['battery_t'][2], 2), round(state['battery_t'][3], 2),
        round(pdv, 4),
        # AI 출력
        mode, round(p_dac, 3),
        dac_vals[0], dac_vals[1], dac_vals[2], dac_vals[3],
        pwm_duty[0], pwm_duty[1], pwm_duty[2], pwm_duty[3],
        # 결과 변화
        round(dv_change, 4),
        round(mt_change, 2),
        round(bt_change, 2),
        'real' if is_real else 'sim',
    ]
    logger_state['writer'].writerow(row)
    
    # 매 10 tick마다 flush (저장 보장)
    if logger_state['tick'] % 10 == 0:
        logger_state['file'].flush()
    
    logger_state['tick'] += 1
    logger_state['prev_pdv'] = pdv
    logger_state['prev_mt_max'] = mt_max
    logger_state['prev_bt_max'] = bt_max
    
    # 24시간마다 새 파일 (rotation)
    if logger_state['start_time'] and (time.time() - logger_state['start_time']) > LOG_ROTATE_HOURS * 3600:
        logger_state['file'].close()
        init_logger()

def close_logger():
    """프로그램 종료 시 호출"""
    if logger_state['file']:
        logger_state['file'].close()
        print(f"[LOG] 로깅 종료, 총 {logger_state['tick']}개 tick 저장")

@app.get("/")
async def get_index():
    # 대전류 전용 페이지 (없으면 일반 index.html fallback)
    if os.path.exists("index_highcurrent.html"):
        return FileResponse("index_highcurrent.html")
    if not os.path.exists("index.html"):
        return {"error": "index.html or index_highcurrent.html not found"}
    return FileResponse("index.html")

@app.get("/api/physics_constants")
async def get_physics_constants():
    """발표용 - 사용된 물리 상수 공개 (검증 가능성 입증)"""
    c = PhysicsConstants()
    return {
        "scale_factor": hc_sim.SCALE,
        "cell": {
            "internal_resistance_ohm": c.R_CELL_INTERNAL,
            "thermal_capacity_J_per_C": c.C_CELL_THERMAL,
            "thermal_resistance_C_per_W": c.R_THETA_CELL,
        },
        "mosfet_IRLZ44N": {
            "R_DS_ON_ohm": c.R_DS_ON,
            "V_DS_sat_V": c.V_DS_SAT,
            "thermal_resistance_no_heatsink_C_per_W": c.R_THETA_MOSFET,
            "junction_T_max_C": c.T_J_MAX,
        },
        "safety_thresholds": {
            "T_mosfet_derate_C": c.T_MOSFET_DERATE,
            "T_mosfet_stop_C": c.T_MOSFET_STOP,
            "T_cell_stop_C": c.T_CELL_STOP,
        },
        "source": "데이터시트(IRLZ44N) + 산업 표준(18650 NMC)",
    }

# ============================================================
# AI 판단
# ============================================================
def ai_decide_mode(max_mosfet_t, max_battery_t, pack_delta_v):
    if max_mosfet_t >= MOSFET_T_STOP or max_battery_t >= BATTERY_T_STOP:
        return "STOP", 1.0
    # 2. 전압차 안전 체크 (추가: 0.01V 이하로 평탄화 완료 시 정지)
    if pack_delta_v <= 0.01:
        return "STOP", 1.0

    if AI_LOADED:
        feat = pd.DataFrame([[max_mosfet_t, max_battery_t, pack_delta_v]], columns=PACK_FEATURES)
        proba = clf.predict_proba(feat)[0]
        cls = clf.classes_
        p_dac = float(proba[list(cls).index('DAC')]) if 'DAC' in cls else 0.5
        mode = clf.predict(feat)[0]
        return mode, p_dac
    
    #AI 비로드시 비상용 판단 로직 (간단한 휴리스틱)
    else:
        if max_mosfet_t >= 50 or max_battery_t >= 45 or pack_delta_v <= 0.02:
            return "DAC", 0.9
        elif pack_delta_v >= 0.05:
            return "PWM", 0.1
        else:
            return "PWM", 0.3

def ai_decide_cell_outputs(mode, mosfet_ts, battery_ts, voltages):
    if mode == "STOP":
        return [0, 0, 0, 0], [0, 0, 0, 0]

    min_v = min(voltages)
    deltas = [v - min_v for v in voltages]

    batch = pd.DataFrame([
        [mosfet_ts[i], battery_ts[i], voltages[i], deltas[i]] for i in range(4)
    ], columns=CELL_FEATURES)

    if AI_LOADED:
        if mode == "DAC":
            dac_vals = reg_dac.predict(batch)
            dac_vals = [int(max(0, min(4095, v))) for v in dac_vals]
            pwm_duty = [0, 0, 0, 0]
        else:
            pwm_duty = reg_pwm.predict(batch)
            pwm_duty = [int(max(0, min(255, v))) for v in pwm_duty]
            dac_vals = [0, 0, 0, 0]

    # AI 비로드 시 비상용 코드
    else:
        if mode == "DAC":
            dac_vals = [int(min(4095, deltas[i] / 0.15 * 4095)) for i in range(4)]
            pwm_duty = [0, 0, 0, 0]
        else:
            pwm_duty = [int(min(255, deltas[i] / 0.15 * 255)) for i in range(4)]
            dac_vals = [0, 0, 0, 0]

    return dac_vals, pwm_duty

# ============================================================
# 셀 전압 변화 (물리법칙 강제 적용)
# [5번 수정] 자가방전 단조 감소 + 측정 노이즈 분리
# ============================================================
def discharge_cell(v, current, smoothness, cell_idx=0):
    """단일 셀 전압 변화 (상태 오염 방지)"""
    # 자가방전: 항상 일정한 단조 감소 (셀마다 고유 속도)
    self_discharge = -SELF_DISCHARGE_RATES[cell_idx]
    if current <= 0.001:
        true_v = v + self_discharge
        return min(v, true_v)
    
    decay = current * smoothness * 0.05
    true_v = v - decay + self_discharge
    return min(v, true_v) # 절대로 이전 전압(v)보다 커질 수 없음

def discharge_cells_balanced(cells, currents, smoothness):
    return [discharge_cell(cells[i], currents[i], smoothness, cell_idx=i) for i in range(len(cells))]

def balance_toward_min(cells, currents, smoothness):
    """가상 우주용 평탄화 (상태 오염 방지)"""
    min_v = min(cells)
    new_cells = []
    
    for i, v in enumerate(cells):
        # [5번] 단조 감소 자가방전
        self_discharge = -SELF_DISCHARGE_RATES[i]
        if currents[i] <= 0.001:
            true_v = v + self_discharge
            new_cells.append(min(v, true_v))
            continue
            
        diff = v - min_v
        decay = diff * currents[i] * smoothness * 0.004
        true_v = v - decay + self_discharge
        
        # 🌟 핵심 방어 로직: 진짜 전압은 절대로 상승할 수 없음 🌟
        new_cells.append(min(v, true_v))
        
    return new_cells

def add_measurement_noise(cells_v):
    """[5번] 표시용 측정 노이즈 (ADC 노이즈 + 양자화)
    실제 셀 전압엔 영향 없음, 표시값만 흔들림"""
    return [v + random.uniform(-MEASUREMENT_NOISE, MEASUREMENT_NOISE) for v in cells_v]

# ============================================================
# [4번 수정] PCB 열 결합 — 이웃 셀 MOSFET 발열이 PCB 통해 전파
# ============================================================
def apply_pcb_heat_coupling(temps):
    """4개 MOSFET이 같은 PCB에 있을 때 열 평형 보정
    각 셀 온도 = (1-k) × 자기 자신 + k × 이웃 평균"""
    if PCB_HEAT_COUPLING <= 0:
        return list(temps)
    
    coupled = []
    n = len(temps)
    for i in range(n):
        # 이웃 셀 (자기 자신 제외) 평균
        neighbors = [temps[j] for j in range(n) if j != i]
        neighbor_avg = sum(neighbors) / len(neighbors)
        # 가중 평균: 본인 (1-k) + 이웃 평균 k
        new_t = (1 - PCB_HEAT_COUPLING) * temps[i] + PCB_HEAT_COUPLING * neighbor_avg
        coupled.append(new_t)
    return coupled

# ============================================================
# WebSocket
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global is_real_mode

    async def listen_commands():
        global is_real_mode
        try:
            while True:
                msg = await websocket.receive_text()
                if json.loads(msg).get("command") == "toggle_mode":
                    is_real_mode = not is_real_mode
                    print(f"[INFO] 시스템 모드: {'REAL' if is_real_mode else 'SIM'}")
        except Exception:
            pass

    asyncio.create_task(listen_commands())
    pwm_phase = True
    prev_mode = "PWM"
    
    # [1번 대응] 로거 초기화 (서버 시작 시 1회)
    if logger_state['file'] is None:
        init_logger()

    try:
        prev_real_mode = is_real_mode

        #실제 모드 데이터 수신 받아올 코드
        while True:

            #모드 토클 시 시뮬레이션 재동기화
            if is_real_mode != prev_real_mode:
                print(f"[INFO] 데이터 소스 전환: {'SIM->REAL' if is_real_mode else 'REAL->SIM'}")
                if is_real_mode:
                    # 실제 모드 진입 시 UART 연결 시도
                    uart_connect()
                prev_real_mode = is_real_mode

            if is_real_mode:
                # ----------------------------------------------------------
                # ESP32에서 센서 데이터 수신 (UART)
                # ----------------------------------------------------------
                # 받는 데이터 형식 (ESP32에서 1초마다 보내는 JSON 한 줄):
                #   {"v":[4.20,4.10,4.05,3.90], "mt":[30.5,...], "bt":[29.8,...], "i":[0.05,...]}
                #
                # v  = 셀 전압 4개 (ADS1115)
                # mt = MOSFET 온도 4개 (NTC 서미스터)
                # bt = 배터리 온도 4개 (DS18B20)
                # i  = 셀 밸런싱 전류 4개 (Rsense)
                received = uart_read_sensor()
                if received is not None:
                    # 데이터 도착 -> sim_state 덮어쓰기
                    try:
                        if "v" in received and len(received["v"]) == 4:
                            sim_state["v"] = [float(x) for x in received["v"]]
                        if "mt" in received and len(received["mt"]) == 4:
                            sim_state["mosfet_t"] = [float(x) for x in received["mt"]]
                        if "bt" in received and len(received["bt"]) == 4:
                            sim_state["battery_t"] = [float(x) for x in received["bt"]]
                        if "i" in received and len(received["i"]) == 4:
                            sim_state["i"] = [float(x) for x in received["i"]]
                    except (ValueError, TypeError, KeyError) as e:
                        print(f"[UART] 데이터 형식 오류 (무시): {e}")
                # 데이터 없으면 이전 sim_state 그대로 유지 (1초 주기에서 자연스러움)

            cells_v = sim_state["v"]
            cells_mt = sim_state["mosfet_t"]
            cells_bt = sim_state["battery_t"]

            # ============================================================
            # [HILS 핵심] 실측 → 가상 대전류 환경 변환
            # 
            # AI는 "가상 대전류 상태"만 입력받음
            # → 객관적 ground truth (물리식)로 환경 시뮬
            # → AI의 의사결정 능력만을 평가
            # ============================================================
            cells_i_real = sim_state["i"]
            cells_i_virtual = hc_sim.scale_current(cells_i_real)  # 0.5A → 50A
            
            # 가상 셀 전압 (IR drop 적용 - 옴의 법칙)
            cells_v_virtual = hc_sim.virtual_cell_voltage(cells_v, cells_i_virtual)
            
            # 가상 환경 온도 업데이트 (물리식)
            v_mt, v_ct, v_p_loss = hc_sim.update_virtual_temperatures(
                cells_i_virtual, prev_mode, dt=1.0, t_amb=TEMP_AMBIENT
            )
            
            # ============================================================
            # AI 입력은 가상 대전류 환경 상태
            # 실측 cells_mt(MOSFET 온도)는 소전류 상태라 의미 없음
            # → 가상 환경 v_mt 사용
            # ============================================================
            pack_dv_virtual = max(cells_v_virtual) - min(cells_v_virtual)
            max_mt_virtual = max(v_mt)
            max_bt_virtual = max(v_ct)
            min_v_virtual = min(cells_v_virtual)

            # AI 의사결정 (가상 대전류 상태 기반)
            mode, p_dac = ai_decide_mode(max_mt_virtual, max_bt_virtual, pack_dv_virtual)
            dac_vals, pwm_duty = ai_decide_cell_outputs(mode, v_mt, v_ct, cells_v_virtual)

            # ============================================================
            # [HILS 안전 출력] AI 출력은 ÷SCALE 해서 실제 회로엔 소전류로
            # ============================================================
            # AI는 "50A 흘려야 한다"고 결정했어도, 실제 회로엔 0.5A만 흐름
            # 모드/듀티 비율은 그대로 유지, 절대 전류만 스케일 다운
            dac_vals_safe = [int(d) for d in dac_vals]   # PWM 듀티는 비율이라 그대로
            pwm_duty_safe = [int(p) for p in pwm_duty]
            # DAC 값은 그대로 ESP32에 전달 (실제 회로의 R_BAL 10Ω 때문에 자동으로 작은 전류만 흐름)

            sim_state["dac_vals"] = dac_vals_safe
            sim_state["pwm_duty"] = pwm_duty_safe
            
            # 발표용 - 가상 환경 데이터를 sim_state에 보관
            sim_state["virtual_current_A"] = cells_i_virtual
            sim_state["virtual_voltage_V"] = cells_v_virtual
            sim_state["virtual_mosfet_T_C"] = v_mt
            sim_state["virtual_cell_T_C"] = v_ct
            sim_state["virtual_power_loss_W"] = v_p_loss

                       
            # ----------------------------------------------------------
            # [실제 모드] AI 출력을 ESP32로 전송 (UART)
            # ----------------------------------------------------------
            if is_real_mode:
                uart_send_command(mode, dac_vals_safe, pwm_duty_safe)

            #모드 전환 시 시뮬레이션 재동기화
            if mode != prev_mode:
                sim_state["pwm_mosfet_t"] = list(cells_mt)
                sim_state["dac_mosfet_t"] = list(cells_mt)
                sim_state["pwm_battery_t"] = list(cells_bt)
                sim_state["dac_battery_t"] = list(cells_bt)
                sim_state["pwm_v"] = list(cells_v)
                sim_state["dac_v"] = list(cells_v)
            prev_mode = mode

            pwm_phase = not pwm_phase
            cell_currents = [0.0] * 4
            cell_dt_mosfet = [0.0] * 4
            cell_dt_battery = [0.0] * 4

            for i in range(4):
                if mode == "STOP":
                    cell_currents[i] = 0.0
                elif mode == "PWM":
                    duty_ratio = pwm_duty[i] / 255.0
                    avg_current = duty_ratio * 0.5
                    if random.random() < duty_ratio:
                        cell_currents[i] = 0.5
                    else:
                        cell_currents[i] = 0.0
                    cell_dt_mosfet[i] = avg_current * HEAT_PWM_MOSFET + random.uniform(-0.02, 0.02)
                    cell_dt_battery[i] = (cells_mt[i] - cells_bt[i]) * HEAT_BATTERY_CONDUCT + avg_current * HEAT_BATTERY_INTERNAL
                else: 
                    target_i = (dac_vals[i] / 4095.0) * 0.4
                    cell_currents[i] = target_i
                    #미세 전류 구간에서(1A 미만) 선형 제어 발열은 전류랑에 거의 정비례하므로 제곱 안 함
                    cell_dt_mosfet[i] = target_i * HEAT_DAC_MOSFET + random.uniform(-0.03, 0.03)
                    cell_dt_battery[i] = (cells_mt[i] - cells_bt[i]) * HEAT_BATTERY_CONDUCT + target_i * HEAT_BATTERY_INTERNAL

            sim_state["i"] = cell_currents

            for i in range(4):
                cells_mt[i] += cell_dt_mosfet[i]
                cells_mt[i] -= COOL_K_MOSFET * (cells_mt[i] - TEMP_AMBIENT)
                cells_mt[i] = max(TEMP_AMBIENT, cells_mt[i])

                cells_bt[i] += cell_dt_battery[i]
                cells_bt[i] -= COOL_K_BATTERY * (cells_bt[i] - TEMP_AMBIENT)
                cells_bt[i] = max(TEMP_AMBIENT, cells_bt[i])
            
            # [4번 수정] PCB 열 결합 — MOSFET들이 PCB 통해 서로 영향
            # 한 셀이 풀파워면 옆 셀도 살짝 따뜻해짐 (현실)
            cells_mt = apply_pcb_heat_coupling(cells_mt)
            # 배터리는 물리적으로 떨어져 있어 열 결합 거의 없음 (그대로)

            # 실제 모드일 때는 위에서 받은 데이터로 sim_state 이미 갱신됨 -> 시뮬 계산 스킵
            # 시뮬 모드일 때만 발열/평탄화 시뮬 적용
            if not is_real_mode:
                if mode == "PWM":
                    cells_v_new = balance_toward_min(cells_v, cell_currents, smoothness=1.0)
                elif mode == "DAC":
                    cells_v_new = balance_toward_min(cells_v, cell_currents, smoothness=0.5)
                else: 
                    # [5번] STOP 모드 자가방전 — 단조 감소 (랜덤 아님)
                    cells_v_new = [v - SELF_DISCHARGE_RATES[i] for i, v in enumerate(cells_v)]

                sim_state["v"] = [max(3.0, min(4.25, v)) for v in cells_v_new]
                sim_state["mosfet_t"] = cells_mt
                sim_state["battery_t"] = cells_bt
            # 실제 모드: sim_state는 ESP32 데이터 그대로 사용 (덮어쓰기 안 함)

            # ============================================================
            # 6. 가상 우주
            # [9번 수정] 가상 우주를 더 공정하게:
            #   - 적응형 동작 (이미 있음, 유지)
            #   - PCB 열 결합도 동일 적용 (공정 비교)
            # ============================================================
            if mode == "PWM":
                sim_state["pwm_v"] = list(sim_state["v"])
                sim_state["pwm_i"] = list(cell_currents)
                sim_state["pwm_mosfet_t"] = list(cells_mt)
                sim_state["pwm_battery_t"] = list(cells_bt)

                dac_virt_min = min(sim_state["dac_v"])
                virt_dv = max(sim_state["dac_v"]) - dac_virt_min
                dac_virt_i_target = min(0.4, virt_dv * 2.5)
                # [9번] 더 공정하게: base 0.30 → 0.15 (DAC도 작은 전류 가능)
                dac_virt_i_base = max(0.15, dac_virt_i_target)  

                if virt_dv > 0.001:
                    dac_currents = [max(0.03, dac_virt_i_base * (v - dac_virt_min) / virt_dv) for v in sim_state["dac_v"]]
                else:
                    dac_currents = [0.03] * 4  

                for i in range(4):
                    ti = dac_currents[i]
                    #미세 전류 구간에서(1A 미만) 선형 제어 발열은 전류랑에 거의 정비례하므로 제곱 안 하고 곱으로만 처리
                    dt_m = ti * HEAT_DAC_MOSFET + random.uniform(-0.03, 0.03)
                    dt_b = ((sim_state["dac_mosfet_t"][i] - sim_state["dac_battery_t"][i]) * HEAT_BATTERY_CONDUCT + ti * HEAT_BATTERY_INTERNAL)
                    sim_state["dac_mosfet_t"][i] = max(TEMP_AMBIENT, sim_state["dac_mosfet_t"][i] + dt_m - COOL_K_MOSFET * (sim_state["dac_mosfet_t"][i] - TEMP_AMBIENT))
                    sim_state["dac_battery_t"][i] = max(TEMP_AMBIENT, sim_state["dac_battery_t"][i] + dt_b - COOL_K_BATTERY * (sim_state["dac_battery_t"][i] - TEMP_AMBIENT))

                # [4번] 가상 DAC 우주도 PCB 열 결합 적용 (공정 비교)
                sim_state["dac_mosfet_t"] = apply_pcb_heat_coupling(sim_state["dac_mosfet_t"])
                sim_state["dac_v"] = balance_toward_min(sim_state["dac_v"], dac_currents, smoothness=0.5)
                sim_state["dac_i"] = list(dac_currents)

            elif mode == "DAC":
                sim_state["dac_v"] = list(sim_state["v"])
                sim_state["dac_i"] = list(cell_currents)
                sim_state["dac_mosfet_t"] = list(cells_mt)
                sim_state["dac_battery_t"] = list(cells_bt)

                pwm_virt_min = min(sim_state["pwm_v"])
                pwm_virt_dv = max(sim_state["pwm_v"]) - pwm_virt_min

                pwm_currents = [0.0] * 4
                for i in range(4):
                    # [9번] 더 공정하게: 최소 듀티 0.3 → 0.1 (적응형 더 강하게)
                    if pwm_virt_dv > 0.001:
                        virt_duty = 0.1 + 0.9 * (sim_state["pwm_v"][i] - pwm_virt_min) / pwm_virt_dv
                    else:
                        virt_duty = 0.1
                    pwm_currents[i] = 0.5 if random.random() < virt_duty else 0.0

                    avg_i = virt_duty * 0.5
                    dt_m = avg_i * HEAT_PWM_MOSFET + random.uniform(-0.02, 0.02)
                    dt_b = ((sim_state["pwm_mosfet_t"][i] - sim_state["pwm_battery_t"][i]) * HEAT_BATTERY_CONDUCT + avg_i * HEAT_BATTERY_INTERNAL)
                    sim_state["pwm_mosfet_t"][i] = max(TEMP_AMBIENT, sim_state["pwm_mosfet_t"][i] + dt_m - COOL_K_MOSFET * (sim_state["pwm_mosfet_t"][i] - TEMP_AMBIENT))
                    sim_state["pwm_battery_t"][i] = max(TEMP_AMBIENT, sim_state["pwm_battery_t"][i] + dt_b - COOL_K_BATTERY * (sim_state["pwm_battery_t"][i] - TEMP_AMBIENT))

                # [4번] 가상 PWM 우주도 PCB 열 결합 적용
                sim_state["pwm_mosfet_t"] = apply_pcb_heat_coupling(sim_state["pwm_mosfet_t"])
                avg_currents = [pwm_currents[i] if pwm_currents[i] > 0 else 0.05 for i in range(4)]
                sim_state["pwm_v"] = balance_toward_min(sim_state["pwm_v"], avg_currents, smoothness=1.0)
                sim_state["pwm_i"] = list(pwm_currents)

            else:  
                sim_state["pwm_v"] = list(sim_state["v"])
                sim_state["dac_v"] = list(sim_state["v"])
                sim_state["pwm_i"] = [0.0] * 4
                sim_state["dac_i"] = [0.0] * 4
                sim_state["pwm_mosfet_t"] = list(cells_mt)
                sim_state["pwm_battery_t"] = list(cells_bt)
                sim_state["dac_mosfet_t"] = list(cells_mt)
                sim_state["dac_battery_t"] = list(cells_bt)

            # ============================================================
            # 7. WebSocket payload (출력 시에만 센서 흔들림 연출)
            # ============================================================
            def add_display_noise(v_list, is_pwm=False):
                # 출력용 가짜 노이즈 (상태 오염 X)
                n_range = 0.015 if is_pwm else 0.001
                return [round(v + random.uniform(-n_range, n_range), 3) for v in v_list]

            # 대전류 환경 위험도 판정
            critical, critical_reason = hc_sim.is_critical(max_mt_virtual, max_bt_virtual)

            payload = {
                "system_mode": "HIGH_CURRENT_HILS",
                "ai_loaded": AI_LOADED,
                "ai_mode": mode,
                "p_dac": round(p_dac, 3),
                
                # ============================================================
                # [HILS 발표용] 스케일 정보
                # ============================================================
                "hils": {
                    "scale_factor": hc_sim.SCALE,
                    "i_real_avg_A": round(sum(cells_i_real)/4, 3),
                    "i_virtual_avg_A": round(sum(cells_i_virtual)/4, 2),
                    "pack_delta_v_real": round(max(cells_v) - min(cells_v), 4),
                    "pack_delta_v_virtual": round(pack_dv_virtual, 4),
                    "critical": critical,
                    "critical_reason": critical_reason,
                    "derating_factor": round(hc_sim.derating_factor(max_mt_virtual), 3),
                    "total_power_loss_W": round(sum(v_p_loss), 2),
                },
                
                # 실제 측정 (안전한 소전류)
                "real": {
                    "v":         add_display_noise(sim_state["v"], is_pwm=(mode=="PWM")),
                    "i":         [round(c, 3) for c in cells_i_real],
                    "mosfet_t":  [round(t, 2) for t in cells_mt],
                    "battery_t": [round(t, 2) for t in cells_bt],
                    "dac_vals":  list(dac_vals_safe),
                    "pwm_duty":  list(pwm_duty_safe),
                },
                
                # ============================================================
                # [HILS 핵심] 가상 대전류 환경 (Plant Model 출력)
                # AI가 이 데이터를 보고 판단함
                # ============================================================
                "virtual": {
                    "i":         [round(c, 2) for c in cells_i_virtual],
                    "v":         [round(v, 3) for v in cells_v_virtual],
                    "mosfet_t":  [round(t, 2) for t in v_mt],
                    "cell_t":    [round(t, 2) for t in v_ct],
                    "p_loss":    [round(p, 2) for p in v_p_loss],
                },
                
                # 가상 우주 (참조용, 원본과 동일)
                "pwm": {
                    "v":         add_display_noise(sim_state["pwm_v"], is_pwm=True),
                    "i":         [round(c, 3) for c in sim_state["pwm_i"]],
                    "mosfet_t":  [round(t, 2) for t in sim_state["pwm_mosfet_t"]],
                    "battery_t": [round(t, 2) for t in sim_state["pwm_battery_t"]],
                },
                "dac": {
                    "v":         add_display_noise(sim_state["dac_v"], is_pwm=False),
                    "i":         [round(c, 3) for c in sim_state["dac_i"]],
                    "mosfet_t":  [round(t, 2) for t in sim_state["dac_mosfet_t"]],
                    "battery_t": [round(t, 2) for t in sim_state["dac_battery_t"]],
                },
            }

            await websocket.send_text(json.dumps(payload))
            
            # [1번 대응] 매 tick 데이터 로깅
            log_tick(sim_state, mode, p_dac, dac_vals_safe, pwm_duty_safe, is_real_mode)
            
            await asyncio.sleep(1.0)

    except Exception as e:
        print(f"[ERROR] 웹소켓: {e}")
        close_logger()

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("A²-BMS 대전류 환경 HILS 시뮬레이션 서버")
    print(f"  Scale Factor: ×{hc_sim.SCALE}")
    print(f"  Port: 8001 (원본 main_server.py는 8000)")
    print(f"  접속: http://localhost:8001")
    print("=" * 60)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    finally:
        close_logger()