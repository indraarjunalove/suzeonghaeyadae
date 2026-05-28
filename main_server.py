import asyncio
import json
import random
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 프론트엔드(index.html)에서 접근 가능하도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Render 클라우드 또는 로컬 PC 단독 가동을 위해 무조건 False로 고정
is_real_mode = False 

# AI 모델 로딩부 (에러 방어 로직 적용)
# (경로에 .pkl 파일이 없어도 서버가 죽지 않고 Rule-base로 돌아가도록 방어막을 쳤습니다)
try:
    import joblib
    # 실제로는 아래 주석을 풀고 모델을 로드하면 됩니다.
    # bms_mode_ai = joblib.load('bms_mode_ai.pkl')
    # bms_pwm_ai = joblib.load('bms_pwm_ai.pkl')
    # bms_dac_ai = joblib.load('bms_dac_ai.pkl')
    ai_loaded = True
    print("[INFO] AI 모델 로딩 성공")
except Exception as e:
    ai_loaded = False
    print(f"[WARNING] AI 모델 로딩 실패 (Rule-base 가동): {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] 웹소켓 클라이언트 접속 완료")
    
    # 1. 초기 실측(Real) 상태 가상 변수
    real_v = [3.85, 3.82, 3.79, 3.75]
    real_i = [0.25, 0.24, 0.23, 0.22]
    real_mt = [32.5, 33.1, 31.8, 32.0]
    real_bt = [28.5, 29.0, 28.2, 28.4]
    
    try:
        while True:
            # ==========================================
            # 1. 시스템 상태 및 AI 판단 흉내내기
            # ==========================================
            delta_v = max(real_v) - min(real_v)
            # 전압차가 0.05V 이상이면 PWM 고속 밸런싱, 아니면 DAC 정밀 밸런싱
            ai_mode = "PWM" if delta_v > 0.05 else "DAC"
            p_dac = 0.85 if ai_mode == "DAC" else 0.15
            
            # 가상 전압 평탄화 (시간이 지날수록 전압이 맞춰짐)
            for i in range(4):
                if real_v[i] > 3.75:
                    real_v[i] -= 0.002
                # 열역학적 방열 및 발열 모의
                real_mt[i] = 30.0 + (real_i[i] * 10)
                real_bt[i] = 28.0 + (real_i[i] * 2)

            # ==========================================
            # 2. [핵심] 모드별 노이즈 주입 및 물리 연산
            # ==========================================
            pwm_v = []
            dac_v = []
            high_v = []
            
            for i in range(4):
                # [저전류 PWM]: ±0.03V 수준의 잔잔한 스위칭 노이즈 주입
                low_noise = random.uniform(-0.03, 0.03)
                pwm_v.append(real_v[i] + low_noise)
                
                # [저전류 DAC]: 리플 노이즈가 없는 완벽한 선형 전압 추종
                dac_v.append(real_v[i])
                
                # [10A 대전류 PWM]: 전류가 큰 만큼 스위칭 시 발생하는 전압 요동을 ±0.15V로 극대화
                high_noise = random.uniform(-0.25, 0.25)
                # I_virtual = 10A, R_cell = 0.022옴 가정 시 전압 강하 (V = IR)
                v_drop = 10.0 * 0.022 
                high_v.append(real_v[i] - v_drop + high_noise)

            # ==========================================
            # 3. 프론트엔드로 쏠 JSON 데이터 패키징
            # ==========================================
            data = {
                "ai_mode": ai_mode,
                "p_dac": p_dac,
                "pack_delta_v": delta_v,
                "pack_min_v": min(real_v),
                "ai_loaded": ai_loaded,
                
                # 대전류(High) 패널 데이터
                "high": { 
                    "ai_mode": "PWM", 
                    "p_dac": 0.10, 
                    "v": high_v, 
                    # 대전류 스위칭 시 전류값도 ±0.2A 정도로 흔들리게 세팅
                    "i": [10.0 + random.uniform(-0.2, 0.2) for _ in range(4)], 
                    # 10A 발열(I^2R) 반영: 온도가 급격히 상승하는 모습
                    "mosfet_t": [min(95, mt + 25.5 + random.uniform(-1, 1)) for mt in real_mt], 
                    "battery_t": [min(70, bt + 12.0) for bt in real_bt] 
                },
                
                # 실측(Real) 패널 데이터
                "real": {
                    "v": real_v,
                    "i": real_i,
                    "mosfet_t": real_mt,
                    "battery_t": real_bt,
                    "dac_vals": [2048, 1024, 512, 256] if ai_mode == "DAC" else [0, 0, 0, 0],
                    "pwm_duty": [200, 150, 100, 50] if ai_mode == "PWM" else [0, 0, 0, 0]
                },
                
                # 저전류 가상 데이터
                "pwm": {
                    "v": pwm_v,
                    "i": [0.5 for _ in range(4)],
                    "mosfet_t": [mt + 2 for mt in real_mt],
                    "battery_t": [bt + 1 for bt in real_bt]
                },
                "dac": {
                    "v": dac_v,
                    "i": [0.25 for _ in range(4)],
                    "mosfet_t": [mt + 5 for mt in real_mt],
                    "battery_t": [bt + 2 for bt in real_bt]
                }
            }
            
            # 클라이언트로 데이터 전송 및 1초 대기
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print("[INFO] 웹소켓 클라이언트 연결 끊김")
    except Exception as e:
        print(f"[ERROR] 시뮬레이션 루프 에러: {e}")

# Render 배포 시 이 구문으로 실행됩니다.
# uvicorn main_server:app --host 0.0.0.0 --port $PORT
