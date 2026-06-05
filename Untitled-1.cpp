/*
 * ============================================================
 * A2-BMS ESP32 Firmware
 * 최종 회로도 기준 버전
 * ============================================================
 *
 * [ESP32 역할]
 * 1. Raspberry Pi로부터 UART(JSON) 명령 수신
 * 2. 셀별 PWM MOSFET 제어
 * 3. 셀별 MCP4725 DAC 제어
 * 4. 셀별 ADS1115 센서 측정
 * 5. 전압 / 전류 / MOSFET 온도 / 배터리 온도 송신
 * 6. 통신 끊김, 과전압, 저전압, 과온 발생 시 출력 차단
 *
 * [최종 회로 구조]
 * - Cell1~Cell4 각각 독립 ADS1115 사용
 * - Cell1~Cell4 각각 독립 MCP4725 사용
 * - TCA9548A로 셀별 I2C 채널 선택
 * - ISO1540으로 셀별 I2C 절연
 *
 * [ADS1115 채널 매핑]
 * A0 = 셀 전압 BATn+
 * A1 = Vsensen, RSENSE 상단 전압
 * A2 = NTC_sensen
 * A3 = 미사용
 *
 * [UART 명령 예시]
 * {"mode":"PWM","dac":[0,0,0,0],"pwm":[200,150,100,50]}
 *
 * mode:
 * STOP   = 전체 출력 차단
 * PWM    = PWM MOSFET만 사용
 * DAC    = DAC 선형제어만 사용
 * HYBRID = PWM + DAC 동시 사용
 */

#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_MCP4725.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ============================================================
// ESP32 핀 설정
// ============================================================

// ESP32 I2C 핀
#define I2C_SDA 21
#define I2C_SCL 22

// DS18B20 OneWire 데이터 핀
#define ONEWIRE_PIN 4

// 셀별 PWM MOSFET 게이트 제어 핀
// Cell1 → GPIO25
// Cell2 → GPIO26
// Cell3 → GPIO27
// Cell4 → GPIO14
const int PWM_PINS[4] = {25, 26, 27, 14};

// ESP32 LEDC PWM 채널
const int PWM_CHANNELS[4] = {0, 1, 2, 3};

// PWM 주파수와 해상도
// 5kHz, 8bit → duty 0~255
const int PWM_FREQ = 5000;
const int PWM_RES = 8;

// ============================================================
// I2C 주소 설정
// ============================================================

// TCA9548A I2C 멀티플렉서 주소
#define TCA_ADDR 0x70

// 각 셀 채널 안에 있는 ADS1115 주소
// TCA로 채널이 분리되어 있으므로 각 ADS1115 주소는 같아도 됨
#define ADS_ADDR 0x48

// 각 셀 채널 안에 있는 MCP4725 주소
// TCA로 채널이 분리되어 있으므로 각 DAC 주소는 같아도 됨
#define DAC_ADDR 0x60

// TCA9548A 채널 매핑
// 회로도 기준:
// CH0 = Cell1
// CH1 = Cell2
// CH2 = Cell3
// CH3 = Cell4
const uint8_t CELL_TCA_CH[4] = {0, 1, 2, 3};

// ============================================================
// ADS1115 채널 매핑
// ============================================================

#define ADS_CH_CELL_VOLTAGE 0  // A0: BATn+ 전압
#define ADS_CH_VSENSE       1  // A1: RSENSE 상단 전압
#define ADS_CH_NTC          2  // A2: NTC_sensen
#define ADS_CH_UNUSED       3  // A3: 미사용

// ============================================================
// 회로 상수
// ============================================================

// 전류 센싱 저항
// Vsense = I × R_SENSE
// 따라서 I = Vsense / R_SENSE
const float R_SENSE = 0.1;

// NTC 계산용 상수
// 회로 가정:
// VCC_BATn -- 10kΩ 고정저항 -- NTC_sense -- NTC -- BATn-
const float NTC_R_FIXED = 10000.0;
const float NTC_BETA = 3950.0;
const float NTC_R0 = 10000.0;
const float T0_K = 298.15;

// ============================================================
// 안전 기준값
// ============================================================

// 셀 과전압 차단 기준
const float CELL_OV_LIMIT = 4.25;

// 셀 저전압 차단 기준
const float CELL_UV_LIMIT = 2.80;

// MOSFET 온도 차단 기준
const float MOS_TEMP_LIMIT = 70.0;

// 배터리 온도 차단 기준
const float BAT_TEMP_LIMIT = 55.0;

// Raspberry Pi 명령이 끊겼을 때 차단 시간
const unsigned long CMD_TIMEOUT_MS = 3000;

// 센서 데이터 송신 주기
const unsigned long SEND_INTERVAL_MS = 1000;

// ============================================================
// 전역 객체
// ============================================================

// TCA 채널을 바꿔가며 같은 객체로 각 셀의 ADS/DAC에 접근
Adafruit_ADS1115 ads;
Adafruit_MCP4725 dac;

// DS18B20 배터리 온도 센서
OneWire oneWire(ONEWIRE_PIN);
DallasTemperature ds18b20(&oneWire);

// 마지막 명령 수신 시간
unsigned long lastCmdTime = 0;

// 마지막 센서 송신 시간
unsigned long lastSendTime = 0;

// ============================================================
// TCA9548A 채널 선택 함수
// ============================================================
//
// TCA9548A는 I2C 스위치 역할을 한다.
// ESP32는 먼저 TCA 채널을 선택하고,
// 그 뒤 해당 채널 안의 ADS1115/MCP4725와 통신한다.
//
// 예:
// ch=0 → Cell1 I2C 버스 연결
// ch=1 → Cell2 I2C 버스 연결
// ch=2 → Cell3 I2C 버스 연결
// ch=3 → Cell4 I2C 버스 연결
//
bool tcaSelect(uint8_t ch) {
  if (ch > 7) return false;

  Wire.beginTransmission(TCA_ADDR);

  // 1 << ch는 특정 채널 하나만 활성화한다.
  // 예: ch=2이면 00000100 전송
  Wire.write(1 << ch);

  return Wire.endTransmission() == 0;
}

// ============================================================
// 셀 선택 함수
// ============================================================
//
// cell 값:
// 0 = Cell1
// 1 = Cell2
// 2 = Cell3
// 3 = Cell4
//
bool selectCell(uint8_t cell) {
  if (cell >= 4) return false;
  return tcaSelect(CELL_TCA_CH[cell]);
}

// ============================================================
// 셀별 ADS1115/MCP4725 초기화 확인
// ============================================================
//
// setup()에서 각 셀 채널을 선택한 뒤,
// 해당 채널의 ADS1115와 MCP4725가 응답하는지 확인한다.
//
bool initCellDevices(uint8_t cell) {
  if (!selectCell(cell)) return false;

  bool adsOk = ads.begin(ADS_ADDR);
  bool dacOk = dac.begin(DAC_ADDR);

  // 부팅 직후 DAC 출력은 항상 0으로 초기화
  if (dacOk) {
    dac.setVoltage(0, false);
  }

  return adsOk && dacOk;
}

// ============================================================
// 전체 출력 차단
// ============================================================
//
// 안전 문제가 생기면 모든 셀의 PWM과 DAC를 0으로 만든다.
// 즉 모든 밸런싱 MOSFET을 OFF 상태로 만든다.
//
void forceShutdown() {
  for (int cell = 0; cell < 4; cell++) {
    // PWM MOSFET OFF
    ledcWrite(PWM_CHANNELS[cell], 0);

    // DAC 선형제어 MOSFET OFF
    if (selectCell(cell)) {
      dac.setVoltage(0, false);
    }
  }
}

// ============================================================
// ADS1115 전압 읽기
// ============================================================
//
// 채널별로 필요한 측정 범위가 다르므로 gain을 다르게 설정한다.
//
// A0 셀 전압:
// - 최대 약 4.2V
// - GAIN_TWOTHIRDS 사용, ±6.144V 범위
//
// A1 Vsense:
// - RSENSE 0.1Ω에 걸리는 전압
// - 밸런싱 전류가 0.4A여도 약 40mV 수준
// - GAIN_SIXTEEN 사용, ±0.256V 범위
//
// A2 NTC:
// - 0~3.3V 범위 분압 신호
// - GAIN_ONE 사용, ±4.096V 범위
//
float readADSVoltage(uint8_t cell, uint8_t channel) {
  if (!selectCell(cell)) return -1.0;

  if (channel == ADS_CH_CELL_VOLTAGE) {
    ads.setGain(GAIN_TWOTHIRDS);
  }
  else if (channel == ADS_CH_VSENSE) {
    ads.setGain(GAIN_SIXTEEN);
  }
  else {
    ads.setGain(GAIN_ONE);
  }

  // gain 변경 후 안정화 대기
  delay(2);

  int16_t raw = ads.readADC_SingleEnded(channel);
  return ads.computeVolts(raw);
}

// ============================================================
// 셀 전압 측정
// ============================================================
//
// 각 셀의 ADS1115는 해당 셀의 BATn-를 GND로 사용한다.
// 따라서 A0에 연결된 BATn+를 읽으면 해당 셀 전압이 된다.
//
float readCellVoltage(uint8_t cell) {
  return readADSVoltage(cell, ADS_CH_CELL_VOLTAGE);
}

// ============================================================
// 밸런싱 전류 측정
// ============================================================
//
// 회로 구조:
//
// BATn+
//   |
// RBAL 10Ω
//   |
// MOSFET
//   |
// Vsense
//   |
// RSENSE 0.1Ω
//   |
// BATn-
//
// ADS1115 A1은 Vsense를 읽는다.
// ADS1115 GND는 BATn- 기준이다.
//
// 따라서 ADS가 읽는 전압은:
// Vsense - BATn- = I × RSENSE
//
// 전류 계산:
// I = Vsense / RSENSE
//
float readBalanceCurrent(uint8_t cell) {
  float vsense = readADSVoltage(cell, ADS_CH_VSENSE);

  if (vsense < 0) return -1.0;

  return vsense / R_SENSE;
}

// ============================================================
// MOSFET 주변 NTC 온도 측정
// ============================================================
//
// 회로 가정:
//
// VCC_BATn
//   |
// 10kΩ 고정저항
//   |
// NTC_sense
//   |
// NTC
//   |
// BATn-
//
// 이 경우:
// R_NTC = R_FIXED × Vntc / (VCC - Vntc)
//
// 이후 Beta 식으로 온도 계산.
//
float readMosTemp(uint8_t cell) {
  float v = readADSVoltage(cell, ADS_CH_NTC);

  // 비정상 범위 방어
  if (v <= 0.01 || v >= 3.29) {
    return -999.0;
  }

  float r_ntc = NTC_R_FIXED * v / (3.3 - v);

  float tempK =
      1.0 / ((1.0 / T0_K) + (log(r_ntc / NTC_R0) / NTC_BETA));

  return tempK - 273.15;
}

// ============================================================
// DAC 출력 설정
// ============================================================
//
// MCP4725는 12bit DAC이다.
// 입력값 0~4095가 출력 전압 0~VCC에 대응한다.
//
// 이 DAC 출력은 Op-Amp/MOSFET 선형제어 쪽으로 들어간다.
//
void setCellDAC(uint8_t cell, int value) {
  value = constrain(value, 0, 4095);

  if (!selectCell(cell)) return;

  dac.setVoltage(value, false);
}

// ============================================================
// PWM 출력 설정
// ============================================================
//
// ESP32 LEDC PWM을 이용해 셀별 PWM MOSFET을 제어한다.
// duty 값은 0~255이다.
//
// 0   = 완전 OFF
// 255 = 거의 항상 ON
//
void setCellPWM(uint8_t cell, int duty) {
  if (cell >= 4) return;

  duty = constrain(duty, 0, 255);
  ledcWrite(PWM_CHANNELS[cell], duty);
}

// ============================================================
// 안전 상태 검사
// ============================================================
//
// 하나라도 위험 조건이면 false 반환.
// false가 나오면 forceShutdown()을 실행한다.
//
bool isSafe(float v[4], float i[4], float mt[4], float bt[4]) {
  for (int cell = 0; cell < 4; cell++) {
    // 센서 읽기 실패
    if (v[cell] < 0) return false;
    if (i[cell] < 0) return false;
    if (mt[cell] == -999.0) return false;

    // 셀 과전압 / 저전압
    if (v[cell] > CELL_OV_LIMIT) return false;
    if (v[cell] < CELL_UV_LIMIT) return false;

    // MOSFET 과온
    if (mt[cell] > MOS_TEMP_LIMIT) return false;

    // DS18B20이 연결되어 있을 때만 배터리 온도 검사
    if (bt[cell] != DEVICE_DISCONNECTED_C && bt[cell] > BAT_TEMP_LIMIT) {
      return false;
    }
  }

  return true;
}

// ============================================================
// Raspberry Pi 명령 처리
// ============================================================
//
// 입력 JSON 예시:
//
// {"mode":"PWM","dac":[0,0,0,0],"pwm":[200,150,100,50]}
//
// mode = STOP:
// - 모든 출력 차단
//
// mode = PWM:
// - DAC 출력은 0
// - PWM만 사용
//
// mode = DAC:
// - PWM 출력은 0
// - DAC만 사용
//
// mode = HYBRID:
// - PWM과 DAC를 모두 사용
//
void processCommand(String json) {
  StaticJsonDocument<512> doc;

  DeserializationError err = deserializeJson(doc, json);

  if (err) {
    forceShutdown();
    return;
  }

  const char* mode = doc["mode"];

  if (mode == nullptr) {
    forceShutdown();
    return;
  }

  if (strcmp(mode, "STOP") == 0) {
    forceShutdown();
    return;
  }

  JsonArray dac_arr = doc["dac"];
  JsonArray pwm_arr = doc["pwm"];

  // 배열이 없거나 길이가 부족하면 위험하므로 차단
  if (dac_arr.size() < 4 || pwm_arr.size() < 4) {
    forceShutdown();
    return;
  }

  if (strcmp(mode, "PWM") == 0) {
    for (int cell = 0; cell < 4; cell++) {
      setCellDAC(cell, 0);
      setCellPWM(cell, pwm_arr[cell].as<int>());
    }
  }
  else if (strcmp(mode, "DAC") == 0) {
    for (int cell = 0; cell < 4; cell++) {
      setCellPWM(cell, 0);
      setCellDAC(cell, dac_arr[cell].as<int>());
    }
  }
  else if (strcmp(mode, "HYBRID") == 0) {
    for (int cell = 0; cell < 4; cell++) {
      setCellDAC(cell, dac_arr[cell].as<int>());
      setCellPWM(cell, pwm_arr[cell].as<int>());
    }
  }
  else {
    // 알 수 없는 mode면 차단
    forceShutdown();
  }
}

// ============================================================
// 센서 데이터 송신
// ============================================================
//
// ESP32 → Raspberry Pi 송신 JSON 예시:
//
// {
//   "v":[4.10,4.08,4.05,4.02],
//   "i":[0.30,0.20,0.10,0.00],
//   "mt":[35.2,36.1,34.8,35.0],
//   "bt":[28.5,28.7,28.4,28.6],
//   "safe":true,
//   "shutdown":false
// }
//
void sendSensorData() {
  StaticJsonDocument<768> doc;

  float v[4];
  float i[4];
  float mt[4];
  float bt[4];

  JsonArray v_arr = doc.createNestedArray("v");
  JsonArray i_arr = doc.createNestedArray("i");
  JsonArray mt_arr = doc.createNestedArray("mt");
  JsonArray bt_arr = doc.createNestedArray("bt");

  // 셀별 ADS1115 데이터 측정
  for (int cell = 0; cell < 4; cell++) {
    v[cell] = readCellVoltage(cell);
    i[cell] = readBalanceCurrent(cell);
    mt[cell] = readMosTemp(cell);

    v_arr.add(v[cell]);
    i_arr.add(i[cell]);
    mt_arr.add(mt[cell]);
  }

  // DS18B20 배터리 온도 측정
  ds18b20.requestTemperatures();

  for (int cell = 0; cell < 4; cell++) {
    bt[cell] = ds18b20.getTempCByIndex(cell);
    bt_arr.add(bt[cell]);
  }

  bool safe = isSafe(v, i, mt, bt);

  doc["safe"] = safe;

  if (!safe) {
    forceShutdown();
    doc["shutdown"] = true;
  }
  else {
    doc["shutdown"] = false;
  }

  serializeJson(doc, Serial);
  Serial.println();
}

// ============================================================
// setup
// ============================================================
void setup() {
  Serial.begin(115200);

  // I2C 시작
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);

  bool allInitOk = true;

  // 셀별 ADS1115/MCP4725 초기화 확인
  for (int cell = 0; cell < 4; cell++) {
    bool ok = initCellDevices(cell);

    if (!ok) {
      allInitOk = false;

      Serial.print("{\"cell_init_error\":");
      Serial.print(cell + 1);
      Serial.println("}");
    }
  }

  // PWM 초기화
  for (int cell = 0; cell < 4; cell++) {
    ledcSetup(PWM_CHANNELS[cell], PWM_FREQ, PWM_RES);
    ledcAttachPin(PWM_PINS[cell], PWM_CHANNELS[cell]);
    ledcWrite(PWM_CHANNELS[cell], 0);
  }

  // DS18B20 초기화
  ds18b20.begin();
  ds18b20.setResolution(12);

  // 부팅 직후 모든 출력 차단
  forceShutdown();

  lastCmdTime = millis();
  lastSendTime = millis();

  if (allInitOk) {
    Serial.println("{\"status\":\"ESP32_READY_FINAL_CIRCUIT\"}");
  }
  else {
    Serial.println("{\"status\":\"ESP32_READY_WITH_INIT_ERROR\"}");
  }
}

// ============================================================
// loop
// ============================================================
void loop() {
  // Raspberry Pi에서 명령 수신
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    processCommand(line);

    // 정상/비정상 명령 여부와 관계없이 통신은 들어왔으므로 시간 갱신
    lastCmdTime = millis();
  }

  // 일정 시간 명령이 없으면 전체 출력 차단
  if (millis() - lastCmdTime > CMD_TIMEOUT_MS) {
    forceShutdown();
  }

  // 1초마다 센서 데이터 송신
  if (millis() - lastSendTime > SEND_INTERVAL_MS) {
    sendSensorData();
    lastSendTime = millis();
  }
}