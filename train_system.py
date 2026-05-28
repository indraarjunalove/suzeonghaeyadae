"""
A²-BMS AI 모델 학습 (소전류 + 대전류 통합)
==================================================
방안 B: 환경별 적응형 AI - 6개 모델 학습

[소전류 모델 (학부 실험)]
  - bms_mode_ai_low.pkl  (분류기)
  - bms_dac_ai_low.pkl   (DAC 회귀기)
  - bms_pwm_ai_low.pkl   (PWM 회귀기)

[대전류 모델 (가상 HILS)]
  - bms_mode_ai_high.pkl (분류기)
  - bms_dac_ai_high.pkl  (DAC 회귀기)
  - bms_pwm_ai_high.pkl  (PWM 회귀기)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
import joblib

PACK_FEATURES = ['Max_Mosfet_T', 'Max_Battery_T', 'Pack_Delta_V']
CELL_FEATURES = ['Mosfet_T', 'Battery_T', 'Cell_V', 'Delta_From_Min']


# ============================================================
# 통합 학습 함수
# ============================================================
def train_classifier(pack_file, model_out, env_name):
    """모드 분류기 학습"""
    print(f"\n[{env_name}] 모드 분류기 학습")
    print("-" * 60)
    pack_df = pd.read_csv(pack_file)
    pwm_n = (pack_df['Label'] == 'PWM').sum()
    dac_n = (pack_df['Label'] == 'DAC').sum()
    print(f"  데이터: {len(pack_df)}개 (PWM:{pwm_n} / DAC:{dac_n})")

    X = pack_df[PACK_FEATURES]
    y = pack_df['Label']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=30, max_depth=8, random_state=42, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred) * 100
    print(f"  정확도: {acc:.2f}%")
    print(f"  Feature 중요도:")
    for n, i in zip(PACK_FEATURES, clf.feature_importances_):
        print(f"    {n}: {i*100:.2f}%")

    joblib.dump(clf, model_out)
    print(f"  → {model_out} 저장")
    return clf


def train_regressor(cell_file, target_col, target_range, model_out, env_name, regressor_type):
    """회귀기 학습 (DAC 또는 PWM)"""
    print(f"\n[{env_name}] {regressor_type} 회귀기 학습")
    print("-" * 60)
    cell_df = pd.read_csv(cell_file)
    print(f"  데이터: {len(cell_df)}개")

    X = cell_df[CELL_FEATURES]
    y = cell_df[target_col]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=30, max_depth=8, random_state=42, n_jobs=-1
    )
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    print(f"  MAE: {mae:.2f} (전체 범위 {target_range}의 {mae/target_range*100:.2f}%)")
    print(f"  Feature 중요도:")
    for n, i in zip(CELL_FEATURES, reg.feature_importances_):
        print(f"    {n}: {i*100:.2f}%")

    joblib.dump(reg, model_out)
    print(f"  → {model_out} 저장")
    return reg


# ============================================================
# 실행
# ============================================================
def main():
    print("=" * 60)
    print("A²-BMS AI 모델 학습 (소전류 + 대전류 통합)")
    print("=" * 60)

    # ===== 소전류 환경 모델 =====
    print("\n" + "=" * 60)
    print("Part 1/2: 소전류 환경 모델 학습 (학부 실험용)")
    print("=" * 60)
    train_classifier(
        'bms_pack_data_low.csv',
        'bms_mode_ai_low.pkl',
        '소전류'
    )
    train_regressor(
        'bms_cell_data_low.csv',
        'DAC_Val', 4095,
        'bms_dac_ai_low.pkl',
        '소전류', 'DAC'
    )
    train_regressor(
        'bms_cell_data_low.csv',
        'PWM_Duty', 255,
        'bms_pwm_ai_low.pkl',
        '소전류', 'PWM'
    )

    # ===== 대전류 환경 모델 =====
    print("\n" + "=" * 60)
    print("Part 2/2: 대전류 환경 모델 학습 (가상 HILS용)")
    print("=" * 60)
    train_classifier(
        'bms_pack_data_high.csv',
        'bms_mode_ai_high.pkl',
        '대전류'
    )
    train_regressor(
        'bms_cell_data_high.csv',
        'DAC_Val', 4095,
        'bms_dac_ai_high.pkl',
        '대전류', 'DAC'
    )
    train_regressor(
        'bms_cell_data_high.csv',
        'PWM_Duty', 255,
        'bms_pwm_ai_high.pkl',
        '대전류', 'PWM'
    )

    # ===== 정리 =====
    print("\n" + "=" * 60)
    print("학습 완료 - 6개 모델 저장됨")
    print("=" * 60)
    print("[소전류 모델 (학부 실험용)]")
    print("  - bms_mode_ai_low.pkl  (모드 분류기)")
    print("  - bms_dac_ai_low.pkl   (DAC 회귀기)")
    print("  - bms_pwm_ai_low.pkl   (PWM 회귀기)")
    print()
    print("[대전류 모델 (가상 HILS용)]")
    print("  - bms_mode_ai_high.pkl (모드 분류기)")
    print("  - bms_dac_ai_high.pkl  (DAC 회귀기)")
    print("  - bms_pwm_ai_high.pkl  (PWM 회귀기)")
    print("=" * 60)


if __name__ == "__main__":
    main()
