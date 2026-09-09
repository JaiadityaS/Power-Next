# Mathematical Relationship Analysis Report

Generated: 2026-09-09 15:09:37.455399

## Regression Relationships (Feature → Reference_Parameter)

| Feature | Pearson r | Spearman r | Linear R² | Poly2 R² | MI | Partial r |
|---------|-----------|------------|-----------|----------|----|-----------|
| Applied_Voltage_kV | 0.2631 | 0.3339 | 0.0692 | 0.0698 | 0.2875 |  |
| Load_Current_A | 0.8664 | 0.8869 | 0.7507 | 0.8661 | 0.989 |  |
| Ambient_Temperature_C | 0.2062 | 0.1873 | 0.0425 | 0.0457 | 0.07 |  |
| Test_Duration_min | -0.0115 | -0.0081 | 0.0001 | 0.0005 | 0.0135 |  |
| Sensor_S1 | 0.5351 | 0.6133 | 0.2863 | 0.2897 | 0.5269 | -0.0214 |
| Sensor_S2 | 0.7045 | 0.7945 | 0.4963 | 0.4994 | 0.9027 | 0.025 |
| Sensor_S3 | 0.4746 | 0.5511 | 0.2253 | 0.2284 | 0.5228 | -0.0139 |
| Sensor_S4 | 0.0021 | -0.0205 | 0.0 | 0.0032 | 0.002 | 0.0683 |

## Classification Relationships (Feature → Validity_Label)

| Feature | Point-biserial r | Cohen's d | AUC | MI |
|---------|------------------|-----------|-----|----|
| Applied_Voltage_kV | 0.0094 | 0.0276 | 0.5084 | 0.0072 |
| Load_Current_A | -0.0182 | -0.0536 | 0.5153 | 0.0197 |
| Ambient_Temperature_C | -0.0042 | -0.0124 | 0.5012 | 0.0191 |
| Test_Duration_min | 0.0627 | 0.1844 | 0.5528 | 0.0055 |
| Sensor_S1 | -0.0072 | -0.0216 | 0.5029 | 0.0399 |
| Sensor_S2 | 0.0304 | 0.0899 | 0.5234 | 0.0454 |
| Sensor_S3 | -0.0112 | -0.0336 | 0.5019 | 0.0389 |
| Sensor_S4 | 0.0326 | 0.0946 | 0.5419 | 0.0266 |

## Bootstrap 95% CIs (Pearson → Ref Param)

- **Applied_Voltage_kV**: [0.2086, 0.3174]
- **Load_Current_A**: [0.853, 0.8804]
- **Ambient_Temperature_C**: [0.1459, 0.2669]
- **Test_Duration_min**: [-0.076, 0.0487]
- **Sensor_S1**: [0.4886, 0.5816]
- **Sensor_S2**: [0.6704, 0.7391]
- **Sensor_S3**: [0.421, 0.523]
- **Sensor_S4**: [-0.0611, 0.0601]

## VIF (Multicollinearity)

- **Applied_Voltage_kV**: VIF = 14.99
- **Load_Current_A**: VIF = 4.36
- **Ambient_Temperature_C**: VIF = 1.01
- **Test_Duration_min**: VIF = 1.05
- **Sensor_S1**: VIF = 6.36
- **Sensor_S2**: VIF = 5.13
- **Sensor_S3**: VIF = 9.88
- **Sensor_S4**: VIF = 1.01

## Formula Investigation

| Model | In-sample R² | In-sample MAE |
|-------|-------------|---------------|
| Linear Regression | 0.8454 | 3.4228 |
| Ridge Regression | 0.8454 | 3.4227 |
| Poly-2 Regression | 0.9626 | 1.0652 |
| Huber Regression | 0.8394 | 3.3166 |
| Spline + Ridge | 0.9658 | 0.9445 |
| HistGBR (baseline) | 0.9821 | 0.5049 |

### Linear Regression Formula

Reference_Parameter = -12.3954 + (0.3717) × Applied_Voltage_kV + (0.3263) × Load_Current_A + (0.2834) × Ambient_Temperature_C + (0.0213) × Test_Duration_min + (-0.0610) × Sensor_S1 + (0.0599) × Sensor_S2 + (-0.0231) × Sensor_S3 + (0.0200) × Sensor_S4

### Cross-Validated Formula Comparison

| model   |   CV_MAE_mean |   CV_MAE_std |   CV_R2_mean |   CV_R2_std |
|:--------|--------------:|-------------:|-------------:|------------:|
| Linear  |        3.4693 |       0.1334 |       0.8394 |      0.0247 |
| Ridge   |        3.4682 |       0.1312 |       0.8394 |      0.0244 |
| Huber   |        3.3652 |       0.1061 |       0.8344 |      0.0193 |
| HistGBR |        0.9644 |       0.094  |       0.9556 |      0.0192 |
