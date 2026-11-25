"""선형화 모델로 PID 게인을 '계산'으로 정한다.

직선 도로 근처에서 오차가 작다고 보면 (sin x ≈ x, tan x ≈ x)
    e_y'   = v * e_psi
    e_psi' = (v/L) * delta - v * kappa
두 식을 합치면
    e_y'' = (v^2/L) * delta - v^2 * kappa
→ 조향각 → 횡방향 오차의 전달함수는 이중 적분기:
    G(s) = E_y(s) / Delta(s) = a / s^2,   a = v^2 / L
곡선 도로의 곡률 kappa 는 '외란'으로 들어온다.
"""
import numpy as np
import control as ct


def plant_tf(v, L):
    a = v ** 2 / L
    return ct.tf([a], [1, 0, 0]), a


def pd_gains(v, L, zeta, wn):
    """폐루프 특성방정식  s^2 + a*Kd*s + a*Kp = 0  을
    표준 2차계  s^2 + 2*zeta*wn*s + wn^2  에 맞춘다."""
    a = v ** 2 / L
    return wn ** 2 / a, 2 * zeta * wn / a


def pid_gains(v, L, zeta, wn, p):
    """특성방정식  s^3 + a*Kd*s^2 + a*Kp*s + a*Ki = 0  을
    (s^2 + 2*zeta*wn*s + wn^2)(s + p) 에 맞춘다 (극점 배치)."""
    a = v ** 2 / L
    Kd = (2 * zeta * wn + p) / a
    Kp = (wn ** 2 + 2 * zeta * wn * p) / a
    Ki = (wn ** 2 * p) / a
    return Kp, Ki, Kd


def routh_ok(a, Kp, Ki, Kd):
    """3차 특성방정식 s^3 + c2 s^2 + c1 s + c0 의 Routh 안정 조건:
    모든 계수 > 0  그리고  c2*c1 > c0   →   Ki < a*Kp*Kd"""
    c2, c1, c0 = a * Kd, a * Kp, a * Ki
    return (c2 > 0 and c1 > 0 and c0 > 0 and c2 * c1 > c0), (c2, c1, c0)


def closed_loop_poles(v, L, Kp, Ki, Kd):
    G, a = plant_tf(v, L)
    C = ct.tf([Kd, Kp, Ki], [1, 0]) if Ki else ct.tf([Kd, Kp], [1])
    T = ct.feedback(C * G, 1)
    return ct.poles(T)


def lateral_error_ss(car, vx):
    """동역학 모델을 '차선 오차' 상태공간으로 바꾼다 (Rajamani, Vehicle Dynamics and Control 2장).

    상태 x = [e_y, e_y', e_psi, e_psi'],  입력 u = delta,  외란 d = 경로 요레이트(vx*kappa)
        x' = A x + B u + E d
    여기서 Cf, Cr은 '축 하나'의 코너링 강성(타이어 2개 합)이다.
    """
    m, Iz, lf, lr, Cf, Cr = car.m, car.Iz, car.lf, car.lr, car.Cf, car.Cr
    A = np.array([
        [0, 1, 0, 0],
        [0, -(Cf + Cr) / (m * vx), (Cf + Cr) / m, (-Cf * lf + Cr * lr) / (m * vx)],
        [0, 0, 0, 1],
        [0, -(Cf * lf - Cr * lr) / (Iz * vx), (Cf * lf - Cr * lr) / Iz, -(Cf * lf**2 + Cr * lr**2) / (Iz * vx)],
    ])
    B = np.array([[0], [Cf / m], [0], [Cf * lf / Iz]])
    E = np.array([[0], [-(Cf * lf - Cr * lr) / (m * vx) - vx], [0], [-(Cf * lf**2 + Cr * lr**2) / (Iz * vx)]])
    return A, B, E


def understeer_gradient(car):
    """언더스티어 계수 K_us [rad/(m/s^2)]: 정상 선회에 필요한 조향각
    delta_ss = L/R + K_us * a_y  (K_us > 0 이면 언더스티어)."""
    return car.m * (car.lr / car.Cf - car.lf / car.Cr) / car.L


def pid_closed_loop_poles_dynamic(car, vx, Kp, Ki, Kd):
    """PID(e_y 피드백)를 동역학 오차 모델에 붙였을 때의 폐루프 극점."""
    A, B, _ = lateral_error_ss(car, vx)
    G = ct.ss(A, B, np.array([[1, 0, 0, 0]]), 0)           # delta -> e_y
    C = ct.tf([Kd, Kp, Ki], [1, 0]) if Ki else ct.tf([Kd, Kp], [1])
    return ct.poles(ct.feedback(C * G, 1))
