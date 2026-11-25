"""기구학적 자전거 모델 (Kinematic Bicycle Model).

차량의 네 바퀴를 앞·뒤 바퀴 하나씩으로 합쳐 '자전거'로 보는 모델이다.
기준점은 뒷바퀴 축 중심이고, 미끄러짐(타이어 슬립)은 없다고 가정한다.

    x'   = v cos(psi)
    y'   = v sin(psi)
    psi' = (v / L) tan(delta)

x, y  : 뒷바퀴 축 중심의 위치 [m]
psi   : 차량 진행 방향(요 각) [rad]
v     : 속도 [m/s] (이번 단계에서는 일정)
L     : 축간거리(wheelbase) [m]
delta : 앞바퀴 조향각 [rad] — 제어 입력
"""
import numpy as np


class KinematicBicycle:
    def __init__(self, L=2.7, delta_max=np.deg2rad(30.0)):
        self.L = L
        self.delta_max = delta_max  # 조향각 한계(포화)

    def deriv(self, state, delta, v):
        x, y, psi = state
        return np.array([
            v * np.cos(psi),
            v * np.sin(psi),
            v / self.L * np.tan(delta),
        ])

    def step(self, state, delta, v, dt):
        """4차 룽게-쿠타(RK4)로 미분방정식을 dt만큼 적분한다."""
        delta = np.clip(delta, -self.delta_max, self.delta_max)
        k1 = self.deriv(state, delta, v)
        k2 = self.deriv(state + 0.5 * dt * k1, delta, v)
        k3 = self.deriv(state + 0.5 * dt * k2, delta, v)
        k4 = self.deriv(state + dt * k3, delta, v)
        return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), delta


class DynamicBicycle:
    """동역학적 자전거 모델 (Dynamic Bicycle Model, 선형 타이어).

    기구학 모델은 '바퀴가 가리키는 방향으로 그대로 간다'고 가정했다.
    실제 타이어는 옆으로 조금 미끄러지면서(슬립각 alpha) 그 크기에 비례하는 횡력을 낸다.

        alpha_f = delta - (vy + lf*r)/vx      앞바퀴 슬립각
        alpha_r =       - (vy - lr*r)/vx      뒷바퀴 슬립각
        Fyf = Cf * alpha_f,  Fyr = Cr * alpha_r   (선형 타이어, Cf·Cr = 코너링 강성)

    뉴턴 법칙 (차체 좌표계):
        m  (vy' + vx*r) = Fyf + Fyr          횡방향 힘 평형
        Iz  r'          = lf*Fyf - lr*Fyr    요 모멘트 평형
    전역 좌표 (무게중심 기준):
        X' = vx cos(psi) - vy sin(psi),  Y' = vx sin(psi) + vy cos(psi),  psi' = r

    상태: [X, Y, psi, vy, r]
    """
    def __init__(self, m=1500.0, Iz=2500.0, lf=1.2, lr=1.5, Cf=80000.0, Cr=80000.0,
                 delta_max=np.deg2rad(30.0)):
        self.m, self.Iz, self.lf, self.lr = m, Iz, lf, lr
        self.Cf, self.Cr = Cf, Cr
        self.L = lf + lr
        self.delta_max = delta_max

    def slip_angles(self, state, delta, vx):
        _, _, _, vy, r = state
        return delta - (vy + self.lf * r) / vx, -(vy - self.lr * r) / vx

    def deriv(self, state, delta, vx):
        X, Y, psi, vy, r = state
        af, ar = self.slip_angles(state, delta, vx)
        Fyf, Fyr = self.Cf * af, self.Cr * ar
        return np.array([
            vx * np.cos(psi) - vy * np.sin(psi),
            vx * np.sin(psi) + vy * np.cos(psi),
            r,
            (Fyf + Fyr) / self.m - vx * r,
            (self.lf * Fyf - self.lr * Fyr) / self.Iz,
        ])

    def step(self, state, delta, vx, dt):
        delta = np.clip(delta, -self.delta_max, self.delta_max)
        k1 = self.deriv(state, delta, vx)
        k2 = self.deriv(state + 0.5 * dt * k1, delta, vx)
        k3 = self.deriv(state + 0.5 * dt * k2, delta, vx)
        k4 = self.deriv(state + dt * k3, delta, vx)
        return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), delta
