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
