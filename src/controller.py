"""횡방향 오차 e_y를 0으로 만드는 PID 조향 제어기.

    delta = -(Kp * e_y + Ki * ∫e_y dt + Kd * de_y/dt)

- 미분항: 측정값을 그대로 미분하면 잡음이 커지므로 1차 저역통과 필터를 거친다.
- 적분항: 조향각이 한계(포화)에 걸린 동안에는 적분을 멈춘다(anti-windup).
"""
import numpy as np


class PIDSteering:
    def __init__(self, Kp, Ki=0.0, Kd=0.0, delta_max=np.deg2rad(30.0), d_filter_tau=0.05):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.delta_max = delta_max
        self.tau = d_filter_tau
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_e = None
        self.d_filt = 0.0

    def __call__(self, e_y, dt):
        if self.prev_e is None:
            self.prev_e = e_y
        d_raw = (e_y - self.prev_e) / dt
        alpha = dt / (self.tau + dt)
        self.d_filt += alpha * (d_raw - self.d_filt)
        self.prev_e = e_y

        u_unsat = -(self.Kp * e_y + self.Ki * (self.integral + e_y * dt) + self.Kd * self.d_filt)
        u = np.clip(u_unsat, -self.delta_max, self.delta_max)
        if u == u_unsat:  # 포화되지 않았을 때만 적분 (anti-windup)
            self.integral += e_y * dt
        return u
