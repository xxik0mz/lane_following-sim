"""2단계: 동역학적 자전거 모델 — 1단계 PID가 속도와 노면에 따라 어떻게 무너지는가.

실행:  python run_stage2.py   → results/ 폴더에 05~08 그래프 저장
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import control as ct

from src.vehicle import KinematicBicycle, DynamicBicycle
from src.path import make_test_track, PathTracker
from src.controller import PIDSteering
from src import design

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("results", exist_ok=True)

DT = 0.01
Y0 = 1.0
# 1단계에서 v = 10 m/s, 기구학 모델로 설계한 PID 게인을 그대로 쓴다
KP, KI, KD = design.pid_gains(10.0, 2.7, 0.8, 2.0, 0.5)
car = DynamicBicycle()
wet = DynamicBicycle(Cf=car.Cf * 0.6, Cr=car.Cr * 0.6)   # 젖은 노면: 코너링 강성 60%
COL = {10: "#2e86ab", 20: "#f18f01", 30: "#d1495b"}

print(f"차량: m={car.m} kg, Iz={car.Iz} kg·m², lf={car.lf} m, lr={car.lr} m, Cf=Cr={car.Cf:.0f} N/rad")
print(f"언더스티어 계수 K_us = {design.understeer_gradient(car)*1000:.2f} mrad/(m/s²)")
print(f"사용 게인 (1단계 PID): Kp={KP:.4f}, Ki={KI:.4f}, Kd={KD:.4f}")


# ── 1. 기구학 vs 동역학: 같은 조향 입력에 대한 요레이트 응답 ──────────────
def yaw_rate_step(model, v, delta=np.deg2rad(2.0), T=2.0):
    if isinstance(model, KinematicBicycle):
        state = np.array([0.0, 0.0, 0.0])
    else:
        state = np.zeros(5)
    t, rs = [], []
    for k in range(int(T / DT)):
        prev_psi = state[2]
        state, _ = model.step(state, delta, v, DT)
        t.append((k + 1) * DT); rs.append((state[2] - prev_psi) / DT)
    return np.array(t), np.array(rs)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
for ax, v in zip(axs, (10, 30)):
    tk, rk = yaw_rate_step(KinematicBicycle(L=car.L), v)
    td, rd = yaw_rate_step(car, v)
    ax.plot(tk, np.rad2deg(rk), "k--", label="기구학 모델 (즉시 반응)")
    ax.plot(td, np.rad2deg(rd), color=COL[v], lw=2, label="동역학 모델 (타이어 힘이 쌓이는 시간)")
    ax.set_title(f"조향 2° 계단 입력, v = {v} m/s ({v*3.6:.0f} km/h)")
    ax.set_xlabel("시간 [s]"); ax.set_ylabel("요레이트 [deg/s]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("같은 조향에도 동역학 모델은 늦게, 그리고 속도가 높을수록 덜 돈다 (언더스티어)")
fig.tight_layout(); fig.savefig("results/05_kinematic_vs_dynamic.png", dpi=150)

# ── 2. 속도에 따른 폐루프 극점 이동 ('속도 근궤적') ───────────────────────
speeds = np.linspace(5, 35, 121)
poles = np.array([np.sort_complex(design.pid_closed_loop_poles_dynamic(car, v, KP, KI, KD)) for v in speeds])
dom = []   # 실수부가 가장 큰(가장 느린/위험한) 복소 극점
for p in poles:
    cpx = p[np.abs(p.imag) > 1e-6]
    dom.append(cpx[np.argmax(cpx.real)] if len(cpx) else p[np.argmax(p.real)])
dom = np.array(dom)
v_cross = speeds[np.argmax(dom.real > 0)] if np.any(dom.real > 0) else None
print(f"폐루프가 불안정해지는 속도: 약 {v_cross:.1f} m/s ({v_cross*3.6:.0f} km/h)" if v_cross else "모든 속도에서 안정")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
sc = ax.scatter(dom.real, dom.imag, c=speeds, cmap="viridis", s=14)
ax.scatter(dom.real, -dom.imag, c=speeds, cmap="viridis", s=14)
for v in (10, 20, 30):
    p = dom[np.argmin(abs(speeds - v))]
    ax.plot(p.real, p.imag, "o", mfc="none", mec=COL[v], ms=13, mew=2)
    ax.annotate(f"{v} m/s", (p.real, p.imag), xytext=(8, 8), textcoords="offset points", color=COL[v], fontsize=11)
ax.axvline(0, color="k", lw=1); ax.axhline(0, color="k", lw=0.5)
ax.fill_betweenx([-8, 8], 0, 1, color="#d1495b", alpha=0.08)
ax.text(0.1, -7.3, "불안정 영역\n(우반면)", color="#d1495b", fontsize=10)
ax.set_xlim(-3, 1); ax.set_ylim(-8, 8)
ax.set_xlabel("실수축"); ax.set_ylabel("허수축"); ax.grid(alpha=0.3)
plt.colorbar(sc, label="속도 v [m/s]")
ax.set_title("같은 PID, 속도만 올렸을 때 지배 극점의 이동")
fig.tight_layout(); fig.savefig("results/06_poles_vs_speed.png", dpi=150)


# ── 3. 비선형(동역학) 시뮬레이션 ──────────────────────────────────────
def simulate(model, v, T_MAX=60.0):
    path = make_test_track(straight1=80, radius=200, turn_deg=45, straight2=80)
    trk = PathTracker(path)
    pid = PIDSteering(KP, KI, KD, model.delta_max)
    state = np.array([0.0, Y0, 0.0, 0.0, 0.0])
    log = {k: [] for k in ("t", "s", "ey", "delta", "af", "ar", "kappa")}
    t = 0.0
    while t < T_MAX and not trk.finished():
        ey, _, kap, s = trk.errors(state[0], state[1], state[2])
        if abs(ey) > 4.0:           # 차선을 완전히 벗어나면 중단
            break
        delta = pid(ey, DT)
        af, ar = model.slip_angles(state, delta, v)
        for k, val in zip(log, (t, s, ey, delta, af, ar, kap)):
            log[k].append(val)
        state, _ = model.step(state, delta, v, DT)
        t += DT
    return {k: np.array(val) for k, val in log.items()}

runs = {v: simulate(car, v) for v in (10, 20, 30)}
run_wet = simulate(wet, 20)

fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
for v, lg in runs.items():
    axs[0].plot(lg["s"], lg["ey"], color=COL[v], lw=1.6, label=f"{v} m/s ({v*3.6:.0f} km/h)")
lg20 = runs[20]
axs[1].plot(lg20["s"], lg20["ey"], color=COL[20], lw=1.6, label="마른 노면 (Cf, Cr 100%)")
axs[1].plot(run_wet["s"], run_wet["ey"], color="#5c4d7d", lw=1.6, ls="--", label="젖은 노면 (Cf, Cr 60%)")
for ax in axs:
    cv = runs[10]["kappa"] > 0
    ax.fill_between(runs[10]["s"], 0, 1, where=cv, transform=ax.get_xaxis_transform(),
                    color="#f2c14e", alpha=0.18, label="곡선 구간 (R = 200 m)")
    ax.set_ylabel("횡방향 오차 e_y [m]"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-4.2, 4.2)
axs[0].set_title("1단계 PID를 동역학 모델에 그대로: 속도가 오르면 흔들리다 발산")
axs[1].set_title("v = 20 m/s에서 노면만 미끄러워져도 흔들림이 커진다")
axs[1].set_xlabel("주행 거리 [m]")
fig.tight_layout(); fig.savefig("results/07_speed_and_road.png", dpi=150)

# ── 4. 선형 타이어 가정 확인 (슬립각) ─────────────────────────────────
print("\n[성능 지표]  (주행 거리 기준, 시작 오차 1 m)")
for name, lg in [("10 m/s", runs[10]), ("20 m/s", runs[20]), ("30 m/s", runs[30]), ("20 m/s 젖은 노면", run_wet)]:
    done = "완주" if len(lg["s"]) and lg["s"][-1] > 300 else f"이탈 (s = {lg['s'][-1]:.0f} m)"
    print(f"{name:14s} 최대 |e_y| {np.abs(lg['ey']).max():.2f} m | 최대 슬립각 앞 {np.rad2deg(np.abs(lg['af']).max()):.1f}°"
          f" 뒤 {np.rad2deg(np.abs(lg['ar']).max()):.1f}° | {done}")
