"""1단계: 기구학적 자전거 모델 + P / PD / PID 조향 제어 비교.

실행:  python run_stage1.py   → results/ 폴더에 그래프 저장
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import control as ct

from src.vehicle import KinematicBicycle
from src.path import make_test_track, PathTracker
from src.controller import PIDSteering
from src import design

plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("results", exist_ok=True)

# ── 1. 조건 ────────────────────────────────────────────
V, L = 10.0, 2.7          # 36 km/h, 일반 승용차 축간거리
DT, T_MAX = 0.01, 25.0
Y0 = 1.0                  # 출발할 때 차선 중앙에서 1 m 왼쪽으로 벗어나 있음
ZETA, WN, P3 = 0.8, 2.0, 0.5   # 목표: 감쇠비 0.8, 고유진동수 2 rad/s

# ── 2. 게인 설계 (선형 모델) ──────────────────────────
G, a = design.plant_tf(V, L)
Kp_pd, Kd_pd = design.pd_gains(V, L, ZETA, WN)
Kp_pid, Ki_pid, Kd_pid = design.pid_gains(V, L, ZETA, WN, P3)
cases = {
    "P":   (Kp_pd, 0.0, 0.0),
    "PD":  (Kp_pd, 0.0, Kd_pd),
    "PID": (Kp_pid, Ki_pid, Kd_pid),
}
print(f"플랜트 G(s) = {a:.2f} / s^2   (a = v^2/L)")
for name, (kp, ki, kd) in cases.items():
    poles = design.closed_loop_poles(V, L, kp, ki, kd)
    print(f"{name:4s} Kp={kp:.4f} Ki={ki:.4f} Kd={kd:.4f}  극점={np.round(poles, 3)}")
ok, coeffs = design.routh_ok(a, Kp_pid, Ki_pid, Kd_pid)
print(f"PID Routh 조건 Ki < a*Kp*Kd : {Ki_pid:.4f} < {a*Kp_pid*Kd_pid:.4f} → {'안정' if ok else '불안정'}")
kappa = 1 / 50.0
print(f"PD 곡선 정상상태 오차 예측: e_y = -L*kappa/Kp = {-L*kappa/Kp_pd:.3f} m")


# ── 3. 비선형 시뮬레이션 ───────────────────────────────
def simulate(kp, ki, kd):
    path = make_test_track()
    car, trk = KinematicBicycle(L), PathTracker(path)
    pid = PIDSteering(kp, ki, kd, car.delta_max)
    state = np.array([0.0, Y0, 0.0])
    log = {k: [] for k in ("t", "x", "y", "ey", "epsi", "delta", "kappa")}
    t = 0.0
    while t < T_MAX and not trk.finished():
        ey, epsi, kap, _ = trk.errors(*state)
        delta = pid(ey, DT)
        for k, val in zip(log, (t, state[0], state[1], ey, epsi, delta, kap)):
            log[k].append(val)
        state, _ = car.step(state, delta, V, DT)
        t += DT
    return path, {k: np.array(v) for k, v in log.items()}


results = {name: simulate(*g) for name, g in cases.items()}
path = results["PD"][0]
colors = {"P": "#d1495b", "PD": "#2e86ab", "PID": "#3bb273"}

# ── 4. 그래프 ─────────────────────────────────────────
# (1) 주행 궤적
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(path["x"], path["y"], "k--", lw=1.2, label="차선 중앙(기준 경로)")
for name, (_, lg) in results.items():
    ax.plot(lg["x"], lg["y"], color=colors[name], lw=1.6, label=name)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("주행 궤적 (v = 36 km/h, 시작 오차 1 m)")
ax.legend(); fig.tight_layout(); fig.savefig("results/01_trajectory.png", dpi=150)

# (2) 횡방향 오차와 조향각
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
lg_ref = results["PD"][1]
curve = lg_ref["kappa"] > 0
for ax in axs:
    ax.fill_between(lg_ref["t"], 0, 1, where=curve, transform=ax.get_xaxis_transform(),
                    color="#f2c14e", alpha=0.18, label="곡선 구간")
for name, (_, lg) in results.items():
    axs[0].plot(lg["t"], lg["ey"], color=colors[name], label=name)
    axs[1].plot(lg["t"], np.rad2deg(lg["delta"]), color=colors[name], label=name)
axs[0].axhline(-L * kappa / Kp_pd, color=colors["PD"], ls=":", lw=1,
               label=f"PD 정상상태 오차 예측 {-L*kappa/Kp_pd:.2f} m")
axs[0].set_ylabel("횡방향 오차 e_y [m]"); axs[0].set_ylim(-1.6, 1.6)
axs[1].set_ylabel("조향각 δ [deg]"); axs[1].set_xlabel("시간 [s]")
for ax in axs:
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
axs[0].set_title("P는 진동, PD는 곡선에서 오차가 남고, PID는 0으로 수렴")
fig.tight_layout(); fig.savefig("results/02_error_steering.png", dpi=150)

# (3) 근궤적: P 제어 vs PD 제어
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
loops = {
    "P 제어: L(s) = K·a/s²": G,
    f"PD 제어: L(s) = K·a(s + {Kp_pd/Kd_pd:.2f})/s²": G * ct.tf([1, Kp_pd / Kd_pd], [1]),
}
for ax, (title, Ls) in zip(axs, loops.items()):
    rl = ct.root_locus_map(Ls, gains=np.linspace(0, 0.6, 600))
    roots = rl.loci
    ax.plot(roots.real, roots.imag, ".", ms=1.5, color="#2e86ab")
    ax.plot([0], [0], "kx", ms=9, mew=2, label="개루프 극점 (s=0 중근)")
    zs = ct.zeros(Ls)
    if len(zs):
        ax.plot(zs.real, zs.imag, "o", mfc="none", mec="k", ms=8, label="영점")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-6, 1); ax.set_ylim(-5, 5)
    ax.set_title(title, fontsize=10); ax.set_xlabel("실수축"); ax.set_ylabel("허수축")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
fig.suptitle("근궤적: P만 쓰면 극점이 허수축 위 → 영원히 진동, D를 더하면 좌반면으로 끌려옴")
fig.tight_layout(); fig.savefig("results/03_root_locus.png", dpi=150)

# (4) 검증: 선형 모델 예측 vs 비선형 시뮬레이션 (직선 구간, PD)
A = np.array([[0, 1], [-a * Kp_pd, -a * Kd_pd]])
sys_cl = ct.ss(A, np.zeros((2, 1)), np.array([[1, 0]]), 0)
t_lin = np.linspace(0, 3.5, 351)
resp = ct.initial_response(sys_cl, T=t_lin, X0=[Y0, 0])
lg = results["PD"][1]
m = lg["t"] <= 3.5
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(t_lin, resp.outputs, "k-", lw=2.5, alpha=0.35, label="선형 모델 예측 (python-control)")
ax.plot(lg["t"][m], lg["ey"][m], color=colors["PD"], lw=1.4, label="비선형 시뮬레이션")
ax.set_xlabel("시간 [s]"); ax.set_ylabel("e_y [m]"); ax.grid(alpha=0.3); ax.legend()
ax.set_title("모델 검증: 설계에 쓴 선형 모델이 실제(비선형) 응답을 잘 예측하는가")
fig.tight_layout(); fig.savefig("results/04_linear_vs_nonlinear.png", dpi=150)

# ── 5. 성능 지표 ──────────────────────────────────────
print("\n[성능 지표]")
for name, (_, lg) in results.items():
    st = lg["t"] < 4.0                      # 직선 구간 (시작 오차 수렴)
    cv = lg["kappa"] > 0                    # 곡선 구간
    cv_late = cv & (np.cumsum(cv) * DT > 4.0)  # 곡선 진입 4초 이후
    os_pct = max(0.0, -lg["ey"][st].min()) / Y0 * 100
    print(f"{name:4s} 직선 오버슈트 {os_pct:5.1f}%  | 곡선 후반 평균 오차 {lg['ey'][cv_late].mean():+.3f} m"
          f"  | 최대 조향각 {np.rad2deg(np.abs(lg['delta']).max()):.1f}°")
