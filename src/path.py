"""기준 경로(차선 중앙선)와 추종 오차 계산."""
import numpy as np


def make_test_track(straight1=40.0, radius=50.0, turn_deg=90.0, straight2=40.0, ds=0.05):
    """직선 → 일정 반경 좌회전 → 직선 경로를 촘촘한 점들로 만든다.

    반환: x, y, heading(psi_ref), curvature(kappa), s(누적 거리)
    """
    xs, ys, hs, ks = [], [], [], []
    # 1) 직선
    for s in np.arange(0, straight1, ds):
        xs.append(s); ys.append(0.0); hs.append(0.0); ks.append(0.0)
    # 2) 원호 (중심 (straight1, radius), 왼쪽으로 회전)
    arc_len = radius * np.deg2rad(turn_deg)
    for s in np.arange(0, arc_len, ds):
        th = s / radius
        xs.append(straight1 + radius * np.sin(th))
        ys.append(radius - radius * np.cos(th))
        hs.append(th); ks.append(1.0 / radius)
    # 3) 직선
    th_end = np.deg2rad(turn_deg)
    x0 = straight1 + radius * np.sin(th_end)
    y0 = radius - radius * np.cos(th_end)
    for s in np.arange(0, straight2, ds):
        xs.append(x0 + s * np.cos(th_end)); ys.append(y0 + s * np.sin(th_end))
        hs.append(th_end); ks.append(0.0)
    xs, ys, hs, ks = map(np.array, (xs, ys, hs, ks))
    s_cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    return {"x": xs, "y": ys, "psi": hs, "kappa": ks, "s": s_cum}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PathTracker:
    """차량 위치에서 가장 가까운 경로 점을 찾아 오차를 계산한다.

    e_y   : 횡방향 오차 [m] (차량이 경로의 왼쪽에 있으면 +)
    e_psi : 방향 오차 [rad] (차량 방향 - 경로 방향)
    """
    def __init__(self, path):
        self.p = path
        self.idx = 0

    def errors(self, x, y, psi, window=400):
        lo, hi = self.idx, min(self.idx + window, len(self.p["x"]))
        d2 = (self.p["x"][lo:hi] - x) ** 2 + (self.p["y"][lo:hi] - y) ** 2
        self.idx = lo + int(np.argmin(d2))
        i = self.idx
        px, py, ph = self.p["x"][i], self.p["y"][i], self.p["psi"][i]
        # 경로 방향의 왼쪽 법선 벡터와의 내적 = 부호 있는 횡방향 거리
        e_y = -(x - px) * np.sin(ph) + (y - py) * np.cos(ph)
        e_psi = wrap(psi - ph)
        return e_y, e_psi, self.p["kappa"][i], self.p["s"][i]

    def finished(self):
        return self.idx >= len(self.p["x"]) - 5
