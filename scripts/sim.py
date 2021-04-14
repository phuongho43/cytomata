import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def sim_idimer(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        [A, B, AB] = y
        kf = params['kf']
        kr = params['kr']
        dA = kr*AB - u*kf*A*B
        dB = kr*AB - u*kf*A*B
        dAB = u*kf*A*B - kr*AB
        return [dA, dB, dAB]
    results = solve_ivp(
        fun=model,
        t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    t = results.t
    A = results.y[0]
    B = results.y[1]
    AB = results.y[2]
    return t, A, B, AB


t = np.arange(0, 300, 1)
u = np.zeros_like(t)
u[60] = 1
uf = interp1d(t, u, bounds_error=False, fill_value=0)
ui = [uf(ti) for ti in t]
y0 = [100, 100, 1]
params = {
    'kf': 0.1,
    'kr': 0.01
}

t, A, B, AB = sim_idimer(t, y0, uf, params)
plt.plot(t, AB)
plt.show()