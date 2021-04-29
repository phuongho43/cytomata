from scipy.integrate import solve_ivp


def sim_idimer(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        [A, B, AB] = y
        ku = params['ku']
        kf = params['kf']
        kr = params['kr']
        dA = kr*AB - (u*ku + kf)*A*B
        dB = kr*AB - (u*ku + kf)*A*B
        dAB = (u*ku + kf)*A*B - kr*AB
        return [dA, dB, dAB]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    return result.t, result.y


def sim_itrans(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        [C, N] = y
        ku = params['ku']
        kf = params['kf']
        kr = params['kr']
        dC = kr*N - (u*ku + kf)*C
        dN = (u*ku + kf)*C - kr*N
        return [dC, dN]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    return result.t, result.y


def sim_express(t, y0, xf, params):
    def model(t, y, dy):
        X = xf(t)
        [R, P] = y
        ka = params['ka']
        kb = params['kb']
        kc = params['kc']
        kd = params['kd']
        n = params['n']
        kf = params['kf']
        kg = params['kg']
        dR = ka + (kb*X**n)/(kc**n + X**n) - kd*R
        dP = kf*R - kg*P
        return [dR, dP]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    return result.t, result.y


def sim_fresca(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        [A1, A2, B, A1B, A2B] = y
        ku1 = params['ku1']
        kf1 = params['kf1']
        kr1 = params['kr1']
        ku2 = params['ku2']
        kf2 = params['kf2']
        kr2 = params['kr2']
        dA1 = kr1*A1B - (u*ku1 + kf1)*A1*B
        dA2 = kr2*A2B - (u*ku2 + kf2)*A2*B
        dB = kr1*A1B + kr2*A2B - (u*ku1 + kf1)*A1*B - (u*ku2 + kf2)*A2*B
        dA1B = (u*ku1 + kf1)*A1*B - kr1*A1B
        dA2B = (u*ku2 + kf2)*A2*B - kr2*A2B
        return [dA1, dA2, dB, dA1B, dA2B]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    return result.t, result.y