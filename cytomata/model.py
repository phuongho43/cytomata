from scipy.integrate import solve_ivp


# def sim_idimer(t, y0, uf, params):
#     def model(t, y):
#         u = uf(t)
#         [A, B, AB] = y
#         kf = params['kf']
#         kr = params['kr']
#         dAB = u*kf*A*B - kr*AB
#         dA = -dAB
#         dB = -dAB
#         return [dA, dB, dAB]
#     result = solve_ivp(
#         fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
#         method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
#     return result.t, result.y


# def sim_fresca(t, y0, uf, params):
#     def model(t, y):
#         u = uf(t)
#         [A1, A2, B, A1B, A2B] = y
#         kf = params['kf']
#         kr1 = params['kr1']
#         kr2 = params['kr2']
#         dA1B = u*kf*A1*B - kr1*A1B + u*kf*A1*A2B - u*kf*A2*A1B
#         dA2B = u*kf*A2*B - kr2*A2B + u*kf*A2*A1B - u*kf*A1*A2B
#         dA1 = -dA1B
#         dA2 = -dA2B
#         dB = -u*kf*A1*B + kr1*A1B - u*kf*A2*B + kr2*A2B
#         return [dA1, dA2, dB, dA1B, dA2B]
#     result = solve_ivp(
#         fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
#         method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
#     return result.t, result.y


def sim_idimer(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        v = 1 if u == 0 else 0
        [Ai, Aa, B, AaB] = y
        kl = params['kl']
        kd = params['kd']
        kf = params['kf']
        kr = params['kr']
        dAi = v*kd*Aa - u*kl*Ai
        dAaB = kf*Aa*B - kr*AaB
        dAa = u*kl*Ai - v*kd*Aa + kr*AaB - kf*Aa*B
        dB = kr*AaB - kf*Aa*B
        return [dAi, dAa, dB, dAaB]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    return result.t, result.y


def sim_fresca(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        v = 1 if u == 0 else 0
        [Ai1, Ai2, Aa1, Aa2, B, Aa1B, Aa2B] = y
        kl = params['kl']
        kd1 = params['kd1']
        kd2 = params['kd2']
        kf = params['kf']
        kr = params['kr']
        dAi1 = v*kd1*Aa1 - u*kl*Ai1
        dAi2 = v*kd2*Aa2 - u*kl*Ai2
        dAa1B = kf*Aa1*B - kr*Aa1B
        dAa2B = kf*Aa2*B - kr*Aa2B
        dAa1 = u*kl*Ai1 - v*kd1*Aa1 + kr*Aa1B - kf*Aa1*B
        dAa2 = u*kl*Ai2 - v*kd2*Aa2 + kr*Aa2B - kf*Aa2*B
        dB = kr*Aa1B - kf*Aa1*B + kr*Aa2B - kf*Aa2*B
        return [dAi1, dAi2, dAa1, dAa2, dB, dAa1B, dAa2B]
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