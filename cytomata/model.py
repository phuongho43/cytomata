from scipy.integrate import solve_ivp


# def sim_idimer(t, y0, uf, params):
#     def model(t, y):
#         u = uf(t)
#         v = 1 if u == 0 else 0
#         [A0, B0, AB0] = y0
#         [A, B, AB] = y
#         kl = params['kl']
#         kd = params['kd']
#         dAB = u*kl*A*B - v*kd*(AB-AB0)
#         dA = -dAB
#         dB = dA
#         return [dA, dB, dAB]
#     result = solve_ivp(
#         fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
#         method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
#     return result.t, result.y


def sim_idimer(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        v = 1 if u == 0 else 0
        [Ai, Aa, B, AiB, AaB] = y
        kl = params['kl']
        kd = params['kd']
        kif = params['kif']
        kaf = params['kaf']
        kr = params['kr']
        dAi = v*kd*Aa - u*kl*Ai + kr*AiB - kif*Ai*B
        dAa = u*kl*Ai - v*kd*Aa + kr*AaB - kaf*Aa*B
        dB = kr*AiB - kif*Ai*B + kr*AaB - kaf*Aa*B
        dAiB = v*kd*AaB - u*kl*AiB + kif*Ai*B - kr*AiB
        dAaB = u*kl*AiB - v*kd*AaB + kaf*Aa*B - kr*AaB
        return [dAi, dAa, dB, dAiB, dAaB]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    return result.t, result.y


def sim_idissoc(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        v = 1 if u == 0 else 0
        [A0, B0, AB0] = y0
        [A, B, AB] = y
        kl = params['kl']
        kd = params['kd']
        dAB = -u*kl*AB + v*kd*(A-A0)*(B)
        dA = -dAB
        dB = dA
        return [dA, dB, dAB]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    return result.t, result.y


def sim_ifate(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        v = 1 if u == 0 else 0
        [A0, B0, C0, D0, AB0, BC0, CD0, BCD0] = y0
        [A, B, C, D, AB, BC, CD, BCD] = y
        kl1 = params['kl1']
        kd1 = params['kd1']
        kl2 = params['kl2']
        kd2 = params['kd2']
        kl3 = params['kl3']
        kd3 = params['kd3']
        kl4 = kl3
        kd4 = kd3
        kl5 = kl2
        kd5 = kd2
        dA = v*kd1*(AB-AB0) - u*kl1*A*B
        dB = v*kd1*(AB-AB0) - u*kl1*A*B + v*kd3*(BC-BC0) - u*kl3*C*B + v*kd4*(BCD-BCD0) - u*kl4*CD*B
        dC = u*kl2*CD - v*kd2*C*(D-D0) + v*kd3*(BC-BC0) - u*kl3*C*B
        dD = u*kl2*CD - v*kd2*C*(D-D0) + u*kl5*BCD - v*kd5*BC*(D-D0)
        dAB = u*kl1*A*B - v*kd1*(AB-AB0)
        dBC = u*kl3*C*B - v*kd3*(BC-BC0) + u*kl5*BCD - v*kd5*BC*(D-D0)
        dCD = v*kd2*C*(D-D0) - u*kl2*CD + v*kd4*(BCD-BCD0) - u*kl4*CD*B
        dBCD = u*kl4*CD*B - v*kd4*(BCD-BCD0) + v*kd5*BC*(D-D0) - u*kl5*BCD
        return [dA, dB, dC, dD, dAB, dBC, dCD, dBCD]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, t_eval=t,
        method='LSODA', rtol=1e-3, atol=1e-6, max_step=1)
    return result.t, result.y