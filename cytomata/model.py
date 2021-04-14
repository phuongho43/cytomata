from scikits.odes import ode
from scikits.odes.odeint import odeint
from scipy.integrate import solve_ivp


def sim_itranslo(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        ku = params['ku']
        kf = params['kf']
        kr = params['kr']
        a = params['a']
        dy[0] = -(u*ku + kf)*y[0] + kr*y[1]
        dy[1] = -a*dy[0]
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size':1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_idimer(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        ku = params['ku']
        kf = params['kf']
        kr = params['kr']
        dy[0] = -(u*ku + kf)*y[0]*y[1] + kr*y[2]
        dy[1] = -(u*ku + kf)*y[0]*y[1] + kr*y[2]
        dy[2] = (u*ku + kf)*y[0]*y[1] - kr*y[2]
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size': 1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_TF(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        k1u = params['k1u']
        k1f = params['k1f']
        k1r = params['k1r']
        a = params['a']
        k2u = params['k2u']
        k2f = params['k2f']
        k2r = params['k2r']
        dy[0] = -(u*k1u + k1f)*y[0] + k1r*y[1]
        dy[1] = -(u*k2u + k2f)*y[1]*y[2] + k2r*y[3]
        dy[2] = -a*dy[0] - dy[1]
        dy[3] = -dy[1]
    options = {'rtol': 1e-10, 'atol': 1e-15}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_iexpress(t, y0, xf, params):
    def model(t, y, dy):
        X = xf(t)
        ka = params['ka']
        kb = params['kb']
        kc = params['kc']
        n = params['n']
        kf = params['kf']
        kg = params['kg']
        dy[0] = (ka*X**n)/(kb**n + X**n) - kc*y[0]
        dy[1] = kf*y[0] - kg*y[1]
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size':1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_ssl(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        [Ai, Bi, C, Aa, Ba, CA, CB] = y
        kua = params['ku']
        kra = params['kra']
        kaa = params['kaa']
        kda = params['kd']
        kub = params['ku']
        krb = params['krb']
        kab = params['kab']
        kdb = params['kd']
        dy[0] = -u*kua*Ai + kra*Aa
        dy[1] = -u*kub*Bi + krb*Ba
        dy[2] = -kaa*Aa*C - kab*Ba*C + kda*CA + kdb*CB
        dy[3] = u*kua*Ai - kra*Aa + kda*CA - kaa*Aa*C
        dy[4] = u*kub*Bi - krb*Ba + kdb*CB - kab*Ba*C
        dy[5] = -kda*CA + kaa*Aa*C
        dy[6] = -kdb*CB + kab*Ba*C
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size':1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_ssl_cn(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        [L, C, LC, H, N, HN] = y
        kcu = params['kcu']
        kcf = params['kcf']
        kcr = params['kcr']
        kcn = params['kcn']
        knc = params['knc']
        knu = params['knu']
        knf = params['knf']
        knr = params['knr']
        dy[0] = -(u*kcu + kcf)*L*C + kcr*LC
        dy[1] = -(u*kcu + kcf)*L*C + kcr*LC - kcn*C + knc*N
        dy[2] = -kcr*LC + (u*kcu + kcf)*L*C
        dy[3] = -(u*knu + knf)*H*N + knr*HN
        dy[4] = -(u*knu + knf)*H*N + knr*HN - knc*N + kcn*C
        dy[5] = -knr*HN + (u*knu + knf)*H*N
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size':1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_CaM_M13(t, y0, Cf):
    import numpy as np
    def model(t, y, dy):
        [CaM, CaM_2C, CaM_P, CaM_2C_P, CaM_4C_P] = y
        C = Cf(t)
        CaM_tot = 10
        P_tot = 10
        k1 = 65
        k2 = 850
        k3 = 6
        k4 = 12
        k11 = 65
        k21 = 425
        k31 = 6
        k41 = 0.06
        k51 = 46
        k61 = 348
        k71 = 46
        k81 = 0.008
        k91 = 46
        k101 = 0.0012
        Pb = CaM_P + CaM_2C_P + CaM_4C_P
        CaM_4C = CaM_tot - (Pb + CaM_2C + CaM)
        Pf = P_tot - Pb
        dy[0] = k61*CaM_P + k4*CaM_2C - CaM*(k3*C**2 + k51*Pf)
        dy[1] = k81*CaM_2C_P + k3*CaM*C**2 + k2*CaM_4C - CaM_2C*(k1*C**2 + k71*Pf + k4)
        dy[2] = k51*CaM*Pf + k41*CaM_2C_P - CaM_P*(k31*C**2 + k61)
        dy[3] = k31*CaM_P*C**2 + k71*CaM_2C*Pf + k21*CaM_4C_P - CaM_2C_P*(k41 + k11*C**2 + k81)
        dy[4] = k91*CaM_4C*Pf + k11*CaM_2C*C**2 - CaM_4C_P*(k21 + k101)
    options = {'rtol': 1e-6, 'atol': 1e-12, 'max_step_size':1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_ilid(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        kf = params['kf']
        kr = params['kr']
        dy[0] = kr*y[1] - u*kf*y[0]
        dy[1] = u*kf*y[0] - kr*y[1]
    options = {'rtol': 1e-3, 'atol': 1e-6, 'max_step_size': 1}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y


def sim_fresca(t, y0, uf, params):
    def model(t, y, dy):
        u = uf(t)
        k1f = params['k1f']
        k1r = params['k1r']
        k2f = params['k2f']
        k2r = params['k2r']
        dy[0] = k1r*y[1] - u*k1f*y[0]*y[4]
        dy[1] = u*k1f*y[0]*y[4] - k1r*y[1]
        dy[2] = k2r*y[3] - u*k2f*y[2]*y[4]
        dy[3] = u*k2f*y[2]*y[4] - k2r*y[3]
        dy[4] = k1r*y[1] + k2r*y[3] - u*k1f*y[0]*y[4] - u*k2f*y[2]*y[4]
    options = {'rtol': 1e-12, 'atol': 1e-15, 'max_step_size': 1, 'max_steps': 1e4}
    solver = ode('cvode', model, old_api=False, **options)
    solution = solver.solve(t, y0)
    return solution.values.t, solution.values.y



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