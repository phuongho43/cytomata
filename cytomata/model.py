from scipy.integrate import solve_ivp


def sim_lov(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        v = 1 if u == 0 else 0
        [Ai0, Aa0, B0, AB0] = y0
        [Ai, Aa, B, AB] = y
        kl = params['kl']
        kd = params['kd']
        kb = params['kb']
        dAi = -u*kl*Ai + v*kd*Aa - kb*Ai*B
        dAa = u*kl*Ai - v*kd*Aa + u*kl*AB
        dB = -kb*Ai*B + u*kl*AB
        dAB = kb*Ai*B - u*kl*AB
        return [dAi, dAa, dB, dAB]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, dense_output=True,
        method='LSODA', rtol=1e-6, atol=1e-6, max_step=1)
    return t, result.sol(t)


def sim_ilid(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        v = 1 if u == 0 else 0
        [Ai0, Aa0, B0, AB0] = y0
        [Ai, Aa, B, AB] = y
        kl = params['kl']
        kd = params['kd']
        kb = params['kb']
        dAi = -u*kl*Ai + v*kd*Aa + v*kd*AB
        dAa = u*kl*Ai - v*kd*Aa - kb*Aa*B
        dB = -kb*Aa*B + v*kd*AB
        dAB = kb*Aa*B - v*kd*AB
        return [dAi, dAa, dB, dAB]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, dense_output=True,
        method='LSODA', rtol=1e-6, atol=1e-6, max_step=1)
    return t, result.sol(t)


def sim_sparser(t, y0, uf, params):
    def model(t, y):
        u = uf(t)
        v = 1 if u == 0 else 0
        [Ai0, Aa0, Bi0, Ba0, C0, AiBi0, AiBa0, BaC0, AiBaC0] = y0
        [Ai, Aa, Bi, Ba, C, AiBi, AiBa, BaC, AiBaC] = y
        kl1 = params['kl1']
        kd1 = params['kd1']
        kb1 = params['kb1']
        kl2 = params['kl2']
        kd2 = params['kd2']
        kb2 = params['kb2']
        dAi = -u*kl1*Ai + v*kd1*Aa - kb1*Ai*Bi - kb1*Ai*Ba - kb1*Ai*BaC
        dAa = u*kl1*Ai - v*kd1*Aa + u*kl1*AiBi + u*kl1*AiBa + u*kl1*AiBaC
        dBi = -u*kl2*Bi + v*kd2*Ba - kb1*Ai*Bi + v*kd2*BaC
        dBa = u*kl2*Bi - v*kd2*Ba - kb1*Ai*Ba - kb2*Ba*C + u*kl1*AiBi + u*kl1*AiBa
        dC = -kb2*Ba*C - kb2*AiBa*C + v*kd2*BaC + v*kd2*(AiBaC-AiBaC0)
        dAiBi = kb1*Ai*Bi -u*kl1*AiBi + v*kd2*(AiBaC-AiBaC0)
        dAiBa = kb1*Ai*Ba - kb2*AiBa*C - u*kl1*AiBa
        dBaC = kb2*Ba*C - kb1*Ai*BaC - v*kd2*BaC + u*kl1*AiBaC
        dAiBaC = kb2*AiBa*C + kb1*Ai*BaC - v*kd2*(AiBaC-AiBaC0) - u*kl1*AiBaC
        return [dAi, dAa, dBi, dBa, dC, dAiBi, dAiBa, dBaC, dAiBaC]
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, dense_output=True,
        method='LSODA', rtol=1e-6, atol=1e-6, max_step=1)
    return t, result.sol(t)


def gen_model(params):
    [kra, krb, krc,
     kua, kub, kuc,
     kta, ktb, ktc,
     kba, kab, kac,
     kca, kcb, kbc,
    ] = params
    kua = -kua if kra > 0 else kua
    kub = -kub if krb > 0 else kub
    kuc = -kuc if krc > 0 else kuc
    Aa0 = 1 if kra > 0 else 0
    Ba0 = 1 if krb > 0 else 0
    Ca0 = 1 if krc > 0 else 0
    Ai0 = 0 if kra > 0 else 1
    Bi0 = 0 if krb > 0 else 1
    Ci0 = 0 if krc > 0 else 1
    y0 = [Ai0, Aa0, Bi0, Ba0, Ci0, Ca0]
    Aa = f"Aat=max(Aa-{kta},0)"
    Ba = f"Bat=max(Ba-{ktb},0)"
    Ca = f"Cat=max(Ca-{ktc},0)"
    dAa = [
        f"+{kra}*A{'i' if kra>0 else 'a'}{'*v' if kua!=0 else ''}",
        f"+{kua}*u*A{'i' if kua>0 else 'a'}",
        f"+{kba}*Bat*A{'i' if kba>0 else 'a'}",
        f"+{kca}*Cat*A{'i' if kca>0 else 'a'}",
    ]
    dBa = [
        f"+{krb}*B{'i' if krb>0 else 'a'}{'*v' if kub!=0 else ''}",
        f"+{kub}*u*B{'i' if kub>0 else 'a'}",
        f"+{kab}*Aat*B{'i' if kab>0 else 'a'}",
        f"+{kcb}*Cat*B{'i' if kcb>0 else 'a'}",
    ]
    dCa = [
        f"+{krc}*C{'i' if krc>0 else 'a'}{'*v' if kuc!=0 else ''}",
        f"+{kuc}*u*C{'i' if kuc>0 else 'a'}",
        f"+{kac}*Aat*C{'i' if kac>0 else 'a'}",
        f"+{kbc}*Bat*C{'i' if kbc>0 else 'a'}",
    ]
    dAi = ["-" + s for s in dAa]
    dBi = ["-" + s for s in dBa]
    dCi = ["-" + s for s in dCa]
    dAa = "dAa=" + "".join(dAa)
    dBa = "dBa=" + "".join(dBa)
    dCa = "dCa=" + "".join(dCa)
    dAi = "dAi=" + "".join(dAi)
    dBi = "dBi=" + "".join(dBi)
    dCi = "dCi=" + "".join(dCi)
    model_eqs = "\n    ".join([Aa, Ba, Ca, dAi, dAa, dBi, dBa, dCi, dCa])
    model_str = f"""def model(t, y, uf):
    u = uf(t)
    v = 1 if u == 0 else 0
    [Ai, Aa, Bi, Ba, Ci, Ca] = y
    {model_eqs}
    return [dAi, dAa, dBi, dBa, dCi, dCa]
    """
    # print(model_str)
    model_fnc = compile(model_str, 'model_fnc', 'exec')
    exec(model_fnc, globals())
    return model, y0


def sim_model(t, y0, uf, model):
    result = solve_ivp(
        fun=model, t_span=[t[0], t[-1]], y0=y0, dense_output=True,
        method='LSODA', rtol=1e-6, atol=1e-6, max_step=1, args=(uf,))
    return t, result.sol(t)


def sim_damped_osc(t):
    y0 = [1, 0]
    gamma = 0.1
    omega = 6
    def model(t, y):
        [x, v] = y
        dx = v
        dv = -gamma*v - (omega**2)*x
        return [dx, dv]
    result = solve_ivp(fun=model, t_span=[t[0], t[-1]], y0=y0, dense_output=True,
        method='LSODA', rtol=1e-6, atol=1e-6, max_step=1)
    return t, result.sol(t)[0]
