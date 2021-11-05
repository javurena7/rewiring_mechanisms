import sympy as sm


def fixed_points(c, na, sa, sb):
    paa, pbb = sm.symbols('paa, pbb', negative=False)

    maa = (c*.5*(paa + 1 - pbb) + (1-c)*na)*sa
    mab = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*(1-sa)
    mba = (c*.5*(paa + 1 - pbb) + (1-c)*na)*(1-sb)
    mbb = (c*.5*(pbb + 1 - paa) + (1-c)*(1-na))*sb

    taa = 2*paa / (paa + 1 - pbb)
    tbb =  2*pbb / (pbb + 1 - paa)

    Paa = na*maa - na*taa*(maa + mab)
    Pbb = (1-na)*mbb - (1-na)*tbb*(mba + mbb)

    PaaEqual = sm.Eq(Paa, 0)
    PbbEqual = sm.Eq(Pbb, 0)

    equilibria = sm.solve((PaaEqual, PbbEqual), paa, pbb)
    equilibria = check_solutions(equilibria)
    equilibria = p_to_t(equilibria)

    return equilibria

def check_solutions(equilibria):
    eqs = []
    for sol in equilibria:
        s1 = sol[0]
        s2 = sol[1]
        if s1.is_real and s2.is_real:
            if valid_sol(s1, s2):
                eqs.append((s1, s2))
        elif sm.im(s1) < 1.e-8 and sm.im(s2) < 1.e-8:
            s1 = sm.re(s1); s2 = sm.re(s2)
            if valid_sol(s1, s2):
                eqs.append((s1, s2))
    return eqs

def valid_sol(s1, s2):
    s1_val = number_in_range(s1)
    s2_val = number_in_range(s2)
    s_val = number_in_range(s1 + s2)
    if s1_val and s2_val and s_val:
        return True
    else:
        return False

def number_in_range(num):
    if (num >= 0) and (num <= 1):
        return True
    else:
        return False

def p_to_t(equilibria):
    ts = []
    for ps in equilibria:
        pa, pb = ps[0], ps[1]
        taa = 2*pa / (pa + 1 - pb)
        tbb = 2*pb / (pb + 1 - pa)
        ts.append((taa, tbb))
    return ts

