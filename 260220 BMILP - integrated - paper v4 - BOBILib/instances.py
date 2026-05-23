import numpy as np

from BMILP import *

def instance_CFL(n, m, flag_MILP):
    '''
    Capacitated Facility Location Interdiction Problem
        Parameters:
            n: num_facility
            m: num_customer
            B: interdiction budget
            cc: reconstruction cost
            c: transportation cost
            d: demand
            C: facility capacity
            CC: reconstructed facility capacity
        Variables:
            x3: facility interdiction, i in [n], binary, 0 for interdiction and 1 for not
            y1: facility reconstruction, i in [m], binary, 1 for reconstruction and 0 for not
            y2: demand ratio of customer j in [m] supplied from facility i in [n], continuous
        Formulation:
            max_{x3} cc @ y1 + sum(c[i,j] * d[j] * y2[i,j] for i in [n] for j in [m])
            s.t. x3 in {0,1}
                 1 @ (1 - x3) <= B
                 y1,y2 in arg min_{yy1,yy2} cc @ yy1 + sum(c[i,j] * d[j] * yy2[i,j] for i in [n] for j in [m])
                                   s.t. yy1 in {0,1}, yy2 in [0,1]
                                        1 @ yy2[:,j] <= 1, for all j in [m]
                                        sum(d[j] * yy2[i,j]) <= C[i] * x3[i] + CC[i] * yy1[i], for all i in [n]
    '''
    B = round(n / 3)
    num_try = 1000000
    for i in range(num_try):
        d = np.random.rand(m)
        C = (2 * np.random.rand(n) +1) / 3 * (m / n)
        C_sort = np.sort(C)[::-1]
        CC = C * B/n
        if (np.sum(d) < np.sum(C)) & (np.sum(d) < np.sum(C) - np.sum(C_sort[0:B]) + np.sum(CC)) & (np.sum(d) > (1 - B/n) * np.sum(C)):
            break
    if i == num_try - 1:
        exit('error: instance generation')
    c = np.random.rand(n, m)
    cc = CC * np.max(c, axis=1) * 2
    c = c - np.max(np.max(c)) * 10

    x3 = cp.Variable(n, boolean=True)
    if flag_MILP:
        y1 = cp.Variable(n, boolean=True)
    else:
        y1 = np.zeros(n)
    y2 = cp.Variable((n,m))

    objctvU = np.zeros(n) @ x3 + np.zeros(n) @ y1 + np.zeros(n) @ y2 @ np.zeros(m) \
              - cc @ y1 - sum(c[i,j] * d[j] * y2[i,j] for i in range(n) for j in range(m))
    cstsU = []
    cstsU += [
        np.ones(n) @ (1 - x3) <= B,
    ]
    prblU = cp.Problem(cp.Minimize(objctvU), cstsU)
    data, _, _ = prblU.get_problem_data(cp.GUROBI)
    cdu = data['q']
    cu = cdu[0:n]
    du = cdu[n:]
    ABu = data['F'].toarray()
    Au = ABu[:,0:n]
    Bu = ABu[:,n:]
    hu = data['G']

    objctvL = np.zeros(n) @ x3 + np.zeros(n) @ y1 + np.zeros(n) @ y2 @ np.zeros(m) \
              - cc @ y1 - sum(c[i,j] * d[j] * y2[i,j] for i in range(n) for j in range(m))
    cstsL = []
    cstsL += [
        0 <= y2,
        np.ones(n) @ y2 <= 1,
    ]
    for i in range(n):
        cstsL += [
            y2[i,:] @ d <= C[i] * x3[i] + CC[i] * y1[i],
        ]
    prblL = cp.Problem(cp.Maximize(objctvL), cstsL)
    data, _, _ = prblL.get_problem_data(cp.GUROBI)
    cdl = -data['q']
    cl = cdl[0:n]
    dl = cdl[n:]
    ABl = data['F'].toarray()
    Al = ABl[:,0:n]
    Bl = ABl[:,n:]
    hl = data['G']

    idx_binary = data['bool_vars_idx']
    sign = ['Continuous' for i in range(len(cdu))]
    for i in range(len(idx_binary)):
        sign[idx_binary[i]] = 'Binary'
    X = sign[0:n]
    Y = sign[n:]

    flag_LowerLevel = 0

    if flag_MILP & (flag_LowerLevel != 0):
        idx_y1_fixed = set(range(n))
    else:
        idx_y1_fixed = set()

    return BMILP(X, Y, Au, Bu, hu, cu, du, Al, Bl, hl, dl, flag_LowerLevel, idx_y1_fixed)

def instance_MILP(n, ratio_y_bianry=0.5, sparsity=0, factor_scale=1):
    '''
    General Mixed-Integer Linear Program
        Parameters:
            num_x3: num_x3
            num_y1: num_y1
            num_y2: num_y2
        Variables:
            x3: upper-level decision, binary, i in [num_x3]
            y1: lower-level decision, binary, i in [num_y1]
            y2: lower-level decision, continuous, i in [num_y2]
        Formulation:
            min_{x3, y1, y2} cu3 @ x3 + du1 @ y1 + du2 @ y2
            s.t. x3 in {0,1}
                 Au3 @ x3 + Bu1 @ y1 + Bu2 @ y2 <= hu
                 y1,y2 in arg max_{yy1,yy2} dl1 @ yy1 + dl2 @ yy2
                                   s.t. yy1 in {0,1}, yy2 in R+
                                        Al3 @ x3 + Bl1 @ y1 + Bl2 @ y2 <= hl
    '''
    num_x3 = n
    num_y = int(factor_scale * num_x3)
    num_y1 = int(num_y * ratio_y_bianry)
    num_y2 = num_y - num_y1
    num_csts_upper = int(0.4 * num_x3)
    num_csts_lower = int(0.4 * num_y)

    cu3 = np.random.rand(num_x3) * 100 - 50
    du1 = np.random.rand(num_y1) * 100 - 50
    du2 = np.random.rand(num_y2) * 100 - 50
    Au3 = np.random.rand(num_csts_upper, num_x3) * 10
    Bu1 = np.random.rand(num_csts_upper, num_y1) * 10
    Bu2 = np.random.rand(num_csts_upper, num_y2) * 10
    hu = np.random.rand(num_csts_upper) * 100 + 30
    dl1 = np.random.rand(num_y1) * 100 - 50
    dl2 = np.random.rand(num_y2) * 100 - 50
    Al3 = np.random.rand(num_csts_lower, num_x3) * 10
    Bl1 = np.random.rand(num_csts_lower, num_y1) * 10
    Bl2 = np.random.rand(num_csts_lower, num_y2) * 10
    hl = np.random.rand(num_csts_lower) * 100 + 10
    for i in range(num_csts_lower):
        for j in range(num_y1):
            if np.random.rand() <= sparsity:
                Bl1[i,j] = 0
    for i in range(num_csts_lower):
        for j in range(num_x3):
            if np.random.rand() <= sparsity:
                Al3[i,j] = 0
    for i in range(num_csts_lower):
        for j in range(num_y2):
            if np.random.rand() <= sparsity:
                Bl2[i,j] = 0
    # for i in range(num_csts_upper):
    #     for j in range(num_y1):
    #         if np.random.rand() <= sparsity:
    #             Bu1[i,j] = 0
    # for i in range(num_csts_upper):
    #     for j in range(num_x3):
    #         if np.random.rand() <= sparsity:
    #             Au3[i,j] = 0
    # for i in range(num_csts_upper):
    #     for j in range(num_y2):
    #         if np.random.rand() <= sparsity:
    #             Bu2[i,j] = 0

    x3 = cp.Variable(num_x3, boolean=True)
    y1 = cp.Variable(num_y1, boolean=True)
    y2 = cp.Variable(num_y2)

    objctvU = np.zeros(num_x3) @ x3 + np.zeros(num_y1) @ y1 + np.zeros(num_y2) @ y2 \
              + cu3 @ x3 + du1 @ y1 + du2 @ y2
    cstsU = []
    cstsU += [
        Au3 @ x3 + Bu1 @ y1 + Bu2 @ y2 <= hu,
    ]
    prblU = cp.Problem(cp.Minimize(objctvU), cstsU)
    data, _, _ = prblU.get_problem_data(cp.GUROBI)
    cdu = data['q']
    cu = cdu[0:num_x3]
    du = cdu[num_x3:]
    ABu = data['F'].toarray()
    Au = ABu[:,0:num_x3]
    Bu = ABu[:,num_x3:]
    hu = data['G']

    objctvL = np.zeros(num_x3) @ x3 + np.zeros(num_y1) @ y1 + np.zeros(num_y2) @ y2 \
              + dl1 @ y1 + dl2 @ y2
    cstsL = []
    cstsL += [
        Al3 @ x3 + Bl1 @ y1 + Bl2 @ y2 <= hl,
        0 <= y2  # , y2 <= 1,
    ]
    prblL = cp.Problem(cp.Maximize(objctvL), cstsL)
    data, _, _ = prblL.get_problem_data(cp.GUROBI)
    cdl = -data['q']
    cl = cdl[0:num_x3]
    dl = cdl[num_x3:]
    ABl = data['F'].toarray()
    Al = ABl[:,0:num_x3]
    Bl = ABl[:,num_x3:]
    hl = data['G']

    idx_binary = data['bool_vars_idx']
    sign = ['Continuous' for i in range(len(cdu))]
    for i in range(len(idx_binary)):
        sign[idx_binary[i]] = 'Binary'
    X = sign[0:num_x3]
    Y = sign[num_x3:]

    flag_LowerLevel = 0
    idx_y1_fixed = set()

    return BMILP(X, Y, Au, Bu, hu, cu, du, Al, Bl, hl, dl, flag_LowerLevel, idx_y1_fixed)

def instance_MILP_relaxed(n, ratio_y_bianry=0.5, sparsity=0, factor_scale=1):
    '''
    General Mixed-Integer Linear Program
        Parameters:
            num_x3: num_x3
            num_y1: num_y1
            num_y2: num_y2
        Variables:
            x3: upper-level decision, binary, i in [num_x3]
            y1: lower-level decision, binary, i in [num_y1]
            y2: lower-level decision, continuous, i in [num_y2]
            su: upper-level slack, continuous, scalar
            sl: lower-level slack, continuous, scalar
        Formulation:
            min_{x3, y1, y2} cu3 @ x3 + du1 @ y1 + du2 @ y2 - Mu * su
            s.t. x3 in {0,1}
                 Au3 @ x3 + Bu1 @ y1 + Bu2 @ y2 <= hu + su
                 y1,y2 in arg max_{yy1,yy2} dl1 @ yy1 + dl2 @ yy2 + Ml * sl
                                   s.t. yy1 in {0,1}, yy2 in R, s in R
                                        Al3 @ x3 + Bl1 @ y1 + Bl2 @ y2 <= hl + sl
    '''
    num_x3 = n
    num_y = factor_scale * num_x3
    num_y1 = int(num_y * ratio_y_bianry)
    num_y2 = num_y - num_y1
    num_csts_upper = int(0.4 * num_x3)
    num_csts_lower = int(0.4 * num_y)

    cu3 = np.random.rand(num_x3) * 100 - 50
    du1 = np.random.rand(num_y1) * 100 - 50
    du2 = np.random.rand(num_y2) * 100 - 50
    Au3 = np.random.rand(num_csts_upper, num_x3) * 10
    Bu1 = np.random.rand(num_csts_upper, num_y1) * 10
    Bu2 = np.random.rand(num_csts_upper, num_y2) * 10
    hu = np.random.rand(num_csts_upper) * 100 + 30
    Mu = 1e3 / (10 * n * 2 - np.min(hu))
    sumax = 1 # 10 * n * 2 - np.min(hu)
    dl1 = np.random.rand(num_y1) * 100 - 50
    dl2 = np.random.rand(num_y2) * 100 - 50
    Al3 = np.random.rand(num_csts_lower, num_x3) * 10
    Bl1 = np.random.rand(num_csts_lower, num_y1) * 10
    Bl2 = np.random.rand(num_csts_lower, num_y2) * 10
    hl = np.random.rand(num_csts_lower) * 100 + 10
    Ml = np.maximum(np.max(dl1), np.max(dl2)) * 10 / (10 * n * 2 - np.min(hl))
    slmax = 1  # 10 * n * 2 - np.min(hl)
    for i in range(num_csts_lower):
        for j in range(num_y1):
            if np.random.rand() <= sparsity:
                Bl1[i,j] = 0
    for i in range(num_csts_lower):
        for j in range(num_x3):
            if np.random.rand() <= sparsity:
                Al3[i,j] = 0
    for i in range(num_csts_lower):
        for j in range(num_y2):
            if np.random.rand() <= sparsity:
                Bl2[i,j] = 0

    x3 = cp.Variable(num_x3, boolean=True)
    y1 = cp.Variable(num_y1, boolean=True)
    y2 = cp.Variable(num_y2)
    su = cp.Variable()
    sl = cp.Variable()

    objctvU = 0 * su + np.zeros(num_x3) @ x3 + np.zeros(num_y1) @ y1 + np.zeros(num_y2) @ y2 + 0 * sl \
              + cu3 @ x3 + du1 @ y1 + du2 @ y2 + Mu * su
    cstsU = []
    cstsU += [
        Au3 @ x3 + Bu1 @ y1 + Bu2 @ y2 <= hu + su * sumax * np.ones(num_csts_upper),
        0 <= su  # , su <= 1
    ]
    prblU = cp.Problem(cp.Minimize(objctvU), cstsU)
    data, _, _ = prblU.get_problem_data(cp.GUROBI)
    cdu = data['q']
    cu = cdu[0:1+num_x3]
    du = cdu[1+num_x3:]
    ABu = data['F'].toarray()
    Au = ABu[:,0:1+num_x3]
    Bu = ABu[:,1+num_x3:]
    hu = data['G']

    objctvL = 0 * su + np.zeros(num_x3) @ x3 + np.zeros(num_y1) @ y1 + np.zeros(num_y2) @ y2 + 0 * sl \
              + dl1 @ y1 + dl2 @ y2 - Ml * sl
    cstsL = []
    cstsL += [
        Al3 @ x3 + Bl1 @ y1 + Bl2 @ y2 <= hl + sl * slmax * np.ones(num_csts_lower),
        0 <= y2, y2 <= 1,
        0 <= sl  # , sl <= 1,
    ]
    prblL = cp.Problem(cp.Maximize(objctvL), cstsL)
    data, _, _ = prblL.get_problem_data(cp.GUROBI)
    cdl = -data['q']
    cl = cdl[0:1+num_x3]
    dl = cdl[1+num_x3:]
    ABl = data['F'].toarray()
    Al = ABl[:,0:1+num_x3]
    Bl = ABl[:,1+num_x3:]
    hl = data['G']

    idx_binary = data['bool_vars_idx']
    sign = ['Continuous' for i in range(len(cdu))]
    for i in range(len(idx_binary)):
        sign[idx_binary[i]] = 'Binary'
    X = sign[0:1+num_x3]
    Y = sign[1+num_x3:]

    flag_LowerLevel = 0
    idx_y1_fixed = set()

    return BMILP(X, Y, Au, Bu, hu, cu, du, Al, Bl, hl, dl, flag_LowerLevel, idx_y1_fixed)

def instance_xuwang(n, idx_run):
    filename = "mixed-integer/xuwang/bmilplib_" + str(n) + "_" + str(idx_run)

    flag_read_obj_lower = False
    flag_read_cons_lower = False
    with open(filename + ".aux", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "@VARSBEGIN":
                flag_read_obj_lower = True
                list_vars_lower = []
                list_obj_lower = []
                continue
            if line.strip() == "@VARSEND":
                flag_read_obj_lower = False
            if line.strip() == "@CONSTRSBEGIN":
                flag_read_cons_lower = True
                list_cons_lower = []
                continue
            if line.strip() == "@CONSTRSEND":
                flag_read_cons_lower = False
            if flag_read_obj_lower:
                name, coeff = line.strip().split()
                list_vars_lower.append(name)
                list_obj_lower.append(float(coeff))
            if flag_read_cons_lower:
                list_cons_lower.append(line.strip())
    dl = -np.array(list_obj_lower)

    model = gp.read(filename + ".mps.gz")

    vars_ = model.getVars()
    list_vars_upper = []
    X = []
    Y = []
    UB_upper = []
    LB_upper = []
    UB_lower = []
    LB_lower = []
    list_obj_upper = []
    list_obj_lower = []
    for v in vars_:
        if v.VType == 'C':
            sign = 'Continuous'
        else:
            if v.VType == 'B':
                sign = 'Binary'
            else:
                if v.VType == 'I':
                    sign = 'Binary'
        if v.VarName not in list_vars_lower:
            list_vars_upper.append(v.VarName)
            X.append(sign)
            # UB_upper.append(v.UB)
            # LB_upper.append(v.LB)
            UB_upper.append(np.infty)
            LB_upper.append(-np.infty)
            list_obj_upper.append(v.Obj)
        else:
            Y.append(sign)
            # UB_lower.append(v.UB)
            # LB_lower.append(v.LB)
            if sign == 'Binary':
                UB_lower.append(np.infty)
                LB_lower.append(-np.infty)
            else:
                UB_lower.append(v.UB)
                LB_lower.append(v.LB)
            list_obj_lower.append(v.Obj)
    cu = np.array(list_obj_upper)
    du = np.array(list_obj_lower)

    constrs = model.getConstrs()
    list_cons_upper = []
    Au = []
    Bu = []
    hu = []
    Al = []
    Bl = []
    hl = []
    for c in constrs:
        if c.Sense != '<':
            exit('error')
        rhs = c.RHS
        row = model.getRow(c)
        temp_upper = np.zeros(len(list_vars_upper))
        temp_lower = np.zeros(len(list_vars_lower))
        for i in range(row.size()):
            name = row.getVar(i).VarName
            coeff = row.getCoeff(i)
            if name in list_vars_upper:
                idx = list_vars_upper.index(name)
                temp_upper[idx] = coeff
            if name in list_vars_lower:
                idx = list_vars_lower.index(name)
                temp_lower[idx] = coeff
        temp_upper = list(temp_upper)
        temp_lower = list(temp_lower)
        if c.ConstrName not in list_cons_lower:
            list_cons_upper.append(c.ConstrName)
            Au.append(temp_upper)
            Bu.append(temp_lower)
            hu.append(rhs)
        else:
            Al.append(temp_upper)
            Bl.append(temp_lower)
            hl.append(rhs)
    for i in range(len(list_vars_upper)):
        if LB_upper[i] != -np.infty:
            temp = np.zeros(len(list_vars_upper))
            temp[i] = -1
            Au.append(list(temp))
            Bu.append(list(np.zeros(len(list_vars_lower))))
            hu.append(-LB_upper[i])
    for i in range(len(list_vars_upper)):
        if UB_upper[i] != np.infty:
            temp = np.zeros(len(list_vars_upper))
            temp[i] = 1
            Au.append(list(temp))
            Bu.append(list(np.zeros(len(list_vars_lower))))
            hu.append(UB_upper[i])
    for i in range(len(list_vars_lower)):
        if LB_lower[i] != -np.infty:
            temp = np.zeros(len(list_vars_upper))
            temp[i] = -1
            Al.append(list(np.zeros(len(list_vars_upper))))
            Bl.append(list(temp))
            hl.append(-LB_lower[i])
    for i in range(len(list_vars_lower)):
        if UB_lower[i] != np.infty:
            temp = np.zeros(len(list_vars_upper))
            temp[i] = 1
            Al.append(list(np.zeros(len(list_vars_upper))))
            Bl.append(list(temp))
            hl.append(UB_lower[i])
    Au = np.array(Au)
    Bu = np.array(Bu)
    hu = np.array(hu)
    Al = np.array(Al)
    Bl = np.array(Bl)
    hl = np.array(hl)

    flag_LowerLevel = 0
    idx_y1_fixed = set()

    return BMILP(X, Y, Au, Bu, hu, cu, du, Al, Bl, hl, dl, flag_LowerLevel, idx_y1_fixed)

def instance_xularge(n, idx_run):
    filename = "mixed-integer/xularge/xuLarge" + str(n) + "-" + str(idx_run)

    flag_read_obj_lower = False
    flag_read_cons_lower = False
    with open(filename + ".aux", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "@VARSBEGIN":
                flag_read_obj_lower = True
                list_vars_lower = []
                list_obj_lower = []
                continue
            if line.strip() == "@VARSEND":
                flag_read_obj_lower = False
            if line.strip() == "@CONSTRSBEGIN":
                flag_read_cons_lower = True
                list_cons_lower = []
                continue
            if line.strip() == "@CONSTRSEND":
                flag_read_cons_lower = False
            if flag_read_obj_lower:
                name, coeff = line.strip().split()
                list_vars_lower.append(name)
                list_obj_lower.append(float(coeff))
            if flag_read_cons_lower:
                list_cons_lower.append(line.strip())
    dl = -np.array(list_obj_lower)

    model = gp.read(filename + ".mps.gz")

    vars_ = model.getVars()
    list_vars_upper = []
    X = []
    Y = []
    UB_upper = []
    LB_upper = []
    UB_lower = []
    LB_lower = []
    list_obj_upper = []
    list_obj_lower = []
    for v in vars_:
        if v.VType == 'C':
            sign = 'Continuous'
        else:
            if v.VType == 'B':
                sign = 'Binary'
            else:
                if v.VType == 'I':
                    sign = 'Binary'
        if v.VarName not in list_vars_lower:
            list_vars_upper.append(v.VarName)
            X.append(sign)
            UB_upper.append(v.UB)
            LB_upper.append(v.LB)
            list_obj_upper.append(v.Obj)
        else:
            Y.append(sign)
            UB_lower.append(v.UB)
            LB_lower.append(v.LB)
            list_obj_lower.append(v.Obj)
    cu = np.array(list_obj_upper)
    du = np.array(list_obj_lower)

    constrs = model.getConstrs()
    list_cons_upper = []
    Au = []
    Bu = []
    hu = []
    Al = []
    Bl = []
    hl = []
    for c in constrs:
        if c.Sense != '<':
            exit('error')
        rhs = c.RHS
        row = model.getRow(c)
        temp_upper = np.zeros(len(list_vars_upper))
        temp_lower = np.zeros(len(list_vars_lower))
        for i in range(row.size()):
            name = row.getVar(i).VarName
            coeff = row.getCoeff(i)
            if name in list_vars_upper:
                idx = list_vars_upper.index(name)
                temp_upper[idx] = coeff
            if name in list_vars_lower:
                idx = list_vars_lower.index(name)
                temp_lower[idx] = coeff
        temp_upper = list(temp_upper)
        temp_lower = list(temp_lower)
        if c.ConstrName not in list_cons_lower:
            list_cons_upper.append(c.ConstrName)
            Au.append(temp_upper)
            Bu.append(temp_lower)
            hu.append(rhs)
        else:
            Al.append(temp_upper)
            Bl.append(temp_lower)
            hl.append(rhs)
    for i in range(len(list_vars_upper)):
        if LB_upper[i] != -np.infty:
            temp = np.zeros(len(list_vars_upper))
            temp[i] = -1
            Au.append(list(temp))
            Bu.append(list(np.zeros(len(list_vars_lower))))
            hu.append(-LB_upper[i])
    for i in range(len(list_vars_upper)):
        if UB_upper[i] != np.infty:
            temp = np.zeros(len(list_vars_upper))
            temp[i] = 1
            Au.append(list(temp))
            Bu.append(list(np.zeros(len(list_vars_lower))))
            hu.append(UB_upper[i])
    for i in range(len(list_vars_lower)):
        if LB_lower[i] != -np.infty:
            temp = np.zeros(len(list_vars_upper))
            temp[i] = -1
            Al.append(list(np.zeros(len(list_vars_upper))))
            Bl.append(list(temp))
            hl.append(-LB_lower[i])
    for i in range(len(list_vars_lower)):
        if UB_lower[i] != np.infty:
            temp = np.zeros(len(list_vars_upper))
            temp[i] = 1
            Al.append(list(np.zeros(len(list_vars_upper))))
            Bl.append(list(temp))
            hl.append(UB_lower[i])
    Au = np.array(Au)
    Bu = np.array(Bu)
    hu = np.array(hu)
    Al = np.array(Al)
    Bl = np.array(Bl)
    hl = np.array(hl)

    flag_LowerLevel = 0
    idx_y1_fixed = set()

    return BMILP(X, Y, Au, Bu, hu, cu, du, Al, Bl, hl, dl, flag_LowerLevel, idx_y1_fixed)