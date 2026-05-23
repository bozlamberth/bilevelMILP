import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB
import pyomo.environ as pe
import pao.pyomo as pao
import time
from scipy.linalg import orth
from NNmodel import *
import random

''' 
Algorithms of Bilevel Mixed-Integer Linear Programs (BMILP)
    Algorithm requirements:
        1. Binary tender
        2. x and y are mixed-binary
        3. There exist continuous variables in the entries of y  
        4. When some entries of y1 are fixed, phi is supermodular or submodular in x3
    
    General Formulation of BMILP:
        min cu1 @ x1 + cu2 @ x2 + cu3 @ x3 + du1 @ y1 + du2 @ y2
        s.t.   x1 in X1, x2 in X2, x3 in X3, y1 in Y1, y2 in Y2
               Au1 @ x1 + Au2 @ x2 + Au3 @ x3 + Bu1 @ y1 + Bu2 @ y2 <= hu
               y1,y2 in arg max dl1 @ yy1 + dl2 @ yy2
                             s.t.   yy1 in Y1, yy2 in Y2
                                    Al1 @ x1 + Al2 @ x2 + Al3 @ x3 + Bl1 @ yy1 + Bl2 @ yy2 <= hl  # Al1 = 0, Al2 = 0
    Value Function:
        phi(x3) = max dl1 @ yy1 + dl2 @ yy2
                     s.t.   yy1 in Y1, yy2 in Y2
                            Al3 @ x3 + Bl1 @ yy1 + Bl2 @ yy2 <= hl
                            
    Value Function with some entries of y1 fixed:
        varphi(x3, yy1_fixed) = dl1 @ yy1 - dl1_fixed @ yy1_fixed + dl2 @ yy2
                     s.t.   yy2 in Y2
                            Al3 @ x3 + Bl1 @ yy1 + Bl2 @ yy2 <= hl
                            yy1[idx_fixed] = yy1_fixed
        phi(x3) = dl1_fixed @ yy1_fixed_optimal + varphi(x3, yy1_fixed_optimal)
        
Special Property
    when y1 with index idx_y1_fixed is fixed, there exists a special property in x3,
    which is indicated by flag_LowerLevel, with 0 for general, 1 for supermodular, and 2 for submodular
'''

class BMILP(object):
    def __init__(self, X, Y, Au, Bu, hu, cu, du, Al, Bl, hl, dl, flag_LowerLevel=0, idx_y1_fixed=set()):
        self.hu = hu
        self.hl = hl

        # identify linking/binary x
        Almax = np.max(Al, axis=0)
        Almin = np.min(Al, axis=0)
        idx_non_linking = set(np.where(Almax == 0)[0]) & set(np.where(Almin == 0)[0])
        idx_linking = set(range(len(X))) - idx_non_linking
        X12 = [X[i] for i in idx_non_linking]
        Au12 = Au[:, list(idx_non_linking)]
        Al12 = Al[:, list(idx_non_linking)]
        cu12 = cu[list(idx_non_linking)]
        idx_binary = set([i for i,x in enumerate(X12) if x == 'Binary'])
        idx_continuous = set([i for i,x in enumerate(X12) if x == 'Continuous'])
        self.X1 = [X12[i] for i in idx_binary]
        self.Au1 = Au12[:, list(idx_binary)]
        self.Al1 = Al12[:, list(idx_binary)]
        self.cu1 = cu12[list(idx_binary)]
        self.X2 = [X12[i] for i in idx_continuous]
        self.Au2 = Au12[:, list(idx_continuous)]
        self.Al2 = Al12[:, list(idx_continuous)]
        self.cu2 = cu12[list(idx_continuous)]
        self.X3 = [X[i] for i in idx_linking]
        self.Au3 = Au[:, list(idx_linking)]
        self.Al3 = Al[:, list(idx_linking)]
        self.cu3 = cu[list(idx_linking)]
        self.num_x1 = len(self.X1)
        self.num_x2 = len(self.X2)
        self.num_x3 = len(self.X3)
        if 'Continuous' in self.X3:
            exit('Error: Binary tender assumption violated')

        # identify binary y
        idx_binary = set([i for i,x in enumerate(Y) if x == 'Binary'])
        idx_continuous = set([i for i,x in enumerate(Y) if x == 'Continuous'])
        self.Y1 = [Y[i] for i in idx_binary]
        self.Bu1 = Bu[:, list(idx_binary)]
        self.Bl1 = Bl[:, list(idx_binary)]
        self.du1 = du[list(idx_binary)]
        self.dl1 = dl[list(idx_binary)]
        self.Y2 = [Y[i] for i in idx_continuous]
        self.Bu2 = Bu[:, list(idx_continuous)]
        self.Bl2 = Bl[:, list(idx_continuous)]
        self.du2 = du[list(idx_continuous)]
        self.dl2 = dl[list(idx_continuous)]
        self.num_y1 = len(self.Y1)
        self.num_y2 = len(self.Y2)
        if self.num_y2 == 0:
            exit('Error: All binary variables in y')

        # MIBS related
        self.time_MIBS = 0
        self.time_MIBS_total = 0
        self.objctv_upper_MIBS_original = None
        self.objctv_lower_MIBS_original = None

        # identify problem
        self.flag_LowerLevel = flag_LowerLevel
        if self.flag_LowerLevel == 0:
            self.idx_y1_fixed = set()
        else:
            self.idx_y1_fixed = idx_y1_fixed
        self.LL_x3, self.LL_prbl = self.LowerLevel()
        self.LL_fixed_x3, self.LL_fixed_y1_fixed, self.LL_fixed_prbl = self.LowerLevel_fixed()
        self.UL_x3, self.UL_phi, self.UL_prbl = self.UpperLevel()
        self.buffer_x1 = list([])
        self.buffer_x2 = list([])
        self.buffer_x3 = list([])
        self.buffer_y1 = list([])
        self.buffer_y2 = list([])
        self.buffer_phi = list([])
        self.buffer_obj_upper = list([])
        self.buffer_y1_fixed = list([])
        self.buffer_x3y1_fixed = list([])
        self.buffer_varphi = list([])
        self.time_Phi = 0
        self.time_varPhi = 0
        self.time_feasible = 0

        # no-good related
        self.flag_no_good = False
        self.flag_incumbent = True & self.flag_no_good
        self.no_good_value_x3 = []
        self.no_good_value_x3_history = []
        self.good_value_x1 = None
        self.good_value_x2 = None
        self.good_value_x3 = None
        self.good_value_y1 = None
        self.good_value_y2 = None
        self.good_UB = np.infty
        self.flag_good_UB = False
        self.flag_incumbent_move = False

        # heuristic UB
        self.flag_heuristic_UB = False
        self.flag_heuristic_UB_warmstart = False & self.flag_heuristic_UB
        self.flag_heuristic_UB_incumbent = True & self.flag_heuristic_UB
        self.flag_heuristic_UB_solving = True
        self.heuristic_UB_frequency = 500
        if self.flag_heuristic_UB:
            # heuristic UB - sampling
            self.heuristic_UB_num_samples_target = 2 * self.num_x3
            self.heuristic_UB_num_sampling = 10 * self.heuristic_UB_num_samples_target
            self.heuristic_UB_num_sampling_only = 2 * self.num_x3
            self.heuristic_UB_max_repeated = 100
            # heuristic UB - training
            self.heuristic_UB_NNArchitecture = None
            self.heuristic_UB_num_samples_guess = 1000
            self.heuristic_UB_num_epoch_initial = max(200, 2 * self.num_x3)
            self.heuristic_UB_num_epoch_continued = 50
            # heuristic UB - record
            self.heuristic_UB_num_all = 0
            self.heuristic_UB_num_win = -1

        # heuristic LB
        self.flag_heuristic_LB = False & (self.num_y1 != 0)
        self.Lagrangian_timelimit_lam = 10 * (self.num_x3 / 10)
        self.Lagrangian_timelimit_lam_heuristic = 30
        self.Lagrangian_timelimit_rho = 30 * (self.num_x3 / 10)
        if self.flag_heuristic_LB:
            # heuristic LB - sampling
            self.heuristic_LB_num_samples_target = 1000  # 10 * (self.num_x3 + self.num_y1)
            self.heuristic_LB_num_sampling = 10 * self.heuristic_LB_num_samples_target
            self.heuristic_LB_max_repeated = 100
            # training
            self.heuristic_LB_num_epoch = 1000
            # heuristic LB - record
            self.heuristic_LB = -np.infty
            self.heuristic_LB_num_all = 0
            self.heuristic_LB_num_win = -1
            # if self.flag_LowerLevel == 0:
            # self.heuristic_LB_rho = self.getRho_heuristic_LB()
            self.LL_heuLB_x3, self.LL_heuLB_yy1, self.LL_heuLB_prbl = self.LowerLevel_fixed_heuristic_LB()
            self.heuristic_LB_lamU, self.heuristic_LB_lamL = self.getLam_heuristic_quasiLL()
            # self.heuristic_LB_lamU, self.heuristic_LB_lamL = self.getLam_heuristic_LB()

        # cutting plane related
        self.flag_buffer_upper = False | self.flag_no_good | self.flag_heuristic_UB | self.flag_heuristic_LB
        self.history_value_x3_lazy = []
        self.heuristic_count = 0
        self.time_CP_solving = 0
        self.time_CP_heuristic_sampling = 0
        self.time_CP_heuristic_training = 0
        self.time_CP_heuristic_solving = 0
        self.time_CP_heuristic_checking = 0
        self.time_CP_heuristic_total = 0
        self.time_CP_total = 0

        # Lagrangian related
        self.flag_LI = False
        self.flag_ALI = True
        self.flag_quasi_relaxed = False
        self.buffer_y1_fixed_lam = list([])
        self.buffer_lamU = list([])
        self.buffer_lamL = list([])
        self.count_lazy_LI = 0
        self.count_lazy_ALI = 0
        if self.flag_LI | self.flag_ALI:
            T1 = time.time()
            if len(self.idx_y1_fixed) == 0:
                if (self.flag_LowerLevel == 0) & (self.flag_LI == False):
                    self.rho = self.getRho_generalLL()
                else:
                    if self.flag_LowerLevel == 0:
                        self.lamU, self.lamL = self.getLam_generalLL()
                    else:
                        self.lamU, self.lamL = self.getLam_specialLL()
                    self.rho = np.maximum(np.max(self.lamU), -np.min(self.lamL))
            else:
                if self.flag_quasi_relaxed:
                    self.lamU, self.lamL = self.getLam_quasiLL_relaxed()
                    self.rho = np.maximum(np.max(self.lamU), -np.min(self.lamL))
            T2 = time.time()
            self.time_Lagrangian = T2 - T1
            # FileName = 'Lagrangian.xlsx'
            # writer = pd.ExcelWriter(FileName, mode='a', if_sheet_exists='new')
            # df_samples = pd.DataFrame([self.lamU, self.lamL, [self.rho]])
            # df_samples.to_excel(writer, str(self.num_x3), header=None, index=False)
            # writer.close()
            # FileName = 'Lagrangian.xlsx'
            # data = pd.read_excel(io=FileName, sheet_name=None, header=None)
            # self.lamU = data[str(self.num_x3)].values[0]
            # self.lamL = data[str(self.num_x3)].values[1]
            # self.rho = data[str(self.num_x3)].values[2][0]

        # property related
        self.flag_SI = False & (self.flag_LowerLevel != 0)
        self.count_lazy_SI = 0
        self.time_EPI_pi = 0
        self.time_EPI_pi_heuristic = 0

        # objective cut
        self.flag_OI = False
        if self.flag_OI:
            self.LP_OI_x3, self.LP_OI_prbl = self.LP_objectiveCut()
            # self.LP_OI_x3, self.LP_OI_prbl = self.LP_objectiveCut2()
            self.MILP_OI_lam, self.MILP_OI_prbl = self.MILP_objectiveCut()
            self.count_OI = 0

        # Jiang's cut
        self.flag_JI = False
        if self.flag_JI:
            # self.MILP_JI_x3, self.MILP_JI_prbl = self.MILP_JiangsCut()
            # self.MILP_JI_x3, self.MILP_JI_prbl = self.MILP_JiangsCut2()
            self.MILP_JI_x3, self.MILP_JI_y1, self.MILP_JI_prbl = self.MILP_JiangsCut2_fixed()
            # self.MILP_JI_x3, self.MILP_JI_prbl = self.MILP_JiangsCut3()
            # self.MILP_JI_x3, self.MILP_JI_y1, self.MILP_JI_prbl = self.MILP_JiangsCut3_fixed()
            self.LL_JI_x3, self.LL_JI_prbl = self.LowerLevel_JI()
            self.count_JI = 0
            self.time_JI = 0

        # NN related
        self.flag_ISNN = False
        self.flag_doubleX = False & self.flag_ISNN
        self.flag_NNSolveByCut = False
        self.num_iter_enhanced = 1
        self.num_sampling = 10000
        self.max_repeated = 100
        self.num_epoch = 1000
        self.samples = None
        self.NNArchitecture = None
        self.time_NN_sampling = 0
        self.time_NN_training = 0
        self.time_NN_solving = 0
        self.time_NN_checking = 0
        self.time_NN_total = 0

        return

    def LowerLevel(self):
        x3 = cp.Parameter(self.num_x3)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        csts = [
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl
        ]
        prbl = cp.Problem(cp.Maximize(self.dl1 @ y1 + self.dl2 @ y2), csts)
        return x3, prbl

    def getPhi(self, value_x3):
        list_value_x3 = list(np.int_(np.round(value_x3)))
        if list_value_x3 in self.buffer_x3:
            idx = self.buffer_x3.index(list_value_x3)
            return self.buffer_phi[idx], np.array(self.buffer_y1_fixed[idx])
        else:
            self.LL_x3.value = value_x3
            t1 = time.time()
            self.LL_prbl.solve(solver=cp.GUROBI, verbose=False)
            t2 = time.time()
            self.time_Phi += t2 - t1
            if self.LL_prbl.status != 'optimal':
                print('Error: Failed in getting phi')
                return None, None
            value_phi = self.LL_prbl.solution.opt_val
            if self.num_y1 == 0:
                value_y1 = np.zeros(self.num_y1)
            else:
                value_y1 = list(self.LL_prbl.solution.primal_vars.values())[0]
            value_y1_fixed = value_y1[list(self.idx_y1_fixed)]
            if self.flag_buffer_upper:
                value_x1, value_x2, value_y1, value_y2, value_obj_upper = self.getFeasible(value_x3, value_phi)
                self.buffer_x1.append(list(np.int_(np.round(value_x1))))
                self.buffer_x2.append(list(value_x2))
                self.buffer_y1.append(list(np.int_(np.round(value_y1))))
                self.buffer_y2.append(list(value_y2))
                self.buffer_obj_upper.append(value_obj_upper)
            self.buffer_x3.append(list(np.int_(np.round(value_x3))))
            self.buffer_phi.append(value_phi)
            self.buffer_y1_fixed.append(list(np.int_(np.round(value_y1_fixed))))
            return value_phi, np.int_(np.round(value_y1_fixed))

    def UpperLevel(self):
        x3 = cp.Parameter(self.num_x3)
        phi = cp.Parameter()

        if self.num_x1 == 0:
            x1 = np.zeros(self.num_x1)
        else:
            x1 = cp.Variable(self.num_x1, boolean=True)
        if self.num_x2 == 0:
            x2 = np.zeros(self.num_x2)
        else:
            x2 = cp.Variable(self.num_x2)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        objctv = self.cu1 @ x1 + self.cu2 @ x2 + self.cu3 @ x3 + self.du1 @ y1 + self.du2 @ y2
        csts = [
            self.Au1 @ x1 + self.Au2 @ x2 + self.Au3 @ x3 + self.Bu1 @ y1 + self.Bu2 @ y2 <= self.hu,
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            self.dl1 @ y1 + self.dl2 @ y2 >= phi
        ]
        prbl = cp.Problem(cp.Minimize(objctv), csts)
        return x3, phi, prbl

    def getFeasible(self, value_x3, value_phi):
        list_value_x3 = list(np.int_(np.round(value_x3)))
        if (list_value_x3 in self.buffer_x3) & self.flag_buffer_upper:
            idx = self.buffer_x3.index(list_value_x3)
            return np.array(self.buffer_x1[idx]), np.array(self.buffer_x2[idx]), np.array(self.buffer_y1[idx]), np.array(self.buffer_y2[idx]), self.buffer_obj_upper[idx]
        else:
            self.UL_x3.value = value_x3
            self.UL_phi.value = value_phi - 1e-8
            t1 = time.time()
            self.UL_prbl.solve(solver=cp.GUROBI, verbose=False)
            t2 = time.time()
            self.time_feasible += t2 - t1
            if self.UL_prbl.status != 'optimal':
                exit('Error: Correction infeasible')
            else:
                idx = 0
                if self.num_x1 == 0:
                    value_x1 = np.zeros(self.num_x1)
                else:
                    value_x1 = list(self.UL_prbl.solution.primal_vars.values())[idx]
                    idx += 1
                if self.num_x2 == 0:
                    value_x2 = np.zeros(self.num_x2)
                else:
                    value_x2 = list(self.UL_prbl.solution.primal_vars.values())[idx]
                    idx += 1
                if self.num_y1 == 0:
                    value_y1 = np.zeros(self.num_y1)
                else:
                    value_y1 = list(self.UL_prbl.solution.primal_vars.values())[idx]
                    idx += 1
                value_y2 = list(self.UL_prbl.solution.primal_vars.values())[idx]
                value_g = self.dl1 @ value_y1 + self.dl2 @ value_y2
                if value_g < value_phi - 1e-6:
                    exit('Error: Bilevel infeasible')
                value_obj_upper = self.UL_prbl.solution.opt_val
            return np.int_(np.round(value_x1)), value_x2, np.int_(np.round(value_y1)), value_y2, value_obj_upper

    def solvebyHPR(self):
        # model
        if self.num_x1 == 0:
            x1 = np.zeros(self.num_x1)
        else:
            x1 = cp.Variable(self.num_x1, boolean=True)
        if self.num_x2 == 0:
            x2 = np.zeros(self.num_x2)
        else:
            x2 = cp.Variable(self.num_x2)
        x3 = cp.Variable(self.num_x3, boolean=True)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        objctv = self.cu1 @ x1 + self.cu2 @ x2 + self.cu3 @ x3 + self.du1 @ y1 + self.du2 @ y2
        csts = [
            self.Au1 @ x1 + self.Au2 @ x2 + self.Au3 @ x3 + self.Bu1 @ y1 + self.Bu2 @ y2 <= self.hu,
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl
        ]

        # solve
        prbl = cp.Problem(cp.Minimize(objctv), csts)
        prbl.solve(solver=cp.GUROBI, verbose=False)  # , threads=4

        # save results
        if prbl.status != 'optimal':
            exit('Error: HPR infeasible')
        else:
            if self.num_x1 == 0:
                value_x1 = x1
            else:
                value_x1 = x1.value
            if self.num_x2 == 0:
                value_x2 = x2
            else:
                value_x2 = x2.value
            value_x3 = x3.value
            if self.num_y1 == 0:
                value_y1 = y1
            else:
                value_y1 = y1.value
            value_y2 = y2.value
            # correct solution
            value_g = self.dl1 @ value_y1 + self.dl2 @ value_y2
            value_phi, _ = self.getPhi(value_x3)
            if abs(value_g - value_phi) >= 1e-6:
                value_x1, value_x2, value_y1, value_y2, _ = self.getFeasible(value_x3, value_phi)
            value_obj_upper = self.cu1 @ value_x1 + self.cu2 @ value_x2 + self.cu3 @ value_x3 + self.du1 @ value_y1 + self.du2 @ value_y2
            value_obj_lower = self.dl1 @ value_y1 + self.dl2 @ value_y2
        return np.int_(np.round(value_x1)), value_x2, np.int_(np.round(value_x3)), np.int_(np.round(value_y1)), value_y2, value_obj_upper, value_obj_lower

    def solvebyMIBS(self):
        bp = pe.ConcreteModel()
        if self.num_x1 == 0:
            bp.x1 = np.zeros(self.num_x1)
        else:
            bp.x1 = pe.Var(range(self.num_x1), within=pe.Binary)
        if self.num_x2 == 0:
            bp.x2 = np.zeros(self.num_x2)
        else:
            bp.x2 = pe.Var(range(self.num_x2), bounds=(-np.infty, np.infty))
        bp.x3 = pe.Var(range(self.num_x3), within=pe.Binary)
        if self.num_y1 == 0:
            bp.y1 = np.zeros(self.num_y1)
        else:
            bp.y1 = pe.Var(range(self.num_y1), within=pe.Binary)
        bp.y2 = pe.Var(range(self.num_y2), bounds=(-np.infty, np.infty))
        # upper level
        bp.objctv = pe.Objective(expr=self.cu2 @ bp.x2 + self.cu3 @ bp.x3 +
                                                       self.du1 @ bp.y1 + self.du2 @ bp.y2,
                                              sense=pe.minimize)
        bp.cstsUL = pe.ConstraintList()
        for i in range(len(self.Au3[:,0])):
            bp.cstsUL.add(expr=self.Au2[i,:] @ bp.x2 + self.Au3[i,:] @ bp.x3 +
                                           self.Bu1[i,:] @ bp.y1 + self.Bu2[i,:] @ bp.y2 <= self.hu[i])
        # lower level   ---   all entries of y1 and y2 must be active in the lower level !!!!
        bp.LL = pao.SubModel(fixed=bp.x3)
        bp.LL.objctv = pe.Objective(expr=self.dl1 @ bp.y1 + self.dl2 @ bp.y2, sense=pe.maximize)
        bp.LL.cstsLL = pe.ConstraintList()
        for i in range(len(self.Al3[:,0])):
            bp.LL.cstsLL.add(expr=self.Al3[i,:] @ bp.x3 + self.Bl1[i,:] @ bp.y1 + self.Bl2[i,:] @ bp.y2 <= self.hl[i])
        # solving
        T1 = time.time()
        with pao.Solver('pao.pyomo.MIBS') as solver:
            results = solver.solve(bp, tee=True)  # , options={'threads': 4}
        T2 = time.time()
        self.time_MIBS += T2 - T1
        if results.solver.best_feasible_objective == None:
            exit('Error: MIBS failed')
        # solution
        value_x1 = np.zeros(self.num_x1)
        for i in range(self.num_x1):
            value_x1[i] = pe.value(bp.x1[i])
        value_x2 = np.zeros(self.num_x2)
        for i in range(self.num_x2):
            value_x2[i] = pe.value(bp.x2[i])
        value_x3 = np.zeros(self.num_x3)
        for i in range(self.num_x3):
            value_x3[i] = pe.value(bp.x3[i])
        value_y1 = np.zeros(self.num_y1)
        for i in range(self.num_y1):
            value_y1[i] = pe.value(bp.y1[i])
        value_y2 = np.zeros(self.num_y2)
        for i in range(self.num_y2):
            value_y2[i] = pe.value(bp.y2[i])
        # check feasibility
        self.objctv_upper_MIBS_original = self.cu1 @ value_x1 + self.cu2 @ value_x2 + self.cu3 @ value_x3 + self.du1 @ value_y1 + self.du2 @ value_y2
        self.objctv_lower_MIBS_original = self.dl1 @ value_y1 + self.dl2 @ value_y2
        value_g = self.dl1 @ value_y1 + self.dl2 @ value_y2
        value_phi, _ = self.getPhi(value_x3)
        if abs(value_g - value_phi) >= 1e-6:
            value_x1, value_x2, value_y1, value_y2, _ = self.getFeasible(value_x3, value_phi)
        value_obj_upper = self.cu1 @ value_x1 + self.cu2 @ value_x2 + self.cu3 @ value_x3 + self.du1 @ value_y1 + self.du2 @ value_y2
        value_obj_lower = self.dl1 @ value_y1 + self.dl2 @ value_y2
        return np.int_(np.round(value_x1)), value_x2, np.int_(np.round(value_x3)), np.int_(np.round(value_y1)), value_y2, value_obj_upper, value_obj_lower

    def solvebyNN(self, num_samples_target):
        self.flag_buffer_upper = True
        _, _, _, _, _, value_obj_upper_HPR, _ = self.solvebyHPR()
        value_UB = value_obj_upper_HPR
        for Ind_iter in range(self.num_iter_enhanced):
            # sampling
            T1 = time.time()
            self.samples = self.samplesGen(num_samples_target, self.num_sampling, value_UB, self.max_repeated)
            # self.samples = self.samplesRead('samples.xlsx')
            T2 = time.time()
            self.time_NN_sampling += T2 - T1
            num_samples = len(self.samples[:, 0])
            if num_samples < self.num_x3 * 2:
                exit('insufficient samples')

            # training
            num_layer = 2
            num_input = self.num_x3 * (2 if self.flag_doubleX else 1)
            num_hidden = calculateNumHidden(num_samples, num_layer, num_input, self.flag_ISNN)
            num_output = 1
            self.NNArchitecture = [num_input, num_hidden, num_hidden, num_output]
            NN = NNmodel(self.NNArchitecture, self.flag_ISNN, self.flag_doubleX)
            T1 = time.time()
            NN.train(self.samples[:, 0:-1], self.num_epoch)  # -num_samples
            # NN.readParameters('NNparametersISNN.xlsx' if NN.flag_ISNN else 'NNparametersGNN.xlsx')
            T2 = time.time()
            self.time_NN_training += T2 - T1

            # solving
            T1 = time.time()
            if not self.flag_NNSolveByCut:
                value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper, value_obj_lower = self.solutionByMIP(NN)
            else:
                value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper, value_obj_lower = self.solutionByCut(NN)
            T2 = time.time()
            self.time_NN_solving += T2 - T1
            value_UB = np.minimum(np.min(self.samples[:,-1]), value_obj_upper)

        if value_obj_upper > np.min(self.samples[:,-1]):
            idx = np.argmin(self.samples[:, -1])
            value_x3 = self.samples[idx, 0:-2]
            value_phi = self.samples[idx, -2]
            T1 = time.time()
            value_x1, value_x2, value_y1, value_y2, _ = self.getFeasible(value_x3, value_phi)
            T2 = time.time()
            self.time_NN_checking += T2 - T1
            value_obj_upper = self.cu1 @ value_x1 + self.cu2 @ value_x2 + self.cu3 @ value_x3 + self.du1 @ value_y1 + self.du2 @ value_y2
            value_obj_lower = self.dl1 @ value_y1 + self.dl2 @ value_y2
        return np.int_(np.round(value_x1)), value_x2, np.int_(np.round(value_x3)), np.int_(np.round(value_y1)), value_y2, value_obj_upper, value_obj_lower

    def samplesGen(self, num_samples_target, num_sampling, value_UB, max_repeated):
        # parameters
        UB = cp.Parameter()
        h = cp.Parameter(self.num_x3)
        Q = cp.Parameter((self.num_x3, self.num_x3))
        if self.num_x1 == 0:
            x1 = np.zeros(self.num_x1)
        else:
            x1 = cp.Variable(self.num_x1, boolean=True)
        if self.num_x2 == 0:
            x2 = np.zeros(self.num_x2)
        else:
            x2 = cp.Variable(self.num_x2)
        x3 = cp.Variable(self.num_x3, boolean=True)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)

        # sampling problem
        csts_sampling = [
            self.Au1 @ x1 + self.Au2 @ x2 + self.Au3 @ x3 + self.Bu1 @ y1 + self.Bu2 @ y2 <= self.hu,
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            # self.cu1 @ x1 + self.cu2 @ x2 + self.cu3 @ x3 + self.du1 @ y1 + self.du2 @ y2 <= UB,
        ]
        obj_sampling = h @ x3 + x3.T @ cp.psd_wrap(Q) @ x3
        prbl_sampling = cp.Problem(cp.Minimize(obj_sampling), csts_sampling)

        # sampling
        count_invalid = 0
        count_repeated = 0
        count_infeasible = 0
        num_infeasible = 0
        for i in range(num_sampling):
            if (len(self.buffer_x3) >= num_samples_target) | (len(self.buffer_x3) >= 2**self.num_x3) | (count_repeated > max_repeated) | (count_infeasible > max_repeated/2):
                if count_repeated > max_repeated:
                    print('Sampling terminated: Repeated')
                if count_infeasible > max_repeated/2:
                    print('Sampling terminated: Infeasible')
                print('Sampling:', i, '/', num_sampling, '| sample:', len(self.buffer_x3), '/', 2 ** self.num_x3, '| Infeasible:', num_infeasible, '/', 2 ** self.num_x3, '| Best UB:', np.round(value_UB, 4))
                break
            UB.value = value_UB
            h.value = 2 * np.random.rand(self.num_x3) - 1
            D = np.diag(np.random.rand(self.num_x3))
            U = orth(np.random.rand(self.num_x3, self.num_x3))
            Q.value = U.T @ D @ U
            prbl_sampling.solve(solver=cp.GUROBI, verbose=False)  # , threads=4, MIPFocus=1, MIPGap=20
            if prbl_sampling.status != 'optimal':
                print('Sampling', i+1, '/', num_sampling, ': Invalid sampling (sample', len(self.buffer_x3), '/', 2**self.num_x3, ', UB =', np.round(value_UB, 4), ', count_invalid =', count_invalid, ')')
                continue
            else:
                count_invalid = 0
            value_x3 = np.int_(np.round(x3.value))
            if list(value_x3) in self.buffer_x3:
                count_repeated += 1
                print('Sampling', i+1, '/', num_sampling, ': Repeated sample (sample', len(self.buffer_x3), '/', 2**self.num_x3, ', UB =', np.round(value_UB, 4), ', count_repeat =', count_repeated, ')')
                continue
            else:
                count_repeated = 0
            value_phi, _ = self.getPhi(value_x3)
            if value_phi == None:
                count_infeasible += 1
                num_infeasible += 1
                print('Sampling', i+1, '/', num_sampling, ': Infeasible sample (sample', len(self.buffer_x3), '/', 2**self.num_x3, ', UB =', np.round(value_UB, 4), ', count_infeasible =', count_infeasible, ')')
                continue
            else:
                count_infeasible = 0
            print('Sampling', i+1, '/', num_sampling, ': Optimal (sample', len(self.buffer_x3), '/', 2**self.num_x3, ', UB =', np.round(value_UB, 4), ')')
            value_UB = min(value_UB, self.buffer_obj_upper[-1] if len(self.buffer_obj_upper) != 0 else 1e6)
        # export samples
        self.samplesSave(self.buffer_x3, self.buffer_phi, self.buffer_obj_upper)
        return np.hstack([np.array(self.buffer_x3),np.array(self.buffer_phi).reshape(-1,1),np.array(self.buffer_obj_upper).reshape(-1,1)])

    def samplesSave(self, samples, labels, UB):
        FileName = 'samples.xlsx'
        writer = pd.ExcelWriter(FileName)
        df_samples = pd.DataFrame(samples)
        df_samples.to_excel(writer, 'samples', header=None, index=False)
        df_labels = pd.DataFrame(labels)
        df_labels.to_excel(writer, 'samples', startcol=self.num_x3, header=None, index=False)
        df_UB = pd.DataFrame(UB)
        df_UB.to_excel(writer, 'samples', startcol=self.num_x3+1, header=None, index=False)
        writer.close()
        print('successfully export samples as ---', FileName, '---')

    def samplesRead(self, FileName):
        data = pd.read_excel(io=FileName, sheet_name=None, header=None)
        samples = data['samples'].values
        return samples

    def solutionByMIP(self, NN):
        # parameters
        bigM = 10

        # model
        if self.num_x1 == 0:
            x1 = np.zeros(self.num_x1)
        else:
            x1 = cp.Variable(self.num_x1, boolean=True)
        if self.num_x2 == 0:
            x2 = np.zeros(self.num_x2)
        else:
            x2 = cp.Variable(self.num_x2)
        x3 = cp.Variable(self.num_x3, boolean=True)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        phi = cp.Variable()
        z = []
        z_temp = []
        z_sign = []
        for i in range(NN.num_layer_hidden + 2):
            z += [cp.Variable(NN.Architecture[i])]
            z_temp += [cp.Variable(NN.Architecture[i])]
            z_sign += [cp.Variable(NN.Architecture[i], boolean=True)]
        objctv = self.cu1 @ x1 + self.cu2 @ x2 + self.cu3 @ x3 + self.du1 @ y1 + self.du2 @ y2
        csts = [
            self.Au1 @ x1 + self.Au2 @ x2 + self.Au3 @ x3 + self.Bu1 @ y1 + self.Bu2 @ y2 <= self.hu,
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            self.dl1 @ y1 + self.dl2 @ y2 >= phi,
            z[-1][0] * (NN.label_max[0] - NN.label_min[0]) + NN.label_min[0] == phi
        ]
        if NN.flag_doubleX:
            csts += [z[0] == cp.hstack([x3, 1 - x3])]
        else:
            csts += [z[0] == x3]
        for i in range(NN.num_layer_hidden + 1):
            if i == 0:
                csts += [
                    z_temp[i+1] == NN.w[i] @ z[i] + NN.b[i][:, 0],
                    0 <= z[i+1], z[i+1] <= bigM * z_sign[i+1],
                    z_temp[i+1] <= z[i+1], z[i+1] <= z_temp[i+1] + bigM * (1 - z_sign[i+1])
                ]
            else:
                csts += [
                    z_temp[i+1] == NN.w[i] @ cp.hstack([z[i], z[0]]) + NN.b[i][:, 0],
                    0 <= z[i+1], z[i+1] <= bigM * z_sign[i+1],
                    z_temp[i+1] <= z[i+1], z[i+1] <= z_temp[i+1] + bigM * (1 - z_sign[i+1])
                ]

        # solve
        prbl = cp.Problem(cp.Minimize(objctv), csts)
        prbl.solve(solver=cp.GUROBI, verbose=True)  # , threads=4

        # save results
        if prbl.status != 'optimal':
            exit('SolveByNN: infeasible')
        if self.num_x1 == 0:
            value_x1 = x1
        else:
            value_x1 = x1.value
        if self.num_x2 == 0:
            value_x2 = x2
        else:
            value_x2 = x2.value
        value_x3 = x3.value
        if self.num_y1 == 0:
            value_y1 = y1
        else:
            value_y1 = y1.value
        value_y2 = y2.value
        # correct solution
        value_g = self.dl1 @ value_y1 + self.dl2 @ value_y2
        value_phi, _ = self.getPhi(value_x3)
        if abs(value_g - value_phi) >= 1e-6:
            value_x1, value_x2, value_y1, value_y2, _ = self.getFeasible(value_x3, value_phi)
        value_obj_upper = self.cu1 @ value_x1 + self.cu2 @ value_x2 + self.cu3 @ value_x3 + self.du1 @ value_y1 + self.du2 @ value_y2
        value_obj_lower = self.dl1 @ value_y1 + self.dl2 @ value_y2
        return np.int_(np.round(value_x1)), value_x2, np.int_(np.round(value_x3)), np.int_(np.round(value_y1)), value_y2, value_obj_upper, value_obj_lower

    def solutionByCut(self, NN):
        def getPi_ISNN(value_z0):
            num_z0 = len(value_z0)
            phi = NN.predict(value_z0)[0]
            phi1 = NN.predict(np.ones(num_z0))[0]
            pi = np.zeros(num_z0)
            for i in range(num_z0):
                ei = np.zeros(num_z0)
                ei[i] = 1
                if value_z0[i] == 0:
                    pi[i] = NN.predict(value_z0 + ei)[0] - phi
                if value_z0[i] == 1:
                    pi[i] = phi1 - NN.predict(np.ones(num_z0) - ei)[0]
            return pi

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                name_z0 = (f"z0[{i}]" for i in range(self.num_x3 * (2 if self.flag_doubleX else 1)))
                temp_z0 = [model.getVarByName(i) for i in name_z0]
                value_z0 = np.int_(np.round(np.array(model.cbGetSolution(temp_z0))))
                temp_g = model.getVarByName('g')
                pi = getPi_ISNN(value_z0)
                value_phi = NN.predict(value_z0)[0]
                idx0 = np.where(value_z0==0)[0]
                idx1 = np.where(value_z0==1)[0]
                model.cbLazy(temp_g >= value_phi + sum(pi[i] * temp_z0[i] for i in idx0) - sum(pi[i] * (1 - temp_z0[i]) for i in idx1))
                return

        if not self.flag_ISNN:
            exit('Error: solutionByCut inapplicable')

        # model
        LB = -1e6
        model = gp.Model()
        x1 = model.addMVar(shape=self.num_x1, vtype=GRB.BINARY, name='x1')
        x2 = model.addMVar(shape=self.num_x2, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='x2')
        x3 = model.addMVar(shape=self.num_x3, vtype=GRB.BINARY, name='x3')
        y1 = model.addMVar(shape=self.num_y1, vtype=GRB.BINARY, name='y1')
        y2 = model.addMVar(shape=self.num_y2, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='y2')
        g = model.addVar(vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='g')
        z0 = model.addMVar(shape=self.num_x3 * (2 if self.flag_doubleX else 1), vtype=GRB.CONTINUOUS, name='z0')
        model.setObjective(self.cu1 @ x1 + self.cu2 @ x2 + self.cu3 @ x3 + self.du1 @ y1 + self.du2 @ y2, GRB.MINIMIZE)
        model.addConstrs((self.Au1[i,:] @ x1 + self.Au2[i,:] @ x2 + self.Au3[i,:] @ x3 +
                                     self.Bu1[i,:] @ y1 + self.Bu2[i,:] @ y2 <= self.hu[i] for i in range(len(self.Au3[:,0]))))
        model.addConstrs((self.Al3[i,:] @ x3 +
                                     self.Bl1[i,:] @ y1 + self.Bl2[i,:] @ y2 <= self.hl[i] for i in range(len(self.Al3[:,0]))))
        model.addConstr(self.dl1 @ y1 + self.dl2 @ y2 == g)
        model.addConstr(g >= LB)
        if NN.flag_doubleX:
            model.addConstrs((z0[i] == x3[i] for i in range(self.num_x3)))
            model.addConstrs((z0[self.num_x3+i] == 1 - x3[i] for i in range(self.num_x3)))
        else:
            model.addConstrs((z0[i] == x3[i] for i in range(self.num_x3)))
        # model._Ind_indicator = 0
        model.Params.lazyConstraints = 1
        model.optimize(callback)
        if model.Status != 2:
            exit('Error: solutionByCut infeasible')
        value_x1 = x1.X
        value_x2 = x2.X
        value_x3 = x3.X
        value_y1 = y1.X
        value_y2 = y2.X
        # correct solution
        value_g = self.dl1 @ value_y1 + self.dl2 @ value_y2
        value_phi, _ = self.getPhi(value_x3)
        if abs(value_g - value_phi) >= 1e-6:
            value_x1, value_x2, value_y1, value_y2, _ = self.getFeasible(value_x3, value_phi)
        value_obj_upper = self.cu1 @ value_x1 + self.cu2 @ value_x2 + self.cu3 @ value_x3 + self.du1 @ value_y1 + self.du2 @ value_y2
        value_obj_lower = self.dl1 @ value_y1 + self.dl2 @ value_y2
        return np.int_(np.round(value_x1)), value_x2, np.int_(np.round(value_x3)), np.int_(np.round(value_y1)), value_y2, value_obj_upper, value_obj_lower

    def solvebyCP(self):
        def MoveIncumbent(model):
            name_x1 = (f"x1[{i}]" for i in range(self.num_x1))
            temp_x1 = [model.getVarByName(i) for i in name_x1]
            name_x2 = (f"x2[{i}]" for i in range(self.num_x2))
            temp_x2 = [model.getVarByName(i) for i in name_x2]
            name_x3 = (f"x3[{i}]" for i in range(self.num_x3))
            temp_x3 = [model.getVarByName(i) for i in name_x3]
            name_y1 = (f"y1[{i}]" for i in range(self.num_y1))
            temp_y1 = [model.getVarByName(i) for i in name_y1]
            name_y2 = (f"y2[{i}]" for i in range(self.num_y2))
            temp_y2 = [model.getVarByName(i) for i in name_y2]
            model.cbSetSolution(temp_x1, self.good_value_x1)
            model.cbSetSolution(temp_x2, self.good_value_x2)
            model.cbSetSolution(temp_x3, self.good_value_x3)
            model.cbSetSolution(temp_y1, self.good_value_y1)
            model.cbSetSolution(temp_y2, self.good_value_y2)
            self.flag_incumbent_move = True
            print('Heuristic UB applied:', np.round(model.cbUseSolution(), 4))
            return

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                name_x3 = (f"x3[{i}]" for i in range(self.num_x3))
                temp_x3 = [model.getVarByName(i) for i in name_x3]
                value_x3 = np.int_(np.round(np.array(model.cbGetSolution(temp_x3))))
                if self.flag_incumbent_move:
                    self.flag_incumbent_move = False
                    if list(value_x3) in self.history_value_x3_lazy:
                        return
                    else:
                        self.flag_good_UB = True
                self.history_value_x3_lazy.append(list(value_x3))
                temp_g = model.getVarByName('g')
                if self.flag_LI | self.flag_ALI:
                    t1 = time.time()
                    value_phi, value_y1_fixed = self.getPhi(value_x3)
                    if len(self.idx_y1_fixed) == 0:
                        if self.flag_LI:
                            lam = self.lamU * (1 - value_x3) + self.lamL * value_x3
                            model.cbLazy(temp_g >= value_phi - lam @ (temp_x3 - value_x3))
                        if self.flag_ALI:
                            rho = self.rho
                            model.cbLazy(temp_g >= value_phi - rho * (np.ones(self.num_x3) @ value_x3 + np.ones(self.num_x3) @ temp_x3 - 2 * value_x3 @ temp_x3))
                    else:
                        if self.flag_quasi_relaxed:
                            lam = self.lamU * (1 - value_x3) + self.lamL * value_x3
                            rho = self.rho
                        else:
                            lamU, lamL = self.getLam_quasiLL_exact(value_y1_fixed)
                            lam = lamU * (1 - value_x3) + lamL * value_x3
                            rho = np.maximum(np.max(lamU), -np.min(lamL))
                        # value_varphi = self.getVarPhi(value_x3,value_y1_fixed)
                        t2 = time.time()
                        self.time_Lagrangian += t2 - t1
                        if self.flag_LI:
                            model.cbLazy(temp_g >= value_phi - lam @ (temp_x3 - value_x3))
                            # model.cbLazy(temp_g >= self.dl1[list(self.idx_y1_fixed)] @ value_y1_fixed + value_varphi - lam @ (temp_x3 - value_x3))
                        if self.flag_ALI:
                            model.cbLazy(temp_g >= value_phi - rho * (np.ones(self.num_x3) @ value_x3 + np.ones(self.num_x3) @ temp_x3 - 2 * value_x3 @ temp_x3))
                            # model.cbLazy(temp_g >= self.dl1[list(self.idx_y1_fixed)] @ value_y1_fixed + value_varphi - rho * (np.ones(self.num_x3) @ value_x3 + np.ones(self.num_x3) @ temp_x3 - 2 * value_x3 @ temp_x3))
                if self.flag_SI:
                    t1 = time.time()
                    if len(self.idx_y1_fixed) == 0:
                        value_y1_fixed = np.zeros(len(self.idx_y1_fixed))
                    else:
                        _, value_y1_fixed = self.getPhi(value_x3)
                    if self.flag_LowerLevel == 1:
                        pi = self.getPi(value_x3, value_y1_fixed)
                        value_varphi = self.getVarPhi(value_x3, value_y1_fixed)
                        t2 = time.time()
                        self.time_EPI_pi += t2 - t1
                        idx0 = np.where(value_x3==0)[0]
                        idx1 = np.where(value_x3==1)[0]
                        model.cbLazy(temp_g >= self.dl1[list(self.idx_y1_fixed)] @ value_y1_fixed + value_varphi + sum(pi[i] * temp_x3[i] for i in idx0) - sum(pi[i] * (1 - temp_x3[i]) for i in idx1))
                    if self.flag_LowerLevel == 2:
                        pi = self.getPi(value_x3, value_y1_fixed)
                        value_varphi0 = self.getVarPhi(np.zeros(self.num_x3), value_y1_fixed)
                        t2 = time.time()
                        self.time_EPI_pi += t2 - t1
                        model.cbLazy(temp_g >= self.dl1[list(self.idx_y1_fixed)] @ value_y1_fixed + value_varphi0 + pi @ temp_x3)
                if self.flag_heuristic_LB & False:
                    name_yy1 = (f"yy1[{i}]" for i in range(self.num_y1))
                    temp_yy1 = [model.getVarByName(i) for i in name_yy1]
                    value_yy1 = np.int_(np.round(np.array(model.cbGetSolution(temp_yy1))))
                    # value_phi, _ = self.getPhi(value_x3)
                    value_varphi = self.getVarPhi_heuristic_LB(value_x3,value_yy1)
                    # model.cbLazy(temp_g >= self.dl1 @ temp_yy1 + value_varphi - self.heuristic_LB_rho * (
                    #         np.ones(self.num_x3) @ value_x3 + np.ones(self.num_x3) @ temp_x3 - 2 * value_x3 @ temp_x3 +
                    #         np.ones(self.num_y1) @ value_yy1 + np.ones(self.num_y1) @ temp_yy1 - 2 * value_yy1 @ temp_yy1))
                    value_x3yy1 = np.hstack([value_x3, value_yy1])
                    lam = self.heuristic_LB_lamU * (1 - value_x3yy1) + self.heuristic_LB_lamL * value_x3yy1
                    model.cbLazy(temp_g >= self.dl1 @ temp_yy1 + value_varphi - lam[0:self.num_x3] @ (temp_x3 - value_x3) - lam[self.num_x3:] @ (temp_yy1 - value_yy1))
                    # if self.flag_LowerLevel == 0:
                    # else:
                    #     t1 = time.time()
                    #     pi = self.getPi_quasi(value_x3, value_yy1)
                    #     value_varphi0 = self.getVarPhi(np.zeros(self.num_x3), np.zeros(self.num_y1))
                    #     t2 = time.time()
                    #     self.time_EPI_pi_heuristic += t2 - t1
                    #     model.cbLazy(temp_g >= self.dl1 @ temp_yy1 + value_varphi0 + pi[0:self.num_x3] @ temp_x3 + pi[self.num_x3:] @ temp_yy1)
                if self.flag_no_good:
                    value_phi, _ = self.getPhi(value_x3)
                    value_x1, value_x2, value_y1, value_y2, value_obj_upper = self.getFeasible(value_x3, value_phi)
                    if value_obj_upper < self.good_UB:
                        self.SaveGoodUB(value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper)
                    else:
                        if (value_obj_upper > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                            self.no_good_value_x3.append(value_x3)
                            self.no_good_value_x3_history.append(list(value_x3))
                if self.flag_JI:
                    # if model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                    #     self.count_JI += 1
                    #     if self.count_JI % 10 == 0:
                    # temp_g = model.getVarByName("g")
                    # value_g = model.cbGetNodeRel(temp_g)
                    # name_x3 = (f"x3[{i}]" for i in range(self.num_x3))
                    # temp_x3 = [model.getVarByName(i) for i in name_x3]
                    # value_x3 = np.array(model.cbGetNodeRel(temp_x3))
                    value_g = model.cbGetSolution(temp_g)
                    self.LL_JI_x3.value = value_x3
                    self.LL_JI_prbl.solve(solver=cp.GUROBI, verbose=False)
                    if self.LL_JI_prbl.status != 'optimal':
                        exit('Error')
                    value_y1 = list(self.LL_JI_prbl.solution.primal_vars.values())[0]
                    self.MILP_JI_y1.value = value_y1
                    self.MILP_JI_x3.value = value_x3
                    t1 = time.time()
                    self.MILP_JI_prbl.solve(solver=cp.GUROBI, verbose=False)
                    if self.MILP_JI_prbl.status != 'optimal':
                        exit('Error')
                    t2 = time.time()
                    self.time_JI += t2 - t1
                    value_alpha = list(self.MILP_JI_prbl.solution.primal_vars.values())[0]
                    value_beta = list(self.MILP_JI_prbl.solution.primal_vars.values())[1]
                    value_obj = self.MILP_JI_prbl.solution.opt_val
                    if value_obj > value_g:
                        # model.cbCut(temp_g >= value_alpha @ temp_x3 + value_beta)
                        model.cbLazy(temp_g >= value_alpha @ temp_x3 + value_beta)
            if where == GRB.Callback.MIPNODE:
                if self.flag_no_good & (len(self.no_good_value_x3) != 0):
                        name_x3 = (f"x3[{i}]" for i in range(self.num_x3))
                        temp_x3 = [model.getVarByName(i) for i in name_x3]
                        for i in range(len(self.no_good_value_x3)):
                            value_x3 = self.no_good_value_x3[i]
                            idx0 = np.where(value_x3 == 0)[0]
                            idx1 = np.where(value_x3 == 1)[0]
                            model.cbCut(sum(temp_x3[i] for i in idx0) + sum(1 - temp_x3[i] for i in idx1) >= 1)
                        self.no_good_value_x3 = []
                self.heuristic_count += 1
                if self.flag_heuristic_UB_incumbent & (self.heuristic_count % self.heuristic_UB_frequency == 0):
                    self.getUB_heuristic(NN_UB)
                if self.flag_good_UB & self.flag_incumbent & (self.good_UB < model.cbGet(gp.GRB.Callback.MIPNODE_OBJBST) - 1e-8):
                    MoveIncumbent(model)
                    self.flag_good_UB = False
            return

        LB = -1e6
        model = gp.Model()
        x1 = model.addMVar(shape=self.num_x1, vtype=GRB.BINARY, name='x1')
        x2 = model.addMVar(shape=self.num_x2, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='x2')
        x3 = model.addMVar(shape=self.num_x3, vtype=GRB.BINARY, name='x3')
        y1 = model.addMVar(shape=self.num_y1, vtype=GRB.BINARY, name='y1')
        y2 = model.addMVar(shape=self.num_y2, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='y2')
        g = model.addVar(vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='g')
        model.setObjective(self.cu1 @ x1 + self.cu2 @ x2 + self.cu3 @ x3 + self.du1 @ y1 + self.du2 @ y2, GRB.MINIMIZE)
        model.addConstrs((self.Au1[i,:] @ x1 + self.Au2[i,:] @ x2 + self.Au3[i,:] @ x3 +
                                     self.Bu1[i,:] @ y1 + self.Bu2[i,:] @ y2 <= self.hu[i] for i in range(len(self.Au3[:,0]))))
        model.addConstrs((self.Al3[i,:] @ x3 +
                                     self.Bl1[i,:] @ y1 + self.Bl2[i,:] @ y2 <= self.hl[i] for i in range(len(self.Al3[:,0]))))
        model.addConstr(self.dl1 @ y1 + self.dl2 @ y2 == g)
        model.addConstr(g >= LB)

        if self.flag_heuristic_UB:
            # initialize NN
            num_layer = 2
            num_input = self.num_x3 * (2 if self.flag_doubleX else 1)
            num_sample_guess = min(self.heuristic_UB_num_samples_guess, 2**self.num_x3)
            num_hidden = calculateNumHidden(num_sample_guess, num_layer, num_input, self.flag_ISNN)
            num_output = 1
            self.heuristic_UB_NNArchitecture = [num_input, num_hidden, num_hidden, num_output]
            NN_UB = NNmodel(self.heuristic_UB_NNArchitecture, self.flag_ISNN, self.flag_doubleX)
            # initial heuristic
            value_x1_heuristic, value_x2_heuristic, value_x3_heuristic, value_y1_heuristic, value_y2_heuristic, value_obj_upper_heuristic, value_obj_lower_heuristic = self.getUB_heuristic(NN_UB)
            # warm start
            if self.flag_heuristic_UB_warmstart:
                x1.start = np.int_(np.round(value_x1_heuristic))
                x2.start = value_x2_heuristic
                x3.start = np.int_(np.round(value_x3_heuristic))
                y1.start = np.int_(np.round(value_y1_heuristic))
                y2.start = value_y2_heuristic

        # if self.flag_heuristic_LB:
        #     # sampling
        #     T1 = time.time()
        #     _, _, _, _, _, value_obj_upper_HPR, _ = self.solvebyHPR()
        #     value_UB = value_obj_upper_HPR
        #     self.samplesGen(self.heuristic_LB_num_samples_target, self.heuristic_LB_num_sampling, value_UB, self.heuristic_LB_max_repeated)
        #     T2 = time.time()
        #     self.time_CP_heuristic_sampling += T2 - T1
        #     x3_sample = np.array(self.buffer_x3)
        #     y1_sample = np.array(self.buffer_y1)
        #     # obj_upper_sample = np.array(self.buffer_obj_upper)
        #     # idx = np.argsort(obj_upper_sample)
        #     # num = max(3, self.heuristic_LB_K)
        #     # x3_sample = x3_sample[idx[:num],:]
        #     # y1_sample = y1_sample[idx[:num],:]
        #     num_sample = len(x3_sample[:,0])
        #     # variables
        #     yy1 = model.addMVar(shape=self.num_y1, vtype=GRB.BINARY, name='yy1')
        #     alpha = model.addMVar(shape=num_sample, vtype=GRB.BINARY, name='alpha')
        #     yy1_kNN = model.addMVar(shape=self.num_y1, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='yy1_kNN')
        #     # constraints
        #     model.addConstrs((yy1[i] - 0.5 <= yy1_kNN[i] for i in range(self.num_y1)))
        #     model.addConstrs((yy1_kNN[i] <= yy1[i] + 0.5 for i in range(self.num_y1)))
        #     model.addConstr(alpha @ y1_sample == self.heuristic_LB_K * yy1_kNN)
        #     model.addConstr(alpha @ np.ones(num_sample) == self.heuristic_LB_K)
        #     model.addConstrs((
        #         (x3_sample[i,:] - x3_sample[j,:]) @ (1 - 2 * x3) <= self.num_x3 * (1 - alpha[i] + alpha[j])
        #         for i in range(num_sample) for j in range(num_sample) if i != j))

        # if self.flag_heuristic_LB:
        #     bigM = 1000
        #     NN_LB = self.getLB_heuristic()
        #     # variables
        #     yy1 = model.addMVar(shape=self.num_y1, vtype=GRB.BINARY, name='yy1')
        #     z = []
        #     z_temp = []
        #     z_sign0 = []
        #     z_sign1 = []
        #     for i in range(NN_LB.num_layer_hidden + 2):
        #         z += [model.addMVar(shape=NN_LB.Architecture[i], vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty)]
        #         z_temp += [model.addMVar(shape=NN_LB.Architecture[i], vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty)]
        #         z_sign0 += [model.addMVar(shape=NN_LB.Architecture[i], vtype=GRB.BINARY)]
        #         z_sign1 += [model.addMVar(shape=NN_LB.Architecture[i], vtype=GRB.BINARY)]
        #
        #     # constraints
        #     model.addConstrs((-bigM * (1 - yy1[i]) <= z[-1][i] for i in range(self.num_y1)))
        #     model.addConstrs((z[-1][i] <= bigM * yy1[i] for i in range(self.num_y1)))
        #     if NN_LB.flag_doubleX:
        #         model.addConstrs((z[0][i] == x3[i] for i in range(self.num_x3)))
        #         model.addConstrs((z[0][i+self.num_x3] == 1 - x3[i] for i in range(self.num_x3)))
        #     else:
        #         model.addConstrs((z[0][i] == x3[i] for i in range(self.num_x3)))
        #     for i in range(NN_LB.num_layer_hidden + 1):
        #         if i == 0:
        #             model.addConstrs((z_temp[i+1][j] == NN_LB.w[i][j,:] @ z[i] + NN_LB.b[i][j, 0] for j in range(NN_LB.Architecture[i+1])))
        #         else:
        #             model.addConstrs((z_temp[i+1][j] == NN_LB.w[i][j,0:NN_LB.Architecture[i]] @ z[i] + NN_LB.w[i][j,NN_LB.Architecture[i]:] @ z[0] + NN_LB.b[i][j, 0] for j in range(NN_LB.Architecture[i+1])))
        #         if i != NN_LB.num_layer_hidden:
        #             model.addConstrs((-bigM * z_sign1[i+1][j] + (0.2 * z_temp[i+1][j] + 0.5) <= z[i+1][j] for j in range(NN_LB.Architecture[i+1])))
        #             model.addConstrs((z[i+1][j] <= (0.2 * z_temp[i+1][j] + 0.5) + bigM * z_sign0[i+1][j] for j in range(NN_LB.Architecture[i+1])))
        #             model.addConstrs((-(1 - z_sign1[i+1][j]) + 1 <= z[i+1][j] for j in range(NN_LB.Architecture[i+1])))
        #             model.addConstrs((z[i+1][j] <= (1 - z_sign0[i+1][j]) for j in range(NN_LB.Architecture[i+1])))
        #             # model.addConstrs((z_sign0[i+1][j] + z_sign1[i+1][j] <= 1 for j in range(NN_LB.Architecture[i+1])))
        #         else:
        #             model.addConstrs((z[i + 1][j] == z_temp[i + 1][j] for j in range(NN_LB.Architecture[i + 1])))

        if self.flag_heuristic_LB:
            bigM = 1000
            NN_LB = self.getLB_heuristic()
            # variables
            yy1 = model.addMVar(shape=self.num_y1, vtype=GRB.BINARY, name='yy1')
            # model.addConstrs((yy1[j] == 1 - x3[j] for j in range(self.num_y1)))
            z = []
            z_temp = []
            z_sign0 = []
            z_sign1 = []
            for i in range(NN_LB.num_layer_hidden + 2):
                z += [model.addMVar(shape=NN_LB.Architecture[i], vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty)]
                z_temp += [model.addMVar(shape=NN_LB.Architecture[i], vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty)]
                z_sign0 += [model.addMVar(shape=NN_LB.Architecture[i], vtype=GRB.BINARY)]
                z_sign1 += [model.addMVar(shape=NN_LB.Architecture[i], vtype=GRB.BINARY)]
            # constraints
            model.addConstrs((-bigM * (1 - yy1[i]) <= z[-1][i] for i in range(self.num_y1)))
            model.addConstrs((z[-1][i] <= bigM * yy1[i] for i in range(self.num_y1)))
            if NN_LB.flag_doubleX:
                model.addConstrs((z[0][i] == x3[i] for i in range(self.num_x3)))
                model.addConstrs((z[0][i+self.num_x3] == 1 - x3[i] for i in range(self.num_x3)))
            else:
                model.addConstrs((z[0][i] == x3[i] for i in range(self.num_x3)))
            for i in range(NN_LB.num_layer_hidden + 1):
                if i == 0:
                    model.addConstrs((z_temp[i+1][j] == NN_LB.w[i][j,:] @ z[i] + NN_LB.b[i][j, 0] for j in range(NN_LB.Architecture[i+1])))
                else:
                    model.addConstrs((z_temp[i+1][j] == NN_LB.w[i][j,0:NN_LB.Architecture[i]] @ z[i] + NN_LB.w[i][j,NN_LB.Architecture[i]:] @ z[0] + NN_LB.b[i][j, 0] for j in range(NN_LB.Architecture[i+1])))
                if i != NN_LB.num_layer_hidden:
                    model.addConstrs((-bigM * z_sign1[i+1][j] + (0.2 * z_temp[i+1][j] + 0.5) <= z[i+1][j] for j in range(NN_LB.Architecture[i+1])))
                    model.addConstrs((z[i+1][j] <= (0.2 * z_temp[i+1][j] + 0.5) + bigM * z_sign0[i+1][j] for j in range(NN_LB.Architecture[i+1])))
                    model.addConstrs((-(1 - z_sign1[i+1][j]) + 1 <= z[i+1][j] for j in range(NN_LB.Architecture[i+1])))
                    model.addConstrs((z[i+1][j] <= (1 - z_sign0[i+1][j]) for j in range(NN_LB.Architecture[i+1])))
                    # model.addConstrs((z_sign0[i+1][j] + z_sign1[i+1][j] <= 1 for j in range(NN_LB.Architecture[i+1])))
                else:
                    model.addConstrs((z[i + 1][j] == z_temp[i + 1][j] for j in range(NN_LB.Architecture[i + 1])))
            # constraints
            w = model.addMVar(shape=len(self.hl), vtype=GRB.CONTINUOUS, lb=0, ub=np.infty, name='w')
            wx = model.addMVar(shape=self.num_x3, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='wx')
            wy1 = model.addMVar(shape=self.num_y1, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='wy1')
            model.addConstr(g >= self.dl1 @ yy1 + self.hl @ w - np.ones(self.num_x3) @ wx - np.ones(self.num_y1) @ wy1)
            model.addConstrs((self.Bl2[:,j] @ w == self.dl2[j] for j in range(self.num_y2)))
            model.addConstrs((-bigM * x3[j] <= wx[j] for j in range(self.num_x3)))
            model.addConstrs((wx[j] <= bigM * x3[j] for j in range(self.num_x3)))
            model.addConstrs((-bigM * (1 - x3[j]) + self.Al3[:,j] @ w <= wx[j] for j in range(self.num_x3)))
            model.addConstrs((wx[j] <= bigM * (1 - x3[j]) + self.Al3[:,j] @ w for j in range(self.num_x3)))
            model.addConstrs((-bigM * yy1[j] <= wy1[j] for j in range(self.num_y1)))
            model.addConstrs((wy1[j] <= bigM * yy1[j] for j in range(self.num_y1)))
            model.addConstrs((-bigM * (1 - yy1[j]) + self.Bl1[:,j] @ w <= wy1[j] for j in range(self.num_y1)))
            model.addConstrs((wy1[j] <= bigM * (1 - yy1[j]) + self.Bl1[:,j] @ w for j in range(self.num_y1)))




        self.flag_buffer_upper = False
        # model._Ind_indicator = 0
        model.Params.lazyConstraints = 1
        model.Params.TimeLimit = 3600
        # model.Params.MIPFocus = 0
        T1 = time.time()
        model.optimize(callback)
        T2 = time.time()
        print('Time:', round(T2-T1, 4))
        if (model.Status != 2) & (model.Status != 9):
            print('error')
            return None
        value_x1 = x1.X
        value_x2 = x2.X
        value_x3 = x3.X
        value_y1 = y1.X
        value_y2 = y2.X
        # correct solution
        value_g = self.dl1 @ value_y1 + self.dl2 @ value_y2
        value_phi, _ = self.getPhi(value_x3)
        if abs(value_g - value_phi) >= 1e-6:
            value_x1, value_x2, value_y1, value_y2, _ = self.getFeasible(value_x3, value_phi)
        value_obj_upper = self.cu1 @ value_x1 + self.cu2 @ value_x2 + self.cu3 @ value_x3 + self.du1 @ value_y1 + self.du2 @ value_y2
        value_obj_lower = self.dl1 @ value_y1 + self.dl2 @ value_y2
        return np.int_(np.round(value_x1)), value_x2, np.int_(np.round(value_x3)), np.int_(np.round(value_y1)), value_y2, value_obj_upper, value_obj_lower

    def LowerLevel_fixed(self):
        if self.num_y1 == 0:
            y1_fixed = np.zeros(len(self.idx_y1_fixed))
            x3, prbl = self.LowerLevel()
        else:
            x3 = cp.Parameter(self.num_x3)
            y1 = cp.Variable(self.num_y1, boolean=True)
            y2 = cp.Variable(self.num_y2)
            if len(self.idx_y1_fixed) == 0:
                y1_fixed = np.zeros(len(self.idx_y1_fixed))
            else:
                y1_fixed = cp.Parameter(len(self.idx_y1_fixed))
            csts = [
                self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            ]
            if len(self.idx_y1_fixed) != 0:
                csts += [
                    y1[list(self.idx_y1_fixed)] == y1_fixed
                ]
            prbl = cp.Problem(cp.Maximize(self.dl1 @ y1 - self.dl1[list(self.idx_y1_fixed)] @ y1_fixed + self.dl2 @ y2), csts)
        return x3, y1_fixed, prbl

    def getVarPhi(self, value_x3, value_y1_fixed):
        value_x3y1_fixed = np.hstack([value_x3, value_y1_fixed])
        list_value_x3y1_fixed = list(np.int_(np.round(value_x3y1_fixed)))
        if list_value_x3y1_fixed in self.buffer_x3y1_fixed:
            idx = self.buffer_x3y1_fixed.index(list_value_x3y1_fixed)
            return self.buffer_varphi[idx]
        else:
            if len(self.idx_y1_fixed) == 0:
                value_varphi, _ = self.getPhi(value_x3)
            else:
                self.LL_fixed_x3.value = value_x3
                self.LL_fixed_y1_fixed.value = value_y1_fixed
                t1 = time.time()
                self.LL_fixed_prbl.solve(solver=cp.GUROBI, verbose=False)
                t2 = time.time()
                self.time_varPhi += t2 - t1
                if self.LL_fixed_prbl.status != 'optimal':
                    exit('Error: Failed in getting varphi')
                value_varphi = self.LL_fixed_prbl.solution.opt_val
            self.buffer_x3y1_fixed.append(list_value_x3y1_fixed)
            self.buffer_varphi.append(value_varphi)
            return value_varphi

    def getLam_generalLL(self):
        x3 = cp.Variable(self.num_x3, boolean=True)
        xx3 = cp.Variable(self.num_x3, boolean=True)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
            yy1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
            yy1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        yy2 = cp.Variable(self.num_y2)
        g = cp.Variable()
        gg = cp.Variable()
        csts = [
            g == self.dl1 @ y1 + self.dl2 @ y2,
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            gg == self.dl1 @ yy1 + self.dl2 @ yy2,
            self.Al3 @ xx3 + self.Bl1 @ yy1 + self.Bl2 @ yy2 <= self.hl,
        ]

        lamU = []
        lamL = []
        for i in range(self.num_x3):
            csts_copy = csts.copy()
            for j in range(self.num_x3):
                if j == i:
                    csts_copy += [
                        x3[i] == 0, xx3[i] == 1,
                    ]
                else:
                    csts_copy += [
                        x3[j] == xx3[j],
                    ]
            prbl = cp.Problem(cp.Maximize(g-gg), csts_copy)
            prbl.solve(solver=cp.GUROBI, verbose=False)  # , threads=4
            if prbl.status != 'optimal':
                exit('Error: Failed in solving lamU')
            lamU.append(g.value-gg.value)
            print('lamU -', i+1, ':', round(g.value-gg.value, 2), '(', prbl.status, ')')
            prbl = cp.Problem(cp.Minimize(g-gg), csts_copy)
            prbl.solve(solver=cp.GUROBI, verbose=False)  # , threads=4
            if prbl.status != 'optimal':
                exit('Error: Failed in solving lamL')
            lamL.append(g.value-gg.value)
            print('lamL -', i+1, ':', round(g.value-gg.value, 2), '(', prbl.status, ')')
        return np.array(lamU), np.array(lamL)

    def getRho_generalLL(self):
        x3 = cp.Variable(self.num_x3, boolean=True)
        xx3 = cp.Variable(self.num_x3, boolean=True)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
            yy1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
            yy1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        yy2 = cp.Variable(self.num_y2)
        g = cp.Variable()
        gg = cp.Variable()
        zzz = cp.Variable(self.num_x3)
        csts = [
            g == self.dl1 @ y1 + self.dl2 @ y2,
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            gg == self.dl1 @ yy1 + self.dl2 @ yy2,
            self.Al3 @ xx3 + self.Bl1 @ yy1 + self.Bl2 @ yy2 <= self.hl,
            np.ones(self.num_x3) @ x3 + np.ones(self.num_x3) @ xx3 - 2 * np.ones(self.num_x3) @ zzz == 1,
            0 <= zzz, zzz >= x3 + xx3 - 1,
            x3 >= zzz, zzz <= xx3,
            ]
        prbl = cp.Problem(cp.Maximize(g-gg), csts)
        prbl.solve(solver=cp.GUROBI, verbose=False)  # , threads=4
        if prbl.status != 'optimal':
            exit('Error: Failed in solving lamU')
        print('Rho :', round(g.value-gg.value, 2), '(', prbl.status, ')')
        return g.value-gg.value

    def getLam_quasiLL_exact(self, value_y1_fixed):
        list_value_y1_fixed = list(np.int_(np.round(value_y1_fixed)))
        if list_value_y1_fixed in self.buffer_y1_fixed_lam:
            idx = self.buffer_y1_fixed_lam.index(list_value_y1_fixed)
            return np.array(self.buffer_lamU[idx]), np.array(self.buffer_lamL[idx])
        else:
            lamU = []
            lamL = []
            if self.flag_LowerLevel == 1:
                for i in range(self.num_x3):
                    ei = np.zeros(self.num_x3)
                    ei[i] = 1
                    lamU.append(self.getVarPhi(np.zeros(self.num_x3), value_y1_fixed) - self.getVarPhi(ei, value_y1_fixed))
                    lamL.append(self.getVarPhi(np.ones(self.num_x3) - ei, value_y1_fixed) - self.getVarPhi(np.ones(self.num_x3), value_y1_fixed))
            if self.flag_LowerLevel == 2:
                for i in range(self.num_x3):
                    ei = np.zeros(self.num_x3)
                    ei[i] = 1
                    lamU.append(self.getVarPhi(np.ones(self.num_x3) - ei, value_y1_fixed) - self.getVarPhi(np.ones(self.num_x3), value_y1_fixed))
                    lamL.append(self.getVarPhi(np.zeros(self.num_x3), value_y1_fixed) - self.getVarPhi(ei, value_y1_fixed))
            self.buffer_y1_fixed_lam.append(list_value_y1_fixed)
            self.buffer_lamU.append(lamU)
            self.buffer_lamL.append(lamL)
        return np.array(lamU), np.array(lamL)

    def getLam_quasiLL_relaxed(self):
        lamU = []
        lamL = []
        if self.flag_LowerLevel == 1:
            for i in range(self.num_x3):
                ei = np.zeros(self.num_x3)
                ei[i] = 1
                lamU.append(self.getLam_quasiLL_relaxed_solve(np.zeros(self.num_x3), ei))
                lamL.append(-self.getLam_quasiLL_relaxed_solve(np.ones(np.ones(self.num_x3), self.num_x3) - ei))
        if self.flag_LowerLevel == 2:
            for i in range(self.num_x3):
                ei = np.zeros(self.num_x3)
                ei[i] = 1
                lamU.append(self.getLam_quasiLL_relaxed_solve(np.ones(self.num_x3) - ei, np.ones(self.num_x3)))
                lamL.append(-self.getLam_quasiLL_relaxed_solve(ei, np.zeros(self.num_x3)))
        return np.array(lamU), np.array(lamL)

    def getLam_quasiLL_relaxed_solve(self, value_z, value_zz):  # sepcial for CFLI !!!!!!!!!!
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                name_y1 = (f"y1[{i}]" for i in range(self.num_y1))
                temp_y1 = [model.getVarByName(i) for i in name_y1]
                value_y1 = np.int_(np.round(np.array(model.cbGetSolution(temp_y1))))
                temp_theta = model.getVarByName('theta')
                pi = self.getPi_quasiLL_relaxed(value_zz, value_y1)
                if self.flag_LowerLevel == 1:
                    theta0 = self.getVarPhi(value_zz, value_y1)
                    idx0 = np.where(value_y1==0)[0]
                    idx1 = np.where(value_y1==1)[0]
                    model.cbLazy(temp_theta >= theta0 + sum(pi[i] * temp_y1[i] for i in idx0) - sum(pi[i] * (1 - temp_y1[i]) for i in idx1))
                if self.flag_LowerLevel == 2:
                    theta0 = self.getVarPhi(value_zz, np.zeros(self.num_y1))
                    model.cbLazy(temp_theta >= theta0 + pi @ temp_y1)
            return

        LB = -1e6
        model = gp.Model()
        y1 = model.addMVar(shape=self.num_y1, vtype=GRB.BINARY, name='y1')
        y2 = model.addMVar(shape=self.num_y2, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='y2')
        theta = model.addVar(vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='theta')
        model.setObjective(self.dl2 @ y2 - theta, GRB.MAXIMIZE)
        model.addConstrs((self.Al3[i,:] @ value_z +
                                     self.Bl1[i,:] @ y1 + self.Bl2[i,:] @ y2 <= self.hl[i] for i in range(len(self.Al3[:,0]))))
        model.addConstr(theta >= LB)
        # x1 = model.addMVar(shape=self.num_x1, vtype=GRB.BINARY, name='x1')
        # x2 = model.addMVar(shape=self.num_x2, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='x2')
        # x3 = model.addMVar(shape=self.num_x3, vtype=GRB.BINARY, name='x3')
        # yy2 = model.addMVar(shape=self.num_y2, vtype=GRB.CONTINUOUS, lb=-np.infty, ub=np.infty, name='y2')
        # model.addConstrs((self.Au1[i,:] @ x1 + self.Au2[i,:] @ x2 + self.Au3[i,:] @ x3 +
        #                              self.Bu1[i,:] @ y1 + self.Bu2[i,:] @ yy2 <= self.hu[i] for i in range(len(self.Au3[:,0]))))
        # model.addConstrs((self.Al3[i,:] @ x3 +
        #                              self.Bl1[i,:] @ y1 + self.Bl2[i,:] @ yy2 <= self.hl[i] for i in range(len(self.Al3[:,0]))))

        # model._Ind_indicator = 0
        model.Params.lazyConstraints = 1
        model.Params.OutputFlag = 0
        model.optimize(callback)
        if (model.Status != 2) & (model.Status != 9):
            print('error')
            return None
        value_y1 = y1.X
        value_y2 = y2.X
        value_theta = theta.X
        value_obj = self.dl2 @ value_y2 - value_theta
        return value_obj

    def getPi_quasiLL_relaxed(self, value_x3, value_y1):
        if self.flag_LowerLevel == 1:
            varphi = self.getVarPhi(value_x3, value_y1)
            varphi1 = self.getVarPhi(value_x3, np.ones(self.num_y1))
            pi = np.zeros(self.num_y1)
            for i in range(self.num_y1):
                ei = np.zeros(self.num_y1)
                ei[i] = 1
                if value_y1[i] == 0:
                    pi[i] = self.getVarPhi(value_x3, value_y1 + ei) - varphi
                if value_y1[i] == 1:
                    pi[i] = varphi1 - self.getVarPhi(value_x3, np.ones(self.num_y1) - ei)
        if self.flag_LowerLevel == 2:
            pi = np.zeros(self.num_y1)
            order = np.argsort(value_y1)
            order = order[::-1]
            phi = np.zeros(self.num_y1 + 1)
            for i in range(self.num_y1 + 1):
                yi = np.zeros(self.num_y1)
                yi[order[0:i]] = 1
                phi[i] = self.getVarPhi(value_x3, yi)
            temp = phi[1:] - phi[0:-1]
            pi[order] = temp
        return pi

    def getLam_specialLL(self):
        lamU = []
        lamL = []
        if self.flag_LowerLevel == 1:
            for i in range(self.num_x3):
                ei = np.zeros(self.num_x3)
                ei[i] = 1
                lamU.append(self.getPhi(np.zeros(self.num_x3)) - self.getPhi(ei))
                lamL.append(self.getPhi(np.ones(self.num_x3) - ei) - self.getPhi(np.ones(self.num_x3)))
        if self.flag_LowerLevel == 2:
            for i in range(self.num_x3):
                ei = np.zeros(self.num_x3)
                ei[i] = 1
                temp1, _ = self.getPhi(np.ones(self.num_x3) - ei)
                temp2, _ = self.getPhi(np.ones(self.num_x3))
                lamU.append(temp1 - temp2)
                temp1, _ = self.getPhi(np.zeros(self.num_x3))
                temp2, _ = self.getPhi(ei)
                lamL.append(temp1 - temp2)
        return np.array(lamU), np.array(lamL)

    def getLam_generalLL_dual(self):
        x3 = cp.Variable(self.num_x3, boolean=True)
        xx3 = cp.Variable(self.num_x3, boolean=True)
        y1 = cp.Variable(self.num_y1, boolean=True)
        yy1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        yy2 = cp.Variable(self.num_y2)
        g = cp.Variable()
        gg = cp.Variable()

        csts0 = [
            np.maximum(self.Al3, 0) @ np.ones(self.num_x3) + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
        ]
        prbl0 = cp.Problem(cp.Maximize(self.dl1 @ y1 + self.dl2 @ y2), csts0)
        prbl0.solve(solver=cp.GUROBI, verbose=False)  # , threads=4
        if prbl0.status != 'optimal':
            exit('Error: Failed in solving y1 guess')
        value_yy1_fixed = y1.value

        bigM = 10000
        ww = cp.Variable(len(self.hl))
        zw = cp.Variable(self.num_x3)
        csts = [
            g == self.dl1 @ y1 + self.dl2 @ y2,
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            gg == self.dl1 @ value_yy1_fixed + (self.hl - self.Bl1 @ value_yy1_fixed) @ ww - np.ones(self.num_x3) @ zw,
            self.Bl2.T @ ww == self.dl2,
            0 <= ww,
            - bigM * xx3 <= zw, zw <= bigM * xx3,
            - bigM * (1 - xx3) + self.Al3.T @ ww <= zw, zw <= self.Al3.T @ ww + bigM * (1 - xx3),
        ]

        lamU = []
        lamL = []
        for i in range(self.num_x3):
            csts_copy = csts.copy()
            for j in range(self.num_x3):
                if j == i:
                    csts_copy += [
                        x3[i] == 0, xx3[i] == 1,
                    ]
                else:
                    csts_copy += [
                        x3[j] == xx3[j],
                    ]
            prblU = cp.Problem(cp.Maximize(g-gg), csts_copy)
            prblU.solve(solver=cp.GUROBI, verbose=False, TimeLimit=self.Lagrangian_timelimit_lam)  # , threads=4
            if prblU.status == 'optimal':
                lam = g.value-gg.value
            else:
                if prblU.status == 'user_limit':
                    lam = -prblU.solution.attr['solver_specific_stats'].ObjBound
                else:
                    exit('Error: Failed in solving lamU')
            if abs(lam) > bigM:
                exit('Attention: Big M')
            lamU.append(lam)
            print('lamU -', i+1, ':', round(lam, 2), '(', prblU.status, ')')

            csts_copy = csts.copy()
            for j in range(self.num_x3):
                if j == i:
                    csts_copy += [
                        x3[i] == 1, xx3[i] == 0,
                    ]
                else:
                    csts_copy += [
                        x3[j] == xx3[j],
                    ]
            prblL = cp.Problem(cp.Maximize(g-gg), csts_copy)
            prblL.solve(solver=cp.GUROBI, verbose=False, TimeLimit=self.Lagrangian_timelimit_lam)  # , threads=4
            if prblL.status == 'optimal':
                lam = -(g.value-gg.value)
            else:
                if prblL.status == 'user_limit':
                    lam = -(-prblL.solution.attr['solver_specific_stats'].ObjBound)
                else:
                    exit('Error: Failed in solving lamL')
            if abs(lam) > bigM:
                exit('Attention: Big M')
            lamL.append(lam)
            print('lamL -', i+1, ':', round(lam, 2), '(', prblL.status, ')')
        return np.array(lamU), np.array(lamL)

    def getRho_general(self):
        x3 = cp.Variable(self.num_x3, boolean=True)
        xx3 = cp.Variable(self.num_x3, boolean=True)
        y1 = cp.Variable(self.num_y1, boolean=True)
        yy1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        yy2 = cp.Variable(self.num_y2)
        g = cp.Variable()
        gg = cp.Variable()

        csts0 = [
            np.maximum(self.Al3, 0) @ np.ones(self.num_x3) + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
        ]
        prbl0 = cp.Problem(cp.Maximize(self.dl1 @ y1 + self.dl2 @ y2), csts0)
        prbl0.solve(solver=cp.GUROBI, verbose=False)  # , threads=4
        if prbl0.status != 'optimal':
            exit('Error: Failed in solving y1 guess')
        value_yy1_fixed = y1.value

        bigM = 100
        ww = cp.Variable(len(self.hl))
        zw = cp.Variable(self.num_x3)
        zzz = cp.Variable(self.num_x3)
        csts = [
            np.ones(self.num_x3) @ x3 + np.ones(self.num_x3) @ xx3 - 2 * np.ones(self.num_x3) @ zzz == 1,
            0 <= zzz, zzz >= x3 + xx3 - 1,
            x3 >= zzz, zzz <= xx3,
            g == self.dl1 @ y1 + self.dl2 @ y2,
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            gg == self.dl1 @ value_yy1_fixed + (self.hl - self.Bl1 @ value_yy1_fixed) @ ww - np.ones(self.num_x3) @ zw,
            self.Bl2.T @ ww == self.dl2,
            0 <= ww,
            - bigM * xx3 <= zw, zw <= bigM * xx3,
            - bigM * (1 - xx3) + self.Al3.T @ ww <= zw, zw <= self.Al3.T @ ww + bigM * (1 - xx3),
        ]
        prbl = cp.Problem(cp.Maximize(g-gg), csts)
        prbl.solve(solver=cp.GUROBI, verbose=False, TimeLimit=self.Lagrangian_timelimit_rho)  # , threads=4
        if prbl.status == 'optimal':
            rho = g.value-gg.value
        else:
            if prbl.status == 'user_limit':
                rho = -prbl.solution.attr['solver_specific_stats'].ObjBound
            else:
                exit('Error: Failed in solving lamU')
        print('rho :', round(rho, 2), '(', prbl.status, ')')
        return rho

    def getPi(self, value_x3, value_y1_fixed):
        if self.flag_LowerLevel == 1:
            varphi = self.getVarPhi(value_x3, value_y1_fixed)
            varphi1 = self.getVarPhi(np.ones(self.num_x3), value_y1_fixed)
            pi = np.zeros(self.num_x3)
            for i in range(self.num_x3):
                ei = np.zeros(self.num_x3)
                ei[i] = 1
                if value_x3[i] == 0:
                    pi[i] = self.getVarPhi(value_x3 + ei, value_y1_fixed) - varphi
                if value_x3[i] == 1:
                    pi[i] = varphi1 - self.getVarPhi(np.ones(self.num_x3) - ei, value_y1_fixed)
        if self.flag_LowerLevel == 2:
            pi = np.zeros(self.num_x3)
            order = np.argsort(value_x3)
            order = order[::-1]
            phi = np.zeros(self.num_x3 + 1)
            for i in range(self.num_x3 + 1):
                xi = np.zeros(self.num_x3)
                xi[order[0:i]] = 1
                phi[i] = self.getVarPhi(xi, value_y1_fixed)
            temp = phi[1:] - phi[0:-1]
            pi[order] = temp
        return pi

    def getPi_quasi(self, value_x3, value_y1):
        value_x3y1 = np.hstack([value_x3, value_y1])
        if self.flag_LowerLevel == 2:
            pi = np.zeros(self.num_x3+self.num_y1)
            order = np.argsort(value_x3y1)
            order = order[::-1]
            phi = np.zeros(self.num_x3 + self.num_y1 + 1)
            for i in range(self.num_x3 + self.num_y1 + 1):
                xi = np.zeros(self.num_x3 + self.num_y1)
                xi[order[0:i]] = 1
                xi_x3 = xi[0:self.num_x3]
                xi_y1 = xi[self.num_x3:]
                phi[i] = self.getVarPhi(xi_x3, xi_y1)
            temp = phi[1:] - phi[0:-1]
            pi[order] = temp
        return pi

    def getUB_heuristic(self, NN):
        self.heuristic_UB_num_all += 1
        if self.flag_heuristic_UB_solving:
            # sampling
            T1 = time.time()
            if len(self.buffer_x3) < self.num_x3 * 2:
                _, _, _, _, _, value_obj_upper_HPR, _ = self.solvebyHPR()
                value_UB = min(value_obj_upper_HPR, self.good_UB)
                samples = self.samplesGen(self.heuristic_UB_num_samples_target, self.heuristic_UB_num_sampling, value_UB, self.heuristic_UB_max_repeated)
                idx = np.argmin(self.buffer_obj_upper)
                value_x1 = np.array(self.buffer_x1[idx])
                value_x2 = np.array(self.buffer_x2[idx])
                value_x3 = np.array(self.buffer_x3[idx])
                value_y1 = np.array(self.buffer_y1[idx])
                value_y2 = np.array(self.buffer_y2[idx])
                value_obj_upper = self.buffer_obj_upper[idx]
                if value_obj_upper < self.good_UB:
                    self.SaveGoodUB(value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper)
                    self.heuristic_UB_num_win += 1
                else:
                    if self.flag_no_good:
                        if (value_obj_upper > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                            self.no_good_value_x3.append(value_x3)
                            self.no_good_value_x3_history.append(list(value_x3))
                if self.flag_no_good:
                    for i in range(len(self.buffer_obj_upper)):
                        if i == idx:
                            continue
                        value_x3 = np.array(self.buffer_x3[i])
                        if (self.buffer_obj_upper[i] > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                            self.no_good_value_x3.append(value_x3)
                            self.no_good_value_x3_history.append(list(value_x3))
            else:
                samples = np.hstack([np.array(self.buffer_x3),np.array(self.buffer_phi).reshape(-1,1),np.array(self.buffer_obj_upper).reshape(-1,1)])
                # value_UB = self.good_UB
                # samples = self.samplesGen(1e8, self.heuristic_UB_num_sampling_only, value_UB, self.heuristic_UB_max_repeated)
                # idx = np.argmin(self.buffer_obj_upper)
                # value_x1 = np.array(self.buffer_x1[idx])
                # value_x2 = np.array(self.buffer_x2[idx])
                # value_x3 = np.array(self.buffer_x3[idx])
                # value_y1 = np.array(self.buffer_y1[idx])
                # value_y2 = np.array(self.buffer_y2[idx])
                # value_obj_upper = self.buffer_obj_upper[idx]
                # if value_obj_upper < self.good_UB:
                #     self.SaveGoodUB(value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper)
                #     self.heuristic_UB_num_win += 1
                # else:
                #     if self.flag_no_good:
                #         if (value_obj_upper > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                #             self.no_good_value_x3.append(value_x3)
                #             self.no_good_value_x3_history.append(list(value_x3))
                # if self.flag_no_good:
                #     for i in range(len(self.buffer_obj_upper)):
                #         if i == idx:
                #             continue
                #         value_x3 = np.array(self.buffer_x3[i])
                #         if (self.buffer_obj_upper[i] > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                #             self.no_good_value_x3.append(value_x3)
                #             self.no_good_value_x3_history.append(list(value_x3))
            T2 = time.time()
            self.time_CP_heuristic_sampling += T2 - T1

            # training
            num_samples = len(samples[:,0])
            T1 = time.time()
            if num_samples >= self.num_x3 * 2:
                if NN.history == None:
                    NN.train(samples[:, 0:-1], self.heuristic_UB_num_epoch_initial)
                else:
                    NN.train_continued(samples[:, 0:-1], self.heuristic_UB_num_epoch_continued)
            T2 = time.time()
            self.time_CP_heuristic_training += T2 - T1

            # solving
            T1 = time.time()
            value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper, value_obj_lower = self.solutionByMIP(NN)
            T2 = time.time()
            self.time_CP_heuristic_solving += T2 - T1
            if value_obj_upper >= np.min(samples[:,-1]):
                if self.flag_no_good:
                    if (value_obj_upper > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                        self.no_good_value_x3.append(value_x3)
                        self.no_good_value_x3_history.append(list(value_x3))
                idx = np.argmin(samples[:, -1])
                value_x3 = samples[idx, 0:-2]
                value_phi = samples[idx, -2]
                T1 = time.time()
                value_x1, value_x2, value_y1, value_y2, _ = self.getFeasible(value_x3, value_phi)
                T2 = time.time()
                self.time_CP_heuristic_checking += T2 - T1
                value_obj_upper = self.cu1 @ value_x1 + self.cu2 @ value_x2 + self.cu3 @ value_x3 + self.du1 @ value_y1 + self.du2 @ value_y2
                value_obj_lower = self.dl1 @ value_y1 + self.dl2 @ value_y2
                print('Heuristic UB-NN calculated:', np.round(value_obj_upper,4), '--- Sampling')
            else:
                if value_obj_upper < self.good_UB:
                    self.SaveGoodUB(value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper)
                    self.heuristic_UB_num_win += 1
                else:
                    if self.flag_no_good:
                        if (value_obj_upper > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                                self.no_good_value_x3.append(value_x3)
                                self.no_good_value_x3_history.append(list(value_x3))
                print('Heuristic UB-NN calculated:', np.round(value_obj_upper,4), '--- Solving')
        else:
            # sampling
            T1 = time.time()
            if self.good_UB == np.infty:
                _, _, _, _, _, value_obj_upper_HPR, _ = self.solvebyHPR()
                value_UB = value_obj_upper_HPR
            else:
                value_UB = self.good_UB
            samples = self.samplesGen(1e8, self.heuristic_UB_num_sampling_only, value_UB, self.heuristic_UB_max_repeated)
            idx = np.argmin(samples[:,-1])
            value_x3 = samples[idx,0:-2]
            value_phi = samples[idx,-2]
            value_obj_upper = samples[idx,-1]
            value_x1, value_x2, value_y1, value_y2, _ = self.getFeasible(value_x3, value_phi)
            value_obj_lower = value_phi
            if value_obj_upper < self.good_UB:
                self.SaveGoodUB(value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper)
                self.heuristic_UB_num_win += 1
            else:
                if self.flag_no_good:
                    if (value_obj_upper > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                        self.no_good_value_x3.append(value_x3)
                        self.no_good_value_x3_history.append(list(value_x3))
            if self.flag_no_good:
                for i in range(len(samples[:,0])):
                    if i == idx:
                        continue
                    value_x3 = samples[i, 0:-2]
                    if (samples[i, -1] > self.good_UB) & (list(value_x3) not in self.no_good_value_x3_history) & (list(value_x3) not in [list(self.good_value_x3)]):
                        self.no_good_value_x3.append(value_x3)
                        self.no_good_value_x3_history.append(list(value_x3))
            T2 = time.time()
            self.time_CP_heuristic_sampling += T2 - T1
            print('Heuristic UB-NN sampled:', np.round(value_obj_upper,4))
        return np.int_(np.round(value_x1)), value_x2, np.int_(np.round(value_x3)), np.int_(np.round(value_y1)), value_y2, value_obj_upper, value_obj_lower

    def SaveGoodUB(self, value_x1, value_x2, value_x3, value_y1, value_y2, value_obj_upper):
        if self.flag_no_good:
            if self.good_UB != np.infty:
                self.no_good_value_x3.append(self.good_value_x3)
                self.no_good_value_x3_history.append(list(self.good_value_x3))
        self.good_value_x1 = value_x1
        self.good_value_x2 = value_x2
        self.good_value_x3 = value_x3
        self.good_value_y1 = value_y1
        self.good_value_y2 = value_y2
        self.good_UB = value_obj_upper
        self.flag_good_UB = True
        return

    def getLB_heuristic(self):
        # sampling
        T1 = time.time()
        # _, _, _, _, _, value_obj_upper_HPR, _ = self.solvebyHPR()
        # value_UB = value_obj_upper_HPR
        # self.samplesGen2(self.heuristic_LB_num_samples_target, self.heuristic_LB_num_sampling, value_UB, self.heuristic_LB_max_repeated)
        # samples = np.hstack([np.array(self.buffer_x3),np.array(self.buffer_y1)])
        samples = self.samplesRead('samples'+str(self.num_x3)+'.xlsx')
        # samplesadd = self.samplesRead('samples15add.xlsx')
        # list_samplesadd = list(range(len(samplesadd[:,0])))
        # idx = random.sample(list_samplesadd, 1000)
        # samples = np.vstack([samples, samplesadd[idx,:]])
        # list_samplesadd = list(set(list_samplesadd) - set(idx))
        # idx = random.sample(list_samplesadd, 1000)
        # samples = np.vstack([samples, samplesadd[idx,:]])
        # list_samplesadd = list(set(list_samplesadd) - set(idx))
        # idx = random.sample(list_samplesadd, 1000)
        # samples = np.vstack([samples, samplesadd[idx,:]])
        # list_samplesadd = list(set(list_samplesadd) - set(idx))
        # samples = np.vstack([samples, samplesadd[list_samplesadd,:]])
        T2 = time.time()
        self.time_CP_heuristic_sampling += T2 - T1
        num_samples = len(samples[:,0])

        # training
        num_layer = 2
        num_input = self.num_x3 if not self.flag_doubleX else self.num_x3 * 2
        num_output = self.num_y1
        num_hidden = calculateNumHidden(num_samples*num_output, num_layer, num_input, self.flag_doubleX)
        # self.heuristic_LB_NNArchitecture = [num_input, num_hidden, num_hidden, num_output]
        self.heuristic_LB_NNArchitecture = [num_input, round(num_output/5), round(num_output/5), num_output]
        NN = NNmodel(self.heuristic_LB_NNArchitecture, self.flag_ISNN, self.flag_doubleX)
        T1 = time.time()
        NN.train(samples, self.heuristic_LB_num_epoch)
        T2 = time.time()
        self.time_CP_heuristic_training += T2 - T1
        return NN

    def samplesGen2(self, num_samples_target, num_sampling, value_UB, max_repeated):
        # sampling
        # num_samples_target = 5000
        num_sampling *= 10
        count_invalid = 0
        count_repeated = 0
        count_infeasible = 0
        num_infeasible = 0
        for i in range(num_sampling):
            if (len(self.buffer_x3) >= num_samples_target) | (len(self.buffer_x3) >= 2**self.num_x3) | (count_repeated > max_repeated) | (count_infeasible > max_repeated/2):
                if count_repeated > max_repeated:
                    print('Sampling terminated: Repeated')
                if count_infeasible > max_repeated/2:
                    print('Sampling terminated: Infeasible')
                print('Sampling:', i, '/', num_sampling, '| sample:', len(self.buffer_x3), '/', 2 ** self.num_x3, '| Infeasible:', num_infeasible, '/', 2 ** self.num_x3, '| Best UB:', np.round(value_UB, 4))
                break
            temp = np.unique(np.random.randint(0, self.num_x3, round(self.num_x3/3)))
            value_x3 = np.ones(self.num_x3)
            value_x3[list(temp)] = 0
            if list(value_x3) in self.buffer_x3:
                count_repeated += 1
                print('Sampling', i+1, '/', num_sampling, ': Repeated sample (sample', len(self.buffer_x3), '/', 2**self.num_x3, ', UB =', np.round(value_UB, 4), ', count_repeat =', count_repeated, ')')
                continue
            else:
                count_repeated = 0
            value_phi, _ = self.getPhi(value_x3)
            if value_phi == None:
                count_infeasible += 1
                num_infeasible += 1
                print('Sampling', i+1, '/', num_sampling, ': Infeasible sample (sample', len(self.buffer_x3), '/', 2**self.num_x3, ', UB =', np.round(value_UB, 4), ', count_infeasible =', count_infeasible, ')')
                continue
            else:
                count_infeasible = 0
            print('Sampling', i+1, '/', num_sampling, ': Optimal (sample', len(self.buffer_x3), '/', 2**self.num_x3, ', UB =', np.round(value_UB, 4), ')')
        # export samples
        self.samplesSave(self.buffer_x3, self.buffer_phi, self.buffer_obj_upper)
        return np.hstack([np.array(self.buffer_x3),np.array(self.buffer_phi).reshape(-1,1),np.array(self.buffer_obj_upper).reshape(-1,1)])

    def samplesGen3(self, num_samples_target, num_sampling, value_UB, max_repeated):
        samples = self.samplesRead('samples15.xlsx')
        temp = samples[:, 0:self.num_x3]
        samples_x3 = [list(temp[i,:]) for i in range(len(temp[:,0]))]
        # sampling
        num_sampling = 2**self.num_x3
        count_infeasible = 0
        num_infeasible = 0
        tsfm = '{:0' + str(self.num_x3) + 'b}'
        for i in range(num_sampling):
            temp = list(tsfm.format(i))
            value_x3 = np.array([int(temp[j]) for j in range(self.num_x3)])
            if np.sum(1-value_x3) > round(self.num_x3/3):
                continue
            if list(value_x3) in self.buffer_x3:
                continue
            if list(value_x3) in samples_x3:
                continue
            value_phi, _ = self.getPhi(value_x3)
            if value_phi == None:
                print('Sampling', i+1, '/', num_sampling, ': Infeasible sample (sample', len(self.buffer_x3), '/', 2**self.num_x3, ', count_infeasible =', count_infeasible, ')')
            else:
                print('Sampling', i+1, '/', num_sampling, ': Optimal (sample', len(self.buffer_x3), '/', 2**self.num_x3, ')')
        print('Sampling:', i+1, '/', num_sampling, '| sample:', len(self.buffer_x3), '/', 2 ** self.num_x3, '| Infeasible:', num_infeasible, '/', 2 ** self.num_x3)
        # export samples
        self.samplesSave(self.buffer_x3, self.buffer_phi, self.buffer_obj_upper)
        return np.hstack([np.array(self.buffer_x3),np.array(self.buffer_phi).reshape(-1,1),np.array(self.buffer_obj_upper).reshape(-1,1)])

    def LowerLevel_fixed_heuristic_LB(self):
        x3 = cp.Parameter(self.num_x3)
        y1 = cp.Parameter(self.num_y1)
        y2 = cp.Variable(self.num_y2)
        csts = [
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
        ]
        prbl = cp.Problem(cp.Maximize(self.dl2 @ y2), csts)
        return x3, y1, prbl

    def getVarPhi_heuristic_LB(self, value_x3, value_yy1):
        value_x3y1_fixed = np.hstack([value_x3, value_yy1])
        list_value_x3y1_fixed = list(np.int_(np.round(value_x3y1_fixed)))
        if list_value_x3y1_fixed in self.buffer_x3y1_fixed:
            idx = self.buffer_x3y1_fixed.index(list_value_x3y1_fixed)
            return self.buffer_varphi[idx]
        else:
            self.LL_heuLB_x3.value = value_x3
            self.LL_heuLB_yy1.value = value_yy1
            self.LL_heuLB_prbl.solve(solver=cp.GUROBI, verbose=False)
            if self.LL_heuLB_prbl.status != 'optimal':
                exit('Error: Failed in getting varphi')
            value_varphi = self.LL_heuLB_prbl.solution.opt_val
        return value_varphi

    def getLam_heuristic_quasiLL(self):
        lamU = []
        lamL = []
        for i in range(self.num_x3+self.num_y1):
            ei = np.zeros(self.num_x3+self.num_y1)
            ei[i] = 1
            ttt = np.ones(self.num_x3+self.num_y1) - ei
            temp1 = self.getVarPhi_heuristic_LB(ttt[0:self.num_x3], ttt[self.num_x3:])
            temp2 = self.getVarPhi_heuristic_LB(np.ones(self.num_x3), np.ones(self.num_y1))
            lamU.append(temp1 - temp2)
            temp1 = self.getVarPhi_heuristic_LB(np.zeros(self.num_x3), np.zeros(self.num_y1))
            temp2 = self.getVarPhi_heuristic_LB(ei[0:self.num_x3], ei[self.num_x3:])
            lamL.append(temp1 - temp2)
        return np.array(lamU), np.array(lamL)

    def getLam_heuristic_LB(self):
        bigM = 1000
        lamU = []
        lamL = []
        A = np.hstack([self.Al3, self.Bl1])
        B = self.Bl2
        h = self.hl
        d = self.dl2
        z = cp.Variable(self.num_x3 + self.num_y1, boolean=True)
        zz = cp.Variable(self.num_x3 + self.num_y1, boolean=True)
        zw = cp.Variable(self.num_x3 + self.num_y1)
        y = cp.Variable(self.num_y2)
        ww = cp.Variable(len(B[:,0]))
        csts = [
            A @ z + B @ y <= h,
            B.T @ ww == d,
            ww >= 0,
            - bigM * zz <= zw, zw <= bigM * zz,
            - bigM * (1 - zz) + A.T @ ww <= zw, zw <= A.T @ ww + bigM * (1 - zz),
        ]
        obj = d.T @ y - h.T @ ww + np.ones(self.num_x3+self.num_y1) @ zw
        for i in range(self.num_x3 + self.num_y1):
            csts_add = []
            for j in range(self.num_x3 + self.num_y1):
                if j == i:
                    csts_add += [
                        z[j] == 0, zz[j] == 1,
                    ]
                else:
                    csts_add += [
                        z[j] == zz[j],
                    ]
            prbl = cp.Problem(cp.Maximize(obj), csts + csts_add)
            prbl.solve(solver=cp.GUROBI, TimeLimit=self.Lagrangian_timelimit_lam_heuristic, verbose=True)  # , threads=4
            if prbl.status == 'optimal':
                lam = obj.value
            else:
                if prbl.status == 'user_limit':
                    lam = -prbl.solution.attr['solver_specific_stats'].ObjBound
                else:
                    exit('Error: Failed in solving lamU')
            if abs(lam) > bigM:
                exit('Attention: Big M')
            lamU.append(lam)
            print('lamU -', i+1, ':', round(lam, 2), '(', prbl.status, ')')

            csts_add = []
            for j in range(self.num_x3 + self.num_y1):
                if j == i:
                    csts_add += [
                        z[j] == 1, zz[j] == 0,
                    ]
                else:
                    csts_add += [
                        z[j] == zz[j],
                    ]
            prbl = cp.Problem(cp.Maximize(obj), csts + csts_add)
            prbl.solve(solver=cp.GUROBI, TimeLimit=self.Lagrangian_timelimit_lam_heuristic, verbose=True)  # , threads=4
            if prbl.status == 'optimal':
                lam = -obj.value
            else:
                if prbl.status == 'user_limit':
                    lam = -(-prbl.solution.attr['solver_specific_stats'].ObjBound)
                else:
                    exit('Error: Failed in solving lamL')
            if abs(lam) > bigM:
                exit('Attention: Big M')
            lamL.append(lam)
            print('lamL -', i+1, ':', round(lam, 2), '(', prbl.status, ')')
        return np.array(lamU), np.array(lamL)

    def getRho_heuristic_LB(self):
        flag_rho = 'exact'
        A = np.hstack([self.Al3, self.Bl1])
        B = self.Bl2
        h = self.hl
        d = self.dl2

        z = cp.Variable(self.num_x3 + self.num_y1, boolean=True)
        zz = cp.Variable(self.num_x3 + self.num_y1, boolean=True)
        zzz = cp.Variable(self.num_x3 + self.num_y1)
        y = cp.Variable(self.num_y2)
        yy = cp.Variable(self.num_y2)
        if flag_rho == 'relaxed':
            csts = [
                np.ones(self.num_x3+self.num_y1) @ z + np.ones(self.num_x3+self.num_y1) @ zz - 2 * np.ones(self.num_x3+self.num_y1) @ zzz == 1,
                A @ z + B @ y <= h,
                A @ zz + B @ yy <= h,
                0 <= zzz, zzz >= z + zz - 1,
                z >= zzz, zzz <= zz,
            ]
            obj = d.T @ y - d.T @ yy
        if flag_rho == 'exact':
            bigM = 100
            zw = cp.Variable(self.num_x3 + self.num_y1)
            ww = cp.Variable(len(B[:,0]))
            csts = [
                np.ones(self.num_x3+self.num_y1) @ z + np.ones(self.num_x3+self.num_y1) @ zz - 2 * np.ones(self.num_x3+self.num_y1) @ zzz == 1,
                A @ z + B @ y <= h,
                B.T @ ww == d,
                ww >= 0,
                - bigM * zz <= zw, zw <= bigM * zz,
                - bigM * (1 - zz) + A.T @ ww <= zw, zw <= A.T @ ww + bigM * (1 - zz),
                0 <= zzz, zzz >= z + zz - 1,
                z >= zzz, zzz <= zz,
            ]
            obj = d.T @ y - h.T @ ww + np.ones(self.num_x3+self.num_y1) @ zw
        prbl = cp.Problem(cp.Maximize(obj), csts)
        prbl.solve(solver=cp.GUROBI, verbose=True, TimeLimit=self.Lagrangian_timelimit_rho)  # , threads=4
        if prbl.status == 'optimal':
            rho = obj.value
        else:
            if prbl.status == 'user_limit':
                rho = -prbl.solution.attr['solver_specific_stats'].ObjBound
            else:
                exit('Error: Failed in solving Rho')
        return rho

    def getApproximation_heuristic_LB(self, samples):
        idx = np.argsort(samples[:,1])
        samples = samples[idx, :]
        idx0 = np.where(samples[:,1]==0)[0]
        y0 = samples[idx0[-1]+1,1]
        idx1 = np.where(samples[:,1]==y0)[0]
        samples0 = samples[idx0, :]
        samples1 = samples[idx1, :]
        samples_linear = samples[idx1[-1]+1:,:]
        x0 = (np.max(samples0[:,0]) + np.min(samples1[:,0]))/2
        k1, b1 = np.polyfit(samples_linear[:,0], samples_linear[:,1], deg=1)
        x1 = (y0 - b1) / k1

        def f_app(x):
            f1 = (x>=x0) * y0
            f2 = k1 * np.maximum(x - x1, 0)
            return f1+f2

        sample = samples[:,0]
        label = samples[:,1]
        phi_pre = f_app(sample)
        err = phi_pre - label
        err_max = np.max(abs(err))
        err_relative = np.array([err[i] / label[i] * 100 if label[i] != 0 else 0 for i in range(len(label))])
        print('max_abs_error:', np.round(err_max, 2))
        print('max_relative_error:', np.round(np.max(err_relative), 2), '%')
        # absolute_error
        plt.figure()
        plt.boxplot(err)
        plt.ylabel('absolute_error')
        plt.show()
        # # relative_error
        # plt.figure()
        # plt.boxplot(err_relative)
        # plt.ylabel('relative_error (%)')
        # plt.show()
        #
        plt.scatter(sample, label, s=2, label='label')
        plt.scatter(sample, phi_pre, s=2, label='prediction')
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid()
        plt.show()

        return x0, y0, k1, x1

    def LP_objectiveCut(self):
        x3 = cp.Parameter(self.num_x3)
        z3 = cp.Variable(self.num_x3)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1)
        y2 = cp.Variable(self.num_y2)
        csts = [
            z3 == x3,
            self.Al3 @ z3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            0 <= y1, y1 <= 1,
            # 0 <= z3, z3 <= 1,
        ]
        prbl = cp.Problem(cp.Minimize(self.cu3 @ z3 + self.du1 @ y1 + self.du2 @ y2), csts)
        return x3, prbl

    def LP_objectiveCut2(self):
        x3 = cp.Parameter(self.num_x3)
        pi1 = cp.Variable(self.num_x3)
        pi2 = cp.Variable(len(self.Al3[:,0]))
        pi3 = cp.Variable(self.num_y1)
        csts = [
            pi1 + self.Al3.T @ pi2 == self.cu3,
            self.Bl1.T @ pi2 + pi3 <= self.du1,
            self.Bl2.T @ pi2 == self.du2,
            pi2 <= 0, pi3 <= 0
        ]
        prbl = cp.Problem(cp.Maximize(x3 @ pi1 + self.hl @ pi2 + np.ones(self.num_y1) @ pi3), csts)
        return x3, prbl

    def MILP_objectiveCut(self):
        lam = cp.Parameter(self.num_x3)
        z3 = cp.Variable(self.num_x3)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        csts = [
            self.Al3 @ z3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            0 <= z3, z3 <= 1,
        ]
        prbl = cp.Problem(cp.Minimize(self.cu3 @ z3 + self.dl1 @ y1 + self.dl2 @ y2 - lam @ z3), csts)
        return lam, prbl

    def MILP_JiangsCut(self):
        x3 = cp.Parameter(self.num_x3)

        alpha = cp.Variable(self.num_x3)
        beta = cp.Variable()
        lam = cp.Variable(self.num_x3)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        yy2 = cp.Variable((self.num_y2, self.num_x3))
        csts = [
            beta <= self.dl1 @ y1 + self.dl2 @ y2 - np.ones(self.num_x3) @ lam,
            self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl,
            lam >= 0,
            lam >= alpha + self.dl2 @ yy2,
            self.Bl2 @ yy2 >= self.Al3
        ]
        # for i in range(self.num_x3):
        #     csts += [
        #         lam[i] >= alpha[i] + self.dl2 @ yy2[:,i],
        #         self.Bl2 @ yy2[:,i] >= self.Al3[:,i],
        #     ]
        prbl = cp.Problem(cp.Maximize(0), csts)
        return x3, prbl

    def MILP_JiangsCut_check(self):
        pi = cp.Variable(len(self.hl))
        csts = [
            self.Bl2.T @ pi == self.dl2,
            pi >= 0,
        ]
        prbl = cp.Problem(cp.Minimize(self.hl @ pi), csts)
        prbl.solve(solver=cp.GUROBI, verbose=True)
        if prbl.status != 'optimal':
            exit('Error')

        y2 = cp.Variable(self.num_y2)
        csts = [
            self.Bl2 @ y2 <= self.hl,
        ]
        prbl = cp.Problem(cp.Maximize(self.dl2 @ y2), csts)
        prbl.solve(solver=cp.GUROBI, verbose=True)
        return

    def MILP_JiangsCut2(self):
        x3 = cp.Parameter(self.num_x3)

        alpha = cp.Variable(self.num_x3)
        beta = cp.Variable()
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        U = cp.Variable((self.num_y2,self.num_x3))
        vn = cp.Variable(self.num_x3)
        vp = cp.Variable((len(self.Al3[:,0]),self.num_x3))
        csts = [
            beta <= self.dl1 @ y1 + self.dl2 @ y2 + np.ones(self.num_x3) @ vn,
            self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl - vp @ np.ones(self.num_x3),
            vn <= 0,
            vn <= U.T @ self.dl2 - alpha,
            vp >= 0,
            vp >= self.Al3 + self.Bl2 @ U
        ]
        prbl = cp.Problem(cp.Maximize(x3 @ alpha + beta), csts)
        return x3, prbl

    def MILP_JiangsCut2_fixed(self):
        x3 = cp.Parameter(self.num_x3)
        y1 = cp.Parameter(self.num_y1)
        m = int(self.num_y2/self.num_x3)

        alpha = cp.Variable(self.num_x3)
        beta = cp.Variable()
        # U = cp.Variable((self.num_y2,self.num_x3))
        U = cp.Variable(self.num_y2)
        y2 = cp.Variable(self.num_y2)
        vn = cp.Variable(self.num_x3)
        vp = cp.Variable((len(self.Al3[:,0]),self.num_x3))
        csts = [
            beta <= self.dl1 @ y1 + self.dl2 @ y2 + np.ones(self.num_x3) @ vn,
            self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl - vp @ np.ones(self.num_x3),
            vn <= 0,
            # vn <= U.T @ self.dl2 - alpha,
            vn <= sum(cp.diag(U[list(self.num_x3 * i +np.array(range(self.num_x3)))]) @ self.dl2[list(self.num_x3 * i +np.array(range(self.num_x3)))] for i in range(m)) - alpha,
            vp >= 0,
            # vp >= self.Al3 + self.Bl2 @ U,
            vp >= self.Al3 + sum(self.Bl2[:,list(self.num_x3 * i +np.array(range(self.num_x3)))] @ cp.diag(U[list(self.num_x3 * i +np.array(range(self.num_x3)))]) for i in range(m)),

        ]
        # for i in range(self.num_x3):
        #     for j in range(m):
        #         for k in range(self.num_x3):
        #             if k != i:
        #                 csts += [
        #                     U[self.num_x3 * j+k, i] == 0
        #                 ]
        prbl = cp.Problem(cp.Maximize(x3 @ alpha + beta + 0 * cp.sum(cp.sum(U))), csts)
        return x3, y1, prbl

    def LowerLevel_JI(self):
        x3 = cp.Parameter(self.num_x3)
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        csts = [
            self.Al3 @ x3 + self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl
        ]
        prbl = cp.Problem(cp.Maximize(self.dl1 @ y1 + self.dl2 @ y2), csts)
        return x3, prbl

    def MILP_JiangsCut3(self):
        x3 = cp.Parameter(self.num_x3)

        alpha = cp.Variable(self.num_x3)
        beta = cp.Variable()
        if self.num_y1 == 0:
            y1 = np.zeros(self.num_y1)
        else:
            y1 = cp.Variable(self.num_y1, boolean=True)
        y2 = cp.Variable(self.num_y2)
        yy2 = cp.Variable((self.num_y2,self.num_x3))
        yyy2 = cp.Variable((self.num_y2,self.num_x3))
        U = cp.Variable((self.num_x3,len(self.hl)))
        lam = cp.Variable(self.num_x3)
        csts = [
            beta <= self.dl1 @ y1 + self.dl2 @ y2 + np.ones(self.num_x3) @ lam,
            self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl + U.T @ np.ones(self.num_x3),
            lam <= self.dl2 @ yy2,
            self.Bl2 @ yy2 <= -U.T,
            lam + alpha <= self.dl2 @ yyy2,
            self.Bl2 @ yyy2 <= -U.T-self.Al3,
        ]
        # for i in range(self.num_x3):
        #     csts += [
        #         lam[i] <= self.dl2 @ yy2[:,i],
        #         self.Bl2 @ yy2[:,i] <= -U[i,:],
        #         lam[i] + alpha[i] <= self.dl2 @ yyy2[:,i],
        #         self.Bl2 @ yyy2[:,i] <= -U[i,:]-self.Al3[:,i],
        #     ]
        prbl = cp.Problem(cp.Maximize(x3 @ alpha + beta), csts)
        return x3, prbl

    def MILP_JiangsCut3_fixed(self):
        x3 = cp.Parameter(self.num_x3)
        y1 = cp.Parameter(self.num_y1)

        alpha = cp.Variable(self.num_x3)
        beta = cp.Variable()
        y2 = cp.Variable(self.num_y2)
        yy2 = cp.Variable((self.num_y2,self.num_x3))
        yyy2 = cp.Variable((self.num_y2,self.num_x3))
        U = cp.Variable((self.num_x3,len(self.hl)))
        lam = cp.Variable(self.num_x3)
        csts = [
            beta <= self.dl1 @ y1 + self.dl2 @ y2 + np.ones(self.num_x3) @ lam,
            self.Bl1 @ y1 + self.Bl2 @ y2 <= self.hl + U.T @ np.ones(self.num_x3),
            lam <= self.dl2 @ yy2,
            self.Bl2 @ yy2 <= -U.T,
            lam + alpha <= self.dl2 @ yyy2,
            self.Bl2 @ yyy2 <= -U.T-self.Al3,
        ]
        prbl = cp.Problem(cp.Maximize(x3 @ alpha + beta), csts)
        return x3, y1, prbl
