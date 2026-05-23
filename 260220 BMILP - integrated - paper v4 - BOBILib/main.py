from instances import *

# diverse instances
# list_n = [460]  # 10, 60, 110, 160, 210, 260, 310, 360, 410, 460
# list_run = [1 , 2, 3, 4, 5, 6, 7, 8, 9, 10]  #
# for n in list_n:
#     for idx_run in list_run:
#         print('\n\n\n\n\n\n\n\nRound' + str(idx_run) + '===================================================')
#         instance = instance_xuwang(n, idx_run)
#         # continue
list_n = [1000]  # 500, 600, 700, 800, 900, 1000
list_run = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # 1 ,
for n in list_n:
    print('\n\n\n\n\n\n\n\nn=' + str(n) + '===================================================')
    for idx_run in list_run:
        print('\n\n\n\n\n\n\n\nRound' + str(idx_run) + '===================================================')
        instance = instance_xularge(n, idx_run)
        # continue

# # instance generation
# list_num_facility = [5]  # 5, 10, 15, 20, 25, 30
# flag_MILP = True
# for num_facility in list_num_facility:
#     num_customer = num_facility * 10
#     np.random.seed(1)
#     random.seed(1)
#     instance = instance_CFL(num_facility, num_customer, flag_MILP)
# list_num_x = [2000]  # 10, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000
# # list_ratio_y_bianry = [0.1, 0.3, 0.7, 0.9]  # 0.1, 0.3, 0.5, 0.7, 0.9
# # list_sparsity = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 0, 0.2, 0.4, 0.6, 0.8
# # list_factor_scale = [1.5]  # 0.5, 0.75, 1, 1.25, 1.5
# # num_x = 2000
# for num_x in list_num_x:
# # for ratio_y_bianry in list_ratio_y_bianry:
# # for sparsity in list_sparsity:
# # for factor_scale in list_factor_scale:
#     np.random.seed(1)
#     # instance = instance_MILP(num_x)
#     # instance = instance_MILP(num_x, ratio_y_bianry=ratio_y_bianry)
#     # instance = instance_MILP(num_x, sparsity=sparsity)
#     instance = instance_MILP(num_x, factor_scale=factor_scale)

        flag_method = 1

        if flag_method == 1:
            T1 = time.time()
            value_x1_MIBS, value_x2_MIBS, value_x3_MIBS, value_y1_MIBS, value_y2_MIBS, value_obj_upper_MIBS, value_obj_lower_MIBS = instance.solvebyMIBS()
            T2 = time.time()
            instance.time_MIBS_total = T2 - T1
            obj_MIBS = round(value_obj_upper_MIBS, 4)
            time_MIBS = round(instance.time_MIBS_total, 4)
            print('===Solved by MIBS solver===')
            print('Objective value:', obj_MIBS, '<--- directly from MIBS:', round(instance.objctv_upper_MIBS_original, 4))
            print('Computation time:', time_MIBS)

        if flag_method == 2:
            num_samples_target = 1000
            T1 = time.time()
            value_x1_NN, value_x2_NN, value_x3_NN, value_y1_NN, value_y2_NN, value_obj_upper_NN, value_obj_lower_NN = instance.solvebyNN(num_samples_target)
            T2 = time.time()
            instance.time_NN_total = T2 - T1
            obj_NN = round(value_obj_upper_NN, 4)
            time_NN = round(instance.time_NN_total, 4)
            print('===Solved by NN approximation===')
            print('ISNN:', instance.flag_ISNN, '| Double input layer:', instance.flag_doubleX)
            print('Iteration number of enhanced sampling:', instance.num_iter_enhanced)
            print('Number of samples:', len(instance.samples[:,0]), '| NN architecture:', instance.NNArchitecture)
            print('Solution algorithm:', 'Cutting Plane' if instance.flag_NNSolveByCut else 'MIP representation')
            print('Objective value:', obj_NN)
            print('Computation time:', time_NN, '~= Sampling', round(instance.time_NN_sampling, 2), '+ Training', round(instance.time_NN_training, 2), '+ Solving', round(instance.time_NN_solving, 2), '+ Checking', round(instance.time_NN_checking, 2))

        if flag_method == 3:
            T1 = time.time()
            value_x1_CP, value_x2_CP, value_x3_CP, value_y1_CP, value_y2_CP, value_obj_upper_CP, value_obj_lower_CP = instance.solvebyCP()
            T2 = time.time()
            instance.time_CP_total = T2 - T1
            instance.time_CP_heuristic_total = instance.time_CP_heuristic_sampling + instance.time_CP_heuristic_training + instance.time_CP_heuristic_solving + instance.time_CP_heuristic_checking
            instance.time_CP_solving = instance.time_CP_total - instance.time_CP_heuristic_total
            obj_CP = round(value_obj_upper_CP, 4)
            time_CP = round(instance.time_CP_total, 4)
            print('===Solved by Proposed Cutting Planes===')
            if instance.flag_LowerLevel == 0:
                print('Problem Property:', 'None')
            else:
                print('Problem Property:', 'Supermodular' if instance.flag_LowerLevel == 1 else 'Submodular', 'under fixed y1 of indices', instance.idx_y1_fixed)
            print('Lazy Constraints ---> Augmented Lagrangian:', instance.flag_ALI, '| Lagrangian:', instance.flag_LI, '| Supermodular/Submodular:', instance.flag_SI)
            print('No-good cuts:', instance.flag_no_good)
            print('Incumbent update:', instance.flag_incumbent)
            if instance.flag_heuristic_UB | instance.flag_heuristic_LB:
                print('Heuristic UB:', instance.flag_heuristic_UB)
                if instance.flag_heuristic_UB:
                    print('Heuristic UB ---> Warm-start:', instance.flag_heuristic_UB_warmstart, '| Bounds:', instance.flag_heuristic_UB_incumbent)
                    print('Heuristic UB ---> Methods:', 'Sampling, Training, and Solving' if instance.flag_heuristic_UB_solving else 'Sampling only')
                    print('Heuristic UB ---> Frequency: every', instance.heuristic_UB_frequency, 'nodes')
                    print('Heuristic UB ---> Target number of initial samples:', instance.heuristic_UB_num_samples_target)
                    if not instance.flag_heuristic_UB_solving:
                        print('Heuristic UB ---> Number of initial training:', instance.heuristic_UB_num_epoch_initial, '| Number of continued training:', instance.heuristic_UB_num_epoch_continued)
                    else:
                        print('Heuristic UB ---> Number of sampling in every call:', instance.heuristic_UB_num_sampling_only)
                    print('Heuristic UB ---> ISNN:', instance.flag_ISNN, '| Double input layer:', instance.flag_doubleX, '| NN architecture:', instance.heuristic_UB_NNArchitecture)
                    print('Heuristic UB ---> Solution algorithm: MIP representation')
                    print('Heuristic UB ---> Number of win:', instance.heuristic_UB_num_win, '/', instance.heuristic_UB_num_all)
                print('Heuristic LB:', instance.flag_heuristic_LB)
                # if instance.flag_heuristic_LB:
                #     print('Heuristic UB ---> Lower-level ---> ISNN: False | Double input layer: False | NN architecture:', instance.NNArchitecture_lower)
            else:
                print('Heuristic Bound: False')
            print('Objective value:', obj_CP)
            print('Computation time:', time_CP, '~= Solving', round(instance.time_CP_solving, 2), ' + Heuristic', round(instance.time_CP_heuristic_total, 2))
            if instance.flag_LI | instance.flag_ALI:
                print('Lagrangian coefficient time:', round(instance.time_Lagrangian, 2))
            if instance.flag_SI:
                print('EPI coefficient time:', round(instance.time_EPI_pi, 2))
            print('Phi time:', round(instance.time_Phi, 2), '/', len(instance.buffer_phi))
            if len(instance.idx_y1_fixed) != 0:
                print('varPhi time:', round(instance.time_varPhi, 2), '/', len(instance.buffer_varphi))
            if instance.flag_buffer_upper:
                print('Feasible time:', round(instance.time_feasible, 2), '/', len(instance.buffer_obj_upper))
            if instance.flag_heuristic_UB | instance.flag_heuristic_LB:
                print('Heuristic time:', round(instance.time_CP_heuristic_total, 2), '~= Sampling', round(instance.time_CP_heuristic_sampling, 2), '+ Training', round(instance.time_CP_heuristic_training, 2), '+ Solving', round(instance.time_CP_heuristic_solving, 2), '+ Checking', round(instance.time_CP_heuristic_checking, 2))

pass