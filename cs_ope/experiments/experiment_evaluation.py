import numpy as np
import argparse
import sys
from pathlib import Path
import os
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
link=Path(os.path.abspath(__file__))
link=link.parent.parent
link=os.path.join(link)
sys.path.append(link)
print(link)
from cs_ope_estimator import ope_estimators



def process_args(arguments):
    """
    Sử lý câu lệnh
    """
    parser = argparse.ArgumentParser(
        description='Covariate Shift Adaptation for Off-policy Evaluation and Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', type=str, default='satimage.scale',
                        help='Name of dataset')
    parser.add_argument('--sample_size', '-s', type=int, default=1000,
                        help='Sample size')
    parser.add_argument('--num_trials', '-n', type=int, default=20,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['satimage300', 'vehicle300', 'pendigits300',
                        'satimage500', 'vehicle500', 'pendigits500','satimage1000', 'pendigits1000','satimage800'],
                        help="Presets of configuration")
    args = parser.parse_args(arguments)

    if args.preset == 'satimage800':
        args.sample_size = 800
        args.dataset = 'satimage'
        args.num_trials = 20
    elif args.preset == 'vehicle':
        args.sample_size = 800
        args.dataset = 'vehicle'
        args.num_trials = 50
    elif args.preset == "pendigits":
        args.sample_size = 800
        args.dataset = 'pendigits'
        args.num_trials = 50

    if args.preset == 'satimage300':
        args.sample_size = 300
        args.dataset = 'satimage'
        args.num_trials = 20
    elif args.preset == 'vehicle300':
        args.sample_size = 300
        args.dataset = 'vehicle'
        args.num_trials = 50
    elif args.preset == "pendigits300":
        args.sample_size = 300
        args.dataset = 'pendigits'
        args.num_trials = 50

    if args.preset == 'satimage500':
        args.sample_size = 500
        args.dataset = 'satimage'
        args.num_trials = 20
    elif args.preset == 'vehicle500':
        args.sample_size = 500
        args.dataset = 'vehicle'
        args.num_trials = 50
    elif args.preset == "pendigits500":
        args.sample_size = 500
        args.dataset = 'pendigits'
        args.num_trials = 50

    if args.preset == 'satimage1000':
        args.sample_size = 1000
        args.dataset = 'satimage'
        args.num_trials = 20
    elif args.preset == "pendigits1000":
        args.sample_size = 1000
        args.dataset = 'pendigits'
        args.num_trials = 20

    return args

def data_generation(data_name, N):
    """
    Khởi tạo data
    data_name:Tên 
    N: số lượng
    """
    X, Y = load_svmlight_file(r'D:\Lab\bandits\cs_ope\cs_ope\experiments\data/%s'%data_name)
    X = X.toarray()
    X = X/X.max(axis=0)
    Y = np.array(Y, np.int64)

    N_train = int(N*0.95)
    N_test= N - N_train

    perm = np.random.permutation(len(X))

    X, Y = X[perm[:N]], Y[perm[:N]]

    if data_name == 'satimage.scale':
        Y = Y - 1
    elif data_name == 'vehicle.scale':
        Y = Y - 1

    classes = np.unique(Y)

    Y_matrix = np.zeros(shape=(N, len(classes)))

    for i in range(N):
        Y_matrix[i, Y[i]] = 1

    prob = 1/(1+np.exp(-(X[:,0]+X[:,1]+X[:,2]+X[:,3]+X[:,4]+0.1*np.random.normal(size=len(X)))))
    rand = np.random.uniform(size=len(X))

    prob_base = prob

    for C in range(100000):
        C /= 1000
        eval_samplesize = np.sum(prob*C > rand)
        if eval_samplesize > N_train:
            break
        prob_base = prob*C

    train_test_split = prob_base > rand
    
    if np.sum(train_test_split) != N_train:
        N_train = np.sum(train_test_split)
        N_test = N - N_train
    
    return X, Y, Y_matrix, train_test_split, classes, N, N_train, N_test

def behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha=0.7):
    """
    Hàm behavior_and_evaluation_policy nhận đầu vào là các đặc trưng (X), nhãn (Y), phân chia dữ liệu huấn luyện và kiểm tra (train_test_split), 
    danh sách các lớp (classes), và một siêu tham số alpha
    return: pi_behavior, pi_evaluation (policy)
    """
    N = len(X)
    num_class = len(classes)
    
    X_train = X[train_test_split]
    Y_train = Y[train_test_split]
    
    classifier = LogisticRegression(random_state=0, penalty='l2', C=0.1, solver='saga', multi_class='multinomial',).fit(X_train, Y_train)
    predict = np.array(classifier.predict(X), np.int64)

    pi_predict = np.zeros(shape=(N, num_class))

    for i in range(N):
        pi_predict[i, predict[i]] = 1

    pi_random = np.random.uniform(size=(N, num_class))
    
    pi_random = pi_random.T
    pi_random /= pi_random.sum(axis=0)
    pi_random = pi_random.T
    
    
    pi_behavior = alpha*pi_predict + (1-alpha)*pi_random
        
    pi_evaluation = 0.9*pi_predict + 0.1*pi_random
        
    return pi_behavior, pi_evaluation

def true_value(Y_matrix, pi_evaluation, N):
     """
     Y*pi
     """
     return np.sum(Y_matrix*pi_evaluation)/N
    
def main(arguments):
    args = process_args(arguments)

    data_name = args.dataset
    num_trials = args.num_trials
    sample_size = args.sample_size

    if data_name == 'satimage':
        data_name = 'satimage.scale'
    elif data_name == 'vehicle':
        data_name = 'vehicle.scale'
    
    # alphas = [0.7, 0.4, 0.0]
    alphas = [ 0.4]

    tau_list = np.zeros(num_trials)
    res_ipw3_list = np.zeros((num_trials, len(alphas)))
    res_dm_list = np.zeros((num_trials, len(alphas)))
    res_dml1_list = np.zeros((num_trials, len(alphas)))
    res_dml2_list = np.zeros((num_trials, len(alphas)))

    res_ipw3_sn_list = np.zeros((num_trials, len(alphas)))
    res_dml1_sn_list = np.zeros((num_trials, len(alphas)))
    res_dml2_sn_list = np.zeros((num_trials, len(alphas)))

    res_ipw3_ML_list = np.zeros((num_trials, len(alphas)))
    res_ipw3_ML_sn_list = np.zeros((num_trials, len(alphas)))
    res_dm_ML_list = np.zeros((num_trials, len(alphas)))

    for trial in range(num_trials):
        X, Y, Y_matrix, train_test_split, classes, N, N_train, N_test = data_generation(data_name, sample_size)
        print(N,N_train,N_test)
        X_train, X_test = X[train_test_split], X[~train_test_split]
        Y_matrix_train, Y_matrix_test = Y_matrix[train_test_split], Y_matrix[~train_test_split]
        

        for idx_alpha in range(len(alphas)):    
            alpha = alphas[idx_alpha]

            pi_behavior, pi_evaluation  = behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha=alpha)
            pi_behavior_train = pi_behavior[train_test_split]
            pi_evaluation_train, pi_evaluation_test = pi_evaluation[train_test_split], pi_evaluation[~train_test_split]

            tau = true_value(Y_matrix_test, pi_evaluation_test, N_test)
            tau_list[trial] = tau

            perm = np.random.permutation(N_train)

            X_seq_train, Y_matrix_seq_train, pi_behavior_seq_train, pi_evaluation_seq_train = X_train[perm], Y_matrix_train[perm], pi_behavior_train[perm], pi_evaluation_train[perm]

            Y_historical_matrix = np.zeros(shape=(N_train, len(classes)))
            A_historical_matrix = np.zeros(shape=(N_train, len(classes)))

            for i in range(N_train):
                a = np.random.choice(classes, p=pi_behavior_seq_train[i])
                Y_historical_matrix[i, a] = Y_matrix_seq_train[i, a]
                A_historical_matrix[i, a] = 1

            estimators = ope_estimators(X_seq_train, A_historical_matrix, Y_historical_matrix, X_test, classes, pi_evaluation_seq_train, pi_evaluation_test)
            res_ipw3 = estimators.ipw(self_norm=False)
            res_ipw3_sn = estimators.ipw(self_norm=True)
            res_ipw3_ML = estimators.ipw_ML(method='Ridge', self_norm=False)
            res_ipw3_ML_sn = estimators.ipw_ML(method='Ridge', self_norm=True)
            res_dm = estimators.dm()
            res_dm_ML = estimators.dm_ML(method='Ridge')
            res_dml1 = estimators.dml(self_norm=False, method='Lasso')
            res_dml2 = estimators.dml(self_norm=False, method='Ridge')
            res_dml1_sn = estimators.dml(self_norm=True, method='Lasso')
            res_dml2_sn = estimators.dml(self_norm=True, method='Ridge')

            print('trial', trial)
            print('True:', tau)
            print('IPW3:', res_ipw3)
            print('IPW3_SN:', res_ipw3_sn)
            print('DM:', res_dm)
            print('DML1:', res_dml1)
            print('DML1_SN:', res_dml1_sn)
            print('DML2:', res_dml2)
            print('DML2_ML:', res_dml2_sn)
            print('IPW3_ML:', res_ipw3_ML)
            print('IPW3_ML_SN:', res_ipw3_ML_sn)
            print('DM_ML:', res_dm_ML)

            res_ipw3_list[trial, idx_alpha] = res_ipw3
            res_ipw3_sn_list[trial, idx_alpha] = res_ipw3_sn
            res_dm_list[trial, idx_alpha] = res_dm
            res_dml1_list[trial, idx_alpha] = res_dml1
            res_dml1_sn_list[trial, idx_alpha] = res_dml1_sn
            res_dml2_list[trial, idx_alpha] = res_dml2
            res_dml2_sn_list[trial, idx_alpha] = res_dml2_sn
            res_ipw3_ML_list[trial, idx_alpha] = res_ipw3_ML
            res_ipw3_ML_sn_list[trial, idx_alpha] = res_ipw3_ML_sn
            res_dm_ML_list[trial, idx_alpha] = res_dm_ML
            os.makedirs("results_ope/", exist_ok=True)

            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/true_value_%s_%d.csv"%(data_name, sample_size), tau_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_ipw3_%s_%d.csv"%(data_name, sample_size), res_ipw3_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_ipw3_sn_%s_%d.csv"%(data_name, sample_size), res_ipw3_sn_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_dm_%s_%d.csv"%(data_name, sample_size), res_dm_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_dml1_%s_%d.csv"%(data_name, sample_size), res_dml1_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_dml1_sn_%s_%d.csv"%(data_name, sample_size), res_dml1_sn_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_dml2_%s_%d.csv"%(data_name, sample_size), res_dml2_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_dml2_sn_%s_%d.csv"%(data_name, sample_size), res_dml２_sn_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_ipw3_ML_%s_%d.csv"%(data_name, sample_size), res_ipw3_ML_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_ipw3_ML_sn_%s_%d.csv"%(data_name, sample_size), res_ipw3_ML_sn_list, delimiter=",")
            np.savetxt(r"D:\Lab\bandits\cs_ope\cs_ope\experiments\results_ope_095/res_dm_ML_%s_%d.csv"%(data_name, sample_size), res_dm_ML_list, delimiter=",")

        tau_list[trial] = tau
    
if __name__ == '__main__':
    main(sys.argv[1:])

    
