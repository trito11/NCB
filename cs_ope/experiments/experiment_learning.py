import numpy as np
import argparse
import sys
from pathlib import Path
import os
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import wandb
link=Path(os.path.abspath(__file__))
link=link.parent.parent
link=os.path.join(link)
sys.path.append(link)
from absl import flags, app
from cs_opl import op_learning
from ucimlrepo import fetch_ucirepo 
import  multiprocessing 
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch    
# print(torch.cuda.is_available())
# print(torch.cuda.current_device())

def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='Covariate Shift Adaptation for Off-policy Evaluation and Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', type=str, default='vehicle',
                        help='Name of dataset')
    parser.add_argument('--sample_size', '-s', type=int, default=1000,
                        help='Sample size')
    parser.add_argument('--ration_size', '-r', type=float, default=0.7,
                        help='Ratio size')
    parser.add_argument('--num_trials', '-n', type=int, default=200,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        # choices=['satimage', 'vehicle', 'pendigits'],
                        help="Presets of configuration")
    parser.add_argument('--data_type', type=str, default='quadratic', help='Dataset to sample from')
    parser.add_argument('--policy', type=str, default='eps-greedy', help='Offline policy, eps-greedy/subset')
    parser.add_argument('--eps', type=float, default=0.1, help='Probability of selecting a random action in eps-greedy')
    parser.add_argument('--subset_r', type=float, default=0.5, help='The ratio of the action spaces to be selected in offline data')
    parser.add_argument('--num_contexts', type=int, default=10000, help='Number of contexts for training.')
    parser.add_argument('--num_test_contexts', type=int, default=10000, help='Number of contexts for test.')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose')
    parser.add_argument('--debug', type=bool, default=True, help='debug')
    parser.add_argument('--normalize', type=bool, default=False, help='normalize the regret')
    parser.add_argument('--update_freq', type=int, default=1, help='Update frequency')
    parser.add_argument('--freq_summary', type=int, default=10, help='Summary frequency')
    parser.add_argument('--test_freq', type=int, default=10, help='Test frequency')
    parser.add_argument('--algo_group', type=str, default='approx-neural', help='baseline/neural')
    parser.add_argument('--num_sim', type=int, default=10, help='Number of simulations')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Noise std')
    parser.add_argument('--chunk_size', type=int, default=500, help='Chunk size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps to train NN.')
    parser.add_argument('--buffer_s', type=int, default=-1, help='Size in the train data buffer.')
    parser.add_argument('--data_rand', type=bool, default=True, help='Whether to randomly sample a data batch or use the latest samples in the buffer')
    parser.add_argument('--rbf_sigma', type=float, default=1, help='RBF sigma for KernLCB')  # [0.1, 1, 10]
    parser.add_argument('--beta', type=float, default=0.1, help='confidence parameter')  # [0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lambd0', type=float, default=0.1, help='minimum eigenvalue')
    parser.add_argument('--lambd', type=float, default=1e-4, help='regularization parameter')
    args = parser.parse_args(arguments)

    if args.preset == 'satimage':
        args.sample_size = 800
        args.dataset = 'satimage'
        args.num_trials = 10
    elif args.preset == 'vehicle':
        args.sample_size = 800
        args.dataset = 'vehicle'
        args.num_trials = 10
    elif args.preset == "pendigits":
        args.sample_size = 800
        args.dataset = 'pendigits'
        args.num_trials = 10
    elif args.preset == "heart_disease":
        args.sample_size = 300
        args.dataset = 'heart_disease'
        args.num_trials = 10
    elif args.preset == "breast_cancer":
        args.sample_size = 500
        args.dataset = 'breast_cancer'
        args.num_trials = 10
    elif args.preset == "mammographic_mass":
        args.sample_size = 800
        args.dataset = 'mammographic_mass'
        args.num_trials = 10
    elif args.preset == "cardiotocography":
        args.sample_size = 800
        args.dataset = 'cardiotocography'
        args.num_trials = 10
    elif args.preset == "blood_transfusion_service_center":
        args.sample_size = 700
        args.dataset = 'blood_transfusion_service_center'
        args.num_trials = 10
    return args


def remove_underrepresented_classes(features, labels, thresh=0.0005):
    """Removes classes when number of datapoints fraction is below a threshold."""

    # Threshold doesn't seem to agree with https://arxiv.org/pdf/1706.04687.pdf
    # Example: for Covertype, they report 4 classes after filtering, we get 7?
    total_count = labels.shape[0]
    unique, counts = np.unique(labels, return_counts=True)
    ratios = counts.astype('float') / total_count
    vals_and_ratios = dict(zip(unique, ratios))
    print('Unique classes and their ratio of total: %s' % vals_and_ratios)
    keep = [vals_and_ratios[v] >= thresh for v in labels]
    return features[keep], labels[np.array(keep)]

def classification_to_bandit_problem(contexts, labels, num_actions=None):
    """Normalize contexts and encode deterministic rewards."""

    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

    # Normalize features
    contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

    # One hot encode labels as rewards
    rewards = np.zeros((num_contexts, num_actions))
    rewards[np.arange(num_contexts), labels] = 1.0

    return contexts, rewards #, (np.ones(num_contexts), labels)

def safe_std(values):
    """Remove zero std values for ones."""
    return np.array([val if val != 0.0 else 1.0 for val in values])

def data_generation(data_name, N, ration):
    id_map={'heart_disease':45,'breast_cancer':17,'mammographic_mass':161,'blood_transfusion_service_center':176,'cardiotocography':193}
    if data_name in id_map.keys():
        data=fetch_ucirepo(id=id_map[data_name]) 
        X=data.data.features 
        Y=data.data.targets 
        if data_name=='cardiotocography':
            name=Y.columns.tolist()
            Y['new_label']=Y[name[0]].astype(str) + '_' + Y[name[1]].astype(str)
            Y= label_encoder.fit_transform(Y['new_label'])

        else:
            name=Y.columns.tolist()[0]
            Y.loc[:, name] = Y[name].astype('category').cat.codes
        X = X.interpolate().values
        Y = np.squeeze(Y)
        Y = np.array(Y, np.int64)
    else:
        X, Y = load_svmlight_file(r'/home/tnguye11/ailab/tri/cs_ope/cs_ope/experiments/data/%s'%data_name)
        X = X.toarray()
        Y = np.array(Y, np.int64)

    X = X/X.max(axis=0)
    

    N_train = int(N*ration)
    N_test= N - N_train

    perm = np.random.permutation(len(X))

    X, Y = X[perm[:N]], Y[perm[:N]]

    if data_name == 'satimage.scale':
        Y = Y - 1
    elif data_name == 'vehicle.scale':
        Y = Y - 1

    unique_labels = np.sort(np.unique(Y))
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    Y=np.array([label_mapping[label] for label in Y])
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

def behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha):
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
     return np.sum(Y_matrix*pi_evaluation)/N

def run_estimator_tasks(task):
    estimators, task_name, params = task
    if task_name == "IPW":
        estimators.ipw_est_parameters()
        return estimators.ipw_fit(**params)
    elif task_name == "DM":
        estimators.dm_est_parameters()
        return estimators.dm_fit(**params)
    elif task_name == "DML":
        estimators.dml_est_parameters(folds=2, method='Ridge')
        return estimators.dml_fit(**params)
    
def main(arguments):
    wandb.login(
    
    key='b98d2b806f364f5af900550ec98e26e2f418e8a7'
    )
    wandb.init(project='cs_ope')
    args = process_args(arguments)

    data_name = args.dataset
    num_trials = args.num_trials
    sample_size = args.sample_size
    ratio=args.ration_size

    if data_name == 'satimage':
        data_name = 'satimage.scale'
    elif data_name == 'vehicle':
        data_name = 'vehicle.scale'

    alphas = [ 0.7, 0.4, 0.0]

    tau_list = np.zeros(num_trials)
    res_ipw3_list = np.zeros((num_trials, len(alphas)))
    res_dm_list = np.zeros((num_trials, len(alphas)))
    res_dml2_list = np.zeros((num_trials, len(alphas)))

    res_ipw3_sn_list = np.zeros((num_trials, len(alphas)))
    res_dml2_sn_list = np.zeros((num_trials, len(alphas)))
    res_NLCB_list=np.zeros((num_trials, len(alphas)))

    for trial in range(num_trials):
        X, Y, Y_matrix, train_test_split, classes, N, N_train, N_test = data_generation(data_name, sample_size,ratio)

        X_train, X_test = X[train_test_split], X[~train_test_split]

        Y_matrix_train, Y_matrix_test = Y_matrix[train_test_split], Y_matrix[~train_test_split]

        for idx_alpha in  range(len(alphas)):    
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
            Action=np.zeros(shape=(N_train),dtype=np.int16)

            for i in range(N_train):
                a = np.random.choice(classes, p=pi_behavior_seq_train[i])
                Y_historical_matrix[i, a] = Y_matrix_seq_train[i, a]
                A_historical_matrix[i, a] = 1
                Action[i]=a

            # dataset = (contexts, actions, rewards, test_contexts, mean_test_rewards)

            estimators = op_learning(X_seq_train, A_historical_matrix, Action, Y_historical_matrix, X_test, Y_matrix_test,classes,args,lcb=True)
            epol_nlcb=estimators.Neural_LCB_est_parameters()
            res_NLCB=true_value(Y_matrix_test, epol_nlcb, N_test)
            wandb.log({f"{data_name}-{alpha}_{ratio}_{N}-res_NLCB":res_NLCB})
            print('start')
            estimators = op_learning(X_seq_train, A_historical_matrix, Action, Y_historical_matrix, X_test, Y_matrix_test,classes,args,lcb=False)
            # estimators.ipw_est_parameters()
            # epol_ipw = estimators.ipw_fit(folds=2, algorithm='Ridge', self_norm=False)
            # epol_ipw_sn = estimators.ipw_fit(folds=5, algorithm='Ridge', self_norm=True)

            # estimators.dm_est_parameters()
            # epol_dm = estimators.dm_fit(folds=2, algorithm='Ridge')
            
            # estimators.dml_est_parameters(folds=2, method='Ridge')
            # epol_dml = estimators.dml_fit(folds=2, algorithm='Ridge', self_norm=False)
            # epol_dml_sn = estimators.dml_fit(folds=2, algorithm='Ridge', self_norm=True)

            # res_NLCB=true_value(Y_matrix_test, epol_nlcb, N_test)
            # res_ipw3 =  true_value(Y_matrix_test, epol_ipw, N_test)
            # res_ipw3_sn =  true_value(Y_matrix_test, epol_ipw_sn, N_test)
            # res_dm =  true_value(Y_matrix_test, epol_dm, N_test)
            # res_dml2 =  true_value(Y_matrix_test, epol_dml, N_test)
            # res_dml2_sn =  true_value(Y_matrix_test, epol_dml_sn, N_test)
            
            tasks = [
                
                (estimators, "IPW", {"folds": 2, "algorithm": "Ridge", "self_norm": False}),
                (estimators, "IPW", {"folds": 5, "algorithm": "Ridge", "self_norm": True}),
                (estimators, "DM", {"folds": 2, "algorithm": "Ridge"}),
                (estimators, "DML", {"folds": 2, "algorithm": "Ridge", "self_norm": False}),
                (estimators, "DML", {"folds": 2, "algorithm": "Ridge", "self_norm": True}),
            ]
            # multiprocessing.set_start_method('spawn')
            # Chạy song song các tác vụ
            with multiprocessing.Pool(processes=5) as pool:  # Sử dụng 6 tiến trình
                results = pool.map(run_estimator_tasks, tasks)
            epol_ipw, epol_ipw_sn, epol_dm, epol_dml, epol_dml_sn = results

            
            res_ipw3 =  true_value(Y_matrix_test, epol_ipw, N_test)
            res_ipw3_sn =  true_value(Y_matrix_test, epol_ipw_sn, N_test)
            res_dm =  true_value(Y_matrix_test, epol_dm, N_test)
            res_dml2 =  true_value(Y_matrix_test, epol_dml, N_test)
            res_dml2_sn =  true_value(Y_matrix_test, epol_dml_sn, N_test)
            print('trial', trial)
            print('True:', tau)
            print('IPW3:', res_ipw3)
            print('IPW3_SN:', res_ipw3_sn)
            print('DM:', res_dm)
            print('DML2:', res_dml2)
            print('DML2_SN:', res_dml2_sn)
            print('NLCB',res_NLCB)
            wandb.log({f"{data_name}-{alpha}_{ratio}_{N}-trial":trial})
            wandb.log({f"{data_name}-{alpha}_{ratio}_{N}-tau":tau})
            wandb.log({f"{data_name}-{alpha}_{ratio}_{N}-res_ipw3":res_ipw3})
            wandb.log({f"{data_name}-{alpha}_{ratio}_{N}-res_ipw3_sn":res_ipw3_sn})
            wandb.log({f"{data_name}-{alpha}_{ratio}_{N}-res_dm":res_dm})
            wandb.log({f"{data_name}-{alpha}_{ratio}_{N}-res_dml2":res_dml2})
            wandb.log({f"{data_name}-{alpha}_{ratio}_{N}-res_dml2_sn":res_dml2_sn})


            

            res_ipw3_list[trial, idx_alpha] = res_ipw3
            res_ipw3_sn_list[trial, idx_alpha] = res_ipw3_sn
            res_dm_list[trial, idx_alpha] = res_dm
            res_dml2_list[trial, idx_alpha] = res_dml2
            res_dml2_sn_list[trial, idx_alpha] = res_dml2_sn
            res_NLCB_list[trial, idx_alpha] = res_NLCB
            directory = rf"results/exp_results_{ratio}_{N}"

            # Check if the directory exists, and create it if not
            if not os.path.exists(directory):
                os.makedirs(directory)
            print(data_name)
            with open(rf"{directory}/true_opl_value_{data_name}.csv", "a") as f:
                np.savetxt(f,tau_list, delimiter=",")
            with open(rf"{directory}/res_opl_ipw3_{data_name}.csv", "a") as f:  
                np.savetxt(f, res_ipw3_list, delimiter=",")
            with open(rf"{directory}/res_opl_ipw3_sn_{data_name}.csv", "a") as f:  
                np.savetxt(f, res_ipw3_sn_list, delimiter=",")
            with open(rf"{directory}/res_opl_dm_{data_name}.csv", "a") as f:  
                np.savetxt(f, res_dm_list, delimiter=",")
            with open(rf"{directory}/res_opl_dml2_{data_name}.csv", "a") as f:  
                np.savetxt(f, res_dml2_list, delimiter=",")
            with open(rf"{directory}/res_opl_dml2_sn_{data_name}.csv", "a") as f:  
                np.savetxt(f, res_dml２_sn_list, delimiter=",")
            with open(rf"{directory}/true_res_NLCB_value_{data_name}.csv", "a") as f:  
                np.savetxt(f, res_NLCB_list, delimiter=",")
            


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main(sys.argv[1:])



    
