import numpy as np
import pandas as pdå
import statsmodels.api as sm

from kernel_regression import KernelRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.neighbors import KernelDensity
from densratio import densratio

from sklearn.kernel_ridge import KernelRidge
import torch
import torch.optim as optim
import torch.nn as nn


class ope_estimators():
    def __init__(self, X, A, Y_matrix, Z, classes, pi_evaluation_train, pi_evaluation_test):
        """
        X: feature matrix train 
        A: historical_matrix (action)
        Y_matrix: Y_historical_matrix (Reward: True=1, False= 0)
        Z: X_test (feature matrix test)
        classes: class
        pi_evaluation_train: policy evaluation train 
        pi_evaluation_test: policy evaluation test


        """
        self.X = X
        self.A = A
        self.Y = Y_matrix
        self.Z = Z
        self.classes = classes
        self.pi_evaluation_train = pi_evaluation_train
        self.pi_evaluation_test = pi_evaluation_test

        self.N_hst, self.dim = X.shape
        self.N_evl = len(Z)

        self.f_hat_kernel = None
        self.bpol_hat_kernel = None
        self.q_hat_kernel = None
        self.p_hat_kernel = None
        self.r_ML_hat_kernel = None
        self.f_ML_hat_kernel = None
        self.bpol_ML_hat_kernel = None
        X = torch.tensor(self.X, dtype=torch.float32)
        y = torch.tensor(self.Y, dtype=torch.long)
        self.model = FCL2Layer(self.dim, self.classes)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=0.1)
        num_epochs = 100
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.model(X)
            loss = criterion(outputs, y)

            # Backward pass và tối ưu hóa
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")





    def ipw(self, self_norm=True):
        """
        Hàm ipw thực hiện ước lượng của giá trị kỳ vọng có điều kiện thông qua phương pháp Inverse Probability Weighting (IPW)
        Hàm KDEMultivariate được sử dụng để ước lượng mật độ xác suất (PDF - Probability Density Function) của dữ liệu đa biến bằng cách sử dụng phân phối Gaussian kernel
        """        
        if self.p_hat_kernel is None:
            dens = sm.nonparametric.KDEMultivariate(data=self.X, var_type='c'*self.dim, bw='normal_reference')
            self.p_hat_kernel = dens.pdf(self.X)
            
        if self.q_hat_kernel is None:
            dens = sm.nonparametric.KDEMultivariate(data=self.Z, var_type='c'*self.dim, bw='normal_reference')
            self.q_hat_kernel = dens.pdf(self.X)
            
        if self.bpol_hat_kernel is None:
            pi_behavior = np.zeros(shape=(self.N_hst, len(self.classes)))
            for c in self.classes:
                while True:
                    perm = np.random.permutation(self.N_hst)
                    A_temp = self.A[perm[:100]]
                    X_temp = self.X[perm[:100]]
                    model = KernelReg(A_temp[:,c], X_temp, var_type='c'*self.dim, reg_type='lc')
                    #model = KernelReg(self.A[:,c], self.X, var_type='c'*self.dim, reg_type='lc', bw=model.bw)
                    #model = KernelReg(self.A[:,c], self.X, var_type='c'*self.dim, reg_type='lc')
                    mu, _ = model.fit(self.X)
                    mu[mu < 0.001] = 0.001
                    mu[mu > 0.999] = 0.999
                    pi_behavior[:, c] = mu
                    if len(mu[~((mu > -100)&(mu < 100))]) == 0:
                        break

            self.bpol_hat_kernel = pi_behavior
        
        r = self.q_hat_kernel/self.p_hat_kernel
        r[r < 0.001] = 0.001
        r[r > 20] = 20
        w = self.pi_evaluation_train/self.bpol_hat_kernel
        
        r = np.array([r for c in range(len(self.classes))]).T
        self.r_hat_kernel = r

        denomnator = self.denominator(self_norm)
        
        return np.sum(self.A*self.Y*w*r/denomnator)

    def dm(self):    
        """
        Direct method Estimator
        """    
        f_matrix = np.zeros(shape=(self.N_evl, len(self.classes)))

        if self.f_hat_kernel is None:
            for c in self.classes:
                while True:
                    perm = np.random.permutation(self.N_hst)
                    Y_temp = self.Y[perm[:100]]
                    X_temp = self.X[perm[:100]]
                    model = KernelReg(Y_temp[:,c], X_temp, var_type='c'*self.dim, reg_type='lc')
                    #model = KernelReg(self.Y[:,c], self.X, var_type='c'*self.dim, reg_type='lc', bw=model.bw)
                    mu, _ = model.fit(self.Z)
                    mu[mu < 0.001] = 0.001
                    mu[mu > 0.999] = 0.999
                    f_matrix[:, c] = mu
                    if len(mu[~((mu > -100)&(mu < 100))]) == 0:
                        break
            
            self.f_hat_kernel = f_matrix
            
        return np.sum(f_matrix*self.pi_evaluation_test)/self.N_evl

    def dml(self, self_norm=True, folds=2, method='Lasso'):
        """
        Hàm này thực hiện phương pháp doubly robust estimator 
        e_fold=fold
        estimate f(a,x) and w(a,x) using method=method   n
        """
        theta_list = []
        
        cv_hst_fold = np.arange(folds) 
        cv_hst_split0 = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        cv_hst_index = cv_hst_split0[np.random.permutation(self.N_hst)]
        
        cv_evl_split0 = np.floor(np.arange(self.N_evl)*folds/self.N_evl)
        cv_evl_index = cv_evl_split0[np.random.permutation(self.N_evl)]

        x_cv = []
        a_cv = []
        y_cv = []
        z_cv = []
        p_evl_hst_cv = []
        p_evl_evl_cv = []
        
        for k in cv_hst_fold:
            x_cv.append(self.X[cv_hst_index==k])
            a_cv.append(self.A[cv_hst_index==k])
            y_cv.append(self.Y[cv_hst_index==k])
            z_cv.append(self.Z[cv_evl_index==k])
            p_evl_hst_cv.append(self.pi_evaluation_train[cv_hst_index==k])
            p_evl_evl_cv.append(self.pi_evaluation_test[cv_evl_index==k])

        for k in range(folds):
            #print(h0_cv[0])
            # calculate the h vectors for training and test
            count = 0
            for j in range(folds):
                if j != k:
                    if count == 0:
                        x_tr = x_cv[j]
                        a_tr = a_cv[j]
                        y_tr = y_cv[j]
                        z_tr = z_cv[j]
                        p_evl_hst_tr = p_evl_hst_cv[j]
                        p_evl_evl_tr = p_evl_evl_cv[j]
                        count += 1
                    else:
                        x_tr = np.append(x_tr, x_cv[j], axis=0)
                        a_tr = np.append(a_tr, a_cv[j], axis=0)
                        y_tr = np.append(y_tr, y_cv[j], axis=0)
                        z_tr = np.append(z_tr, z_cv[j], axis=0)
                        p_evl_hst_tr = np.append(p_evl_hst_tr, p_evl_hst_cv[j], axis=0)
                        p_evl_evl_tr = np.append(p_evl_evl_tr, p_evl_evl_cv[j], axis=0)
                        
            f_hst_matrix = np.zeros(shape=(len(x_tr), len(self.classes)))
            f_evl_matrix = np.zeros(shape=(len(z_tr), len(self.classes)))
            #w_matrix = np.zeros(shape=(len(x_tr), len(self.classes)))

            _, a_temp = np.where(a_tr == 1)
            clf, x_ker_train, x_ker_test = KernelRegression(x_tr, a_temp, z_tr, algorithm=method, logit=True)
            clf.fit(x_ker_train, a_temp)
            p_bhv = clf.predict_proba(x_ker_train)

            w_matrix = p_evl_hst_tr/p_bhv
            
            for c in self.classes:
                clf, x_ker_train, x_ker_test = KernelRegression(x_tr, y_tr[:, c], z_tr, algorithm=method, logit=False)
                clf.fit(x_ker_train, y_tr[:, c])
                f_hst_matrix[:, c] = clf.predict(x_ker_train)
                f_evl_matrix[:, c] = clf.predict(x_ker_test)
            
            densratio_obj = densratio(z_tr, x_tr)
            r = densratio_obj.compute_density_ratio(x_tr)
            r = np.array([r for c in self.classes]).T

            if self_norm:
                denominator = np.sum(r*p_evl_hst_tr/p_bhv)
            else:
                denominator = self.N_hst
            
            theta = np.sum(a_tr*(y_tr-f_hst_matrix)*w_matrix*r/denominator) + np.sum(f_evl_matrix*p_evl_evl_tr)/self.N_evl
            theta_list.append(theta)
        
        return np.mean(theta_list)

    def denominator(self, self_norm):
        """
        if self_norm=True return np.sum(self.r_hat_kernel*(self.pi_evaluation_train/self.bpol_hat_kernel))
        else: return self.N_hst
        """
        if self_norm:
            if self.bpol_hat_kernel is None:
                pi_behavior = np.zeros(shape=(self.N_hst, len(self.classes)))
                for c in self.classes:
                    perm = np.random.permutation(self.N_hst)
                    A_temp = self.A[perm[:50]]
                    X_temp = self.X[perm[:50]]
                    model = KernelReg(A_temp[:,c], X_temp, var_type='c'*self.dim, reg_type='lc')
                    model = KernelReg(self.A[:,c], self.X, var_type='c'*self.dim, reg_type='lc', bw=model.bw)
                    #model = KernelReg(self.A[:,c], self.X, var_type='c'*self.dim, reg_type='lc')
                    mu, _ = model.fit(self.X)
                    pi_behavior[:, c] = mu
                    
                self.bpol_hat_kernel = pi_behavior

            return np.sum(self.r_hat_kernel*(self.pi_evaluation_train/self.bpol_hat_kernel))

        else:
            return self.N_hst

    def ipw_ML(self, method='Ridge', self_norm=True):
        if self.bpol_ML_hat_kernel is None:
            _, a_temp = np.where(self.A == 1)
            clf, x_ker_train, x_ker_test = KernelRegression(self.X, a_temp, self.X, algorithm=method, logit=True)
            clf.fit(x_ker_train, a_temp)

            self.bpol_ML_hat_kernel = clf.predict_proba(x_ker_train)
        
        if self.r_ML_hat_kernel is None:
            densratio_obj = densratio(self.Z, self.X)
            r = densratio_obj.compute_density_ratio(self.X)
            r = np.array([r for c in self.classes]).T
            self.r_ML_hat_kernel = r

        w = self.pi_evaluation_train/self.bpol_ML_hat_kernel
        
        denomnator = self.denominator_ML(self_norm)
        
        return np.sum(self.A*self.Y*w*self.r_ML_hat_kernel/denomnator)

    def dm_ML(self, method='Ridge'):        
        f_matrix = np.zeros(shape=(self.N_evl, len(self.classes)))

        if self.f_ML_hat_kernel is None:
            for c in self.classes:
                clf, x_ker_train, x_ker_test = KernelRegression(self.X, self.Y[:, c], self.Z, algorithm=method, logit=False)
                clf.fit(x_ker_train, self.Y[:, c])
                f_matrix[:, c] = clf.predict(x_ker_test)
            
            self.f_ML_hat_kernel = f_matrix
            
        return np.sum(self.f_ML_hat_kernel*self.pi_evaluation_test)/self.N_evl
    
    def denominator_ML(self, self_norm):
        if self_norm:
            if self.bpol_ML_hat_kernel is None:
                _, a_temp = np.where(self.A == 1)
                clf, x_ker_train, x_ker_test = KernelRegression(self.X, a_temp, self.X, algorithm=method, logit=True)
                clf.fit(x_ker_train, a_temp)

                self.bpol_ML_hat_kernel = clf.predict_proba(x_ker_train)
                    
            return np.sum(self.r_ML_hat_kernel*(self.pi_evaluation_train/self.bpol_ML_hat_kernel))

        else:
            return self.N_hst
        
        
        
    def lcb(self, self_norm=True, folds=2, method='Lasso'):
        """
        Hàm này thực hiện phương pháp doubly robust estimator 
        e_fold=fold
        estimate f(a,x) and w(a,x) using method=method   n
        """
        theta_list = []
        
        cv_hst_fold = np.arange(folds) 
        cv_hst_split0 = np.floor(np.arange(self.N_hst)*folds/self.N_hst)
        cv_hst_index = cv_hst_split0[np.random.permutation(self.N_hst)]
        
        cv_evl_split0 = np.floor(np.arange(self.N_evl)*folds/self.N_evl)
        cv_evl_index = cv_evl_split0[np.random.permutation(self.N_evl)]

        x_cv = []
        a_cv = []
        y_cv = []
        z_cv = []
        p_evl_hst_cv = []
        p_evl_evl_cv = []
        
        for k in cv_hst_fold:
            x_cv.append(self.X[cv_hst_index==k])
            a_cv.append(self.A[cv_hst_index==k])
            y_cv.append(self.Y[cv_hst_index==k])
            z_cv.append(self.Z[cv_evl_index==k])
            p_evl_hst_cv.append(self.pi_evaluation_train[cv_hst_index==k])
            p_evl_evl_cv.append(self.pi_evaluation_test[cv_evl_index==k])

        for k in range(folds):
            #print(h0_cv[0])
            # calculate the h vectors for training and test
            count = 0
            for j in range(folds):
                if j != k:
                    if count == 0:
                        x_tr = x_cv[j]
                        a_tr = a_cv[j]
                        y_tr = y_cv[j]
                        z_tr = z_cv[j]
                        p_evl_hst_tr = p_evl_hst_cv[j]
                        p_evl_evl_tr = p_evl_evl_cv[j]
                        count += 1
                    else:
                        x_tr = np.append(x_tr, x_cv[j], axis=0)
                        a_tr = np.append(a_tr, a_cv[j], axis=0)
                        y_tr = np.append(y_tr, y_cv[j], axis=0)
                        z_tr = np.append(z_tr, z_cv[j], axis=0)
                        p_evl_hst_tr = np.append(p_evl_hst_tr, p_evl_hst_cv[j], axis=0)
                        p_evl_evl_tr = np.append(p_evl_evl_tr, p_evl_evl_cv[j], axis=0)
                        
            f_hst_matrix = np.zeros(shape=(len(x_tr), len(self.classes)))
            f_evl_matrix = np.zeros(shape=(len(z_tr), len(self.classes)))
            #w_matrix = np.zeros(shape=(len(x_tr), len(self.classes)))

            _, a_temp = np.where(a_tr == 1)
            clf, x_ker_train, x_ker_test = KernelRegression(x_tr, a_temp, z_tr, algorithm=method, logit=True)
            clf.fit(x_ker_train, a_temp)
            p_bhv = clf.predict_proba(x_ker_train)

            w_matrix = p_evl_hst_tr/p_bhv
            f_hst_matrix=self.model.pre
            for c in self.classes:
                clf, x_ker_train, x_ker_test = KernelRegression(x_tr, y_tr[:, c], z_tr, algorithm=method, logit=False)
                clf.fit(x_ker_train, y_tr[:, c])
                f_hst_matrix[:, c] = clf.predict(x_ker_train)
                f_evl_matrix[:, c] = clf.predict(x_ker_test)
            
            densratio_obj = densratio(z_tr, x_tr)
            r = densratio_obj.compute_density_ratio(x_tr)
            r = np.array([r for c in self.classes]).T

            if self_norm:
                denominator = np.sum(r*p_evl_hst_tr/p_bhv)
            else:
                denominator = self.N_hst
            
            theta = np.sum(a_tr*(y_tr-f_hst_matrix)*w_matrix*r/denominator) + np.sum(f_evl_matrix*p_evl_evl_tr)/self.N_evl
            theta_list.append(theta)
        
        return np.mean(theta_list)