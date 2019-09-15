import logging
import random
import numpy as np
from cvxopt import matrix, solvers
import copy
import time
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
#from matplotlib import style
#style.use('ggplot')

class BinarySVM:
    def __init__(self, kernel='linear', alg='SMO', gamma=None, degree=None, C=1.0, verbose=False, iterations = 5000000):
        self.eps = 1e-6
        self.alg = alg
        self.C, self.w, self.b, self.ksi = C, [], 0.0, []
        self.n_sv = -1
        self.sv_x, self.sv_y, self.alphas = np.zeros(0), np.zeros(0), np.zeros(0)
        #self.visualization = visualization
        #self.colors = {1:'r',-1:'b'}
        '''if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)'''
            
        self.kernel = kernel
        self.iterations = iterations
        if self.kernel == 'poly':
            self.degree = degree if degree is not None else 2.0
        elif self.kernel == 'rbf':
            self.gamma = gamma
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def _kernel(self, x, z=None):
        if z is None:
            z = x
        if self.kernel == 'linear':
            return np.dot(x, z.T)
        elif self.kernel == 'poly':
            return (np.dot(x, z.T) + 1.0) ** self.degree
        elif self.kernel == 'rbf':
            xx, zz = np.sum(x*x, axis=1), np.sum(z*z, axis=1)
            res = -2.0 * np.dot(x, z.T) + xx.reshape((-1, 1)) + zz.reshape((1, -1))
            # del xx, zz
            return np.exp(-self.gamma * res)
        # elif self.kernel == 'sigmoid':
        #     pass
        else:
            print('Unknown kernel')
            exit(3)

    def _QP(self, x, y):
        # In QP formulation (dual): m variables, 2m+1 constraints (1 equation, 2m inequations)
        m = len(y)
        print(x.shape, y.shape)
        P = self._kernel(x) * np.outer(y, y)
        P, q = matrix(P, tc='d'), matrix(-np.ones((m, 1)), tc='d')
        G = matrix(np.r_[-np.eye(m), np.eye(m)], tc='d')
        h = matrix(np.r_[np.zeros((m,1)), np.zeros((m,1)) + self.C], tc='d')
        A, b = matrix(y.reshape((1,-1)), tc='d'), matrix([0.0])
        # print "P, q:"
        # print P, q
        # print "G, h"
        # print G, h
        # print "A, b"
        # print A, b
        solution = solvers.qp(P, q, G, h, A, b)
        if solution['status'] == 'unknown':
            print('Not PSD!')
            exit(2)
        else:
            self.alphas = np.array(solution['x']).squeeze()

    def _SMO5(self, K, y):
        def choose_alphas():  # choose alpha heuristically
            check_all_examples = False  # whether or not need to check all examples
            # passes, max_passes = 0, 2
            # while passes < max_passes:
            while True:
                num_changed_alphas = 0
                if check_all_examples:
                    # in range_i, unbounded alphas rank first
                    # range_i = np.argsort((self.alphas - self.eps) * (self.alphas + self.eps - self.C))
                    range_i = range(m)
                else:
                    # check unbounded examples only
                    range_i = [i for i in range(m) if self.eps < self.alphas[i] < self.C - self.eps]
                # print 'range_i:', range_i
                for i in range_i:
                    # print 'Try i:', i
                    yi, ai_old = y[i], self.alphas[i]
                    Ei = np.sum(self.alphas * y * K[i]) + self.b - yi
                    logging.debug('Ei = ' + str(Ei))
                    if (yi*Ei < -tol and ai_old < self.C) or (yi*Ei > tol and ai_old > 0):
                        # print i, 'is against KKT conditions'
                        range_j = list(range(m))
                        random.shuffle(range_j)
                        # print 'range j:', range_j
                        # for j in xrange(m):
                        for j in range_j:
                            if j == i:
                                continue
                            yj = y[j]
                            Ej = np.sum(self.alphas * y * K[j]) + self.b - yj
                            # if abs(Ej - Ei) <= self.eps:
                            #     continue
                            yield (i, j, Ei, Ej)
                            if updated[0]:  # if (i, j) pair changed in the latest iteration
                                num_changed_alphas += 1
                                break
                if num_changed_alphas == 0:
                    if check_all_examples:  # if have checked all examples and no alpha violates KKT condition
                        break
                    else:
                        check_all_examples = True  # check all examples in the next iteration as a safeguard
                else:
                    check_all_examples = False
            yield -1, -1, 0.0, 0.0

        print('Begin SMO5...')
        m = len(y)
        self.alphas, self.b = np.zeros(m), 0.0
        tol = self.eps
        logging.debug('m = ' + str(m))
        gen = choose_alphas()
        n_iter = 0
        iteration = 0
        updated = [False]  # use mutable object to pass message between functions
        pbar = tqdm(total=self.iterations)
        while iteration < self.iterations:
            pbar.update(1)
            n_iter += 1
            iteration += 1
            logging.info('Iteration ' + str(n_iter))
            # logging.debug('passes = ' + str(passes))
            # run over pair (i, j).  But for some alpha_i, only choose one alpha_j in an iteration epoch.
            try:
                # gen.send(updated)
                i, j, Ei, Ej = next(gen)
            except StopIteration as e:
                break
            if i == -1:  # no more (i, j) pairs against KKT condition
                break
            updated[0] = False
            yi, yj, ai_old, aj_old = y[i], y[j], self.alphas[i], self.alphas[j]
            if yi != yj:
                L, H = max(0.0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
            else:
                L, H = max(0.0, ai_old + aj_old - self.C), min(self.C, ai_old + aj_old)
            logging.debug('L = ' + str(L) + ', H = ' + str(H))
            if H - L < self.eps:
                continue
            eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
            logging.debug('eta = ' + str(eta))
            if eta <= 0:  # This should not be happen, because gram matrix should be PSD
                if eta == 0.0:
                    logging.warning('eta = 0, possibly identical examples encountered!')
                else:
                    logging.error('GRAM MATRIX IS NOT PSD!')
                #input('*****************')
                continue
            aj_new = aj_old + yj * (Ei - Ej) / eta
            logging.debug('i, j, yi, yj = ' + ' '.join(map(str, [i, j, yi, yj])))
            logging.debug('Ei = ' + str(Ei) + ', Ej = ' + str(Ej))
            logging.debug('aj_new = ' + str(aj_new))
            if aj_new > H:
                aj_new = H
            elif aj_new < L:
                aj_new = L
            delta_j = aj_new - aj_old
            if abs(delta_j) < 1e-5:
                # print "j = %d, is not moving enough" % j
                continue
            ai_new = ai_old + yi * yj * (aj_old - aj_new)
            # ai_new = ai_old - yi * yj * delta_j
            delta_i = ai_new - ai_old
            self.alphas[i], self.alphas[j] = ai_new, aj_new
            bi = self.b - Ei - yi * delta_i * K[i, i] - yj * delta_j * K[i, j]
            bj = self.b - Ej - yi * delta_i * K[i, j] - yj * delta_j * K[j, j]
            if 0 < ai_new < self.C:
                self.b = bi
            elif 0 < aj_new < self.C:
                self.b = bj
            else:
                self.b = (bi + bj) / 2.0
            updated[0] = True
            logging.info('Updated through' + str(i) + str(j))
            logging.debug('alphas:' + str(self.alphas))

        print('Finish SMO5...')
        return self.alphas, self.b

    def fit(self, x, y):
        # x, y: np.ndarray
        # x.shape: (m, n), where m = # samples, n = # features.
        # y.shape: (m,), m labels which range from {-1.0, +1.0}.
        assert type(x) == np.ndarray
        # print type(x), type(y)
        print(x.shape, y.shape)
        # In the design matrix x: m examples, n features
        # In QP formulation (dual): m variables, 2m+1 constraints (1 equation, 2m inequations)
        # print 'x = ', x
        # print 'y = ', y
        if self.kernel == 'rbf' and self.gamma is None:
            self.gamma = 1.0 / x.shape[1]
            print('gamma = ', self.gamma)
        if self.alg == 'dual':
            self._QP(x, y)
        else:
            assert self.alg == 'SMO'
            K = self._kernel(x)
            #print(type(K))
            self._SMO5(K, y)

        logging.info('self.alphas = ' + str(self.alphas))
        sv_indices = [i for i in range(len(y)) if self.alphas[i] > self.eps]
        self.sv_x, self.sv_y, self.alphas = x[sv_indices], y[sv_indices], self.alphas[sv_indices]
        self.n_sv = len(sv_indices)
        logging.info('sv_indices:' + str(sv_indices))
        print(self.n_sv, 'SVs!')
        logging.info(str(np.c_[self.sv_x, self.sv_y]))
        if self.kernel == 'linear':
            self.w = np.dot(self.alphas * self.sv_y, self.sv_x)
        if self.alg == 'dual':
            # calculate b: average over all support vectors
            sv_boundary = self.alphas < self.C - self.eps
            self.b = np.mean(self.sv_y[sv_boundary] - np.dot(self.alphas * self.sv_y,
                                                             self._kernel(self.sv_x, self.sv_x[sv_boundary])))

    def predict_score(self, x):
        return np.dot(self.alphas * self.sv_y, self._kernel(self.sv_x, x)) + self.b

    '''def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,c=self.colors[i]) for x in data_dict[i]] for i in data_dict]
        
        # hyperplane = x.w+b (actually its a line)
        # v = x0.w0+x1.w1+b -> x1 = (v-w[0].x[0]-b)/w1
        #psv = 1     psv line ->  x.w+b = 1a small value of b we will increase it later
        #nsv = -1    nsv line ->  x.w+b = -1
        # dec = 0    db line  ->  x.w+b = 0
        def hyperplane(x,w,b,v):
            #returns a x2 value on line when given x1
            return (-w[0]*x-b+v)/w[1]
       
        hyp_x_min= self.min_feature_value*0.9
        hyp_x_max = self.max_feature_value*1.1
        
        # (w.x+b)=1
        # positive support vector hyperplane
        pav1 = hyperplane(hyp_x_min,self.w,self.b,1)
        pav2 = hyperplane(hyp_x_max,self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[pav1,pav2],'k')
        
        # (w.x+b)=-1
        # negative support vector hyperplane
        nav1 = hyperplane(hyp_x_min,self.w,self.b,-1)
        nav2 = hyperplane(hyp_x_max,self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nav1,nav2],'k')
        
        # (w.x+b)=0
        # db support vector hyperplane
        db1 = hyperplane(hyp_x_min,self.w,self.b,0)
        db2 = hyperplane(hyp_x_max,self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y--')'''

    def show(self):
        if (self.alg == 'dual') or (self.alg == 'SMO'):
            print('\nFitted parameters:')
            print('n_sv = ', self.n_sv)
            print('sv_x = ', self.sv_x)
            print('sv_y = ', self.sv_y)
            print('alphas = ', self.alphas)
            if self.kernel == 'linear':
                print('w = ', self.w)
            print('b = ', self.b)
        else:
            print('No known optimization method!')

    def predict(self, x):
        return np.sign(self.predict_score(x))

    def save(self, file_name='BinarySVM1.pkl'):
        fh = open('model/' + file_name, 'wb')
        pickle.dump(self, fh)
        fh.close()


class MultiSVM:
    def __init__(self, kernel='linear', alg='SMO', decision_function='ovr', gamma=None, degree=None, C=1.0):
        self.degree, self.gamma, self.decision_function = degree, gamma, decision_function
        self.alg, self.C = alg, C
        self.kernel = kernel
        self.n_class, self.classifiers = 0, []

    def fit(self, x, y):
        labels = np.unique(y)
        self.n_class = len(labels)
        print(labels)
        if self.decision_function == 'ovr':  # one-vs-rest method
            for label in labels:
                y1 = np.array(y)
                y1[y1 != label] = -1.0
                y1[y1 == label] = 1.0
                print('Begin training for label', label, 'at', \
                    time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time())))
                t1 = time.time()
                clf = BinarySVM(self.kernel, self.alg, self.gamma, self.degree, self.C)
                clf.fit(x, y1)
                t2 = time.time()
                print('Training time for ' + str(label) + '-vs-rest:', t2 - t1, 'seconds')
                self.classifiers.append(copy.deepcopy(clf))
        else:  # use one-vs-one method
            assert self.decision_function == 'ovo'
            n_labels = len(labels)
            for i in range(n_labels):
                for j in range(i+1, n_labels):
                    neg_id, pos_id = y == labels[i], y == labels[j]
                    x1, y1 = np.r_[x[neg_id], x[pos_id]], np.r_[y[neg_id], y[pos_id]]
                    y1[y1 == labels[i]] = -1.0
                    y1[y1 == labels[j]] = 1.0
                    # print 'y1 = ', y1
                    print('Begin training classifier for label', labels[i], 'and label', labels[j], 'at', \
                        time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time())))
                    t1 = time.time()
                    clf = BinarySVM(self.kernel, self.alg, self.gamma, self.degree, self.C)
                    clf.fit(x1, y1)
                    t2 = time.time()
                    print('Training time for ' + str(labels[i]) + '-vs-' + str(labels[j]) + ':', t2 - t1, 'seconds')
                    self.classifiers.append(copy.deepcopy(clf))

    def predict(self, test_data):
        n_samples = test_data.shape[0]
        if self.decision_function == 'ovr':
            score = np.zeros((n_samples, self.n_class))
            for i in range(self.n_class):
                clf = self.classifiers[i]
                score[:, i] = clf.predict_score(test_data)
            return np.argmax(score, axis=1)
        else:
            assert self.decision_function == 'ovo'
            assert len(self.classifiers) == self.n_class * (self.n_class - 1) / 2
            vote = np.zeros((n_samples, self.n_class))
            clf_id = 0
            for i in range(self.n_class):
                for j in range(i+1, self.n_class):
                    res = self.classifiers[clf_id].predict(test_data)
                    vote[res < 0, i] += 1.0  # negative sample: class i
                    vote[res > 0, j] += 1.0  # positive sample: class j
                    # print i, j
                    # print 'res = ', res
                    # print 'vote = ', vote
                    clf_id += 1
            return np.argmax(vote, axis=1)

    def save(self, file_name='MultiSVM1.pkl'):
        fh = open('model/' + file_name, 'wb')
        pickle.dump(self, fh)
        fh.close()

    def show(self):
        for clf in self.classifiers:
            clf.show()

