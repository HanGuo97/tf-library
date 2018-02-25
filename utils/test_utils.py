import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error




def random_integers(low, high, shape, dtype=tf.int32):
    randint = np.random.randint(low=low, high=high, size=shape)
    randint = tf.convert_to_tensor(randint, dtype=dtype)
    return randint
    
    
def random_tensor(shape, *args, **kargs):
    return tf.random_normal(shape=shape, *args, **kargs)


def random_vector(shape, *args, **kargs):
    return np.random.normal(size=shape, *args, **kargs)


def check_equality(A, B, descip="", tolerance=1e-7):
    if isinstance(A, (list, tuple, np.ndarray)):
        diff = L2_distance(A, B)
    else:
        diff = np.square(A - B)
    print("SSE: %.11f\t%s: %r" % (diff, descip, diff < tolerance))


def check_inequality(A, B, descip="", tolerance=1e-3):
    if isinstance(A, (list, tuple, np.ndarray)):
        diff = L2_distance(A, B)
    else:
        diff = np.square(A - B)
    print("SSE: %.11f\t%s: %r" % (diff, descip, diff > tolerance))


def L2_distance(X, Y):
    # MSE = doubled_l2 / batch_size
    # L2 = doubled_l2 / 2
    doubled_l2 = np.sum(np.square(X - Y))
    # follow TF's implementation
    return doubled_l2 / 2


def test_message(msg):
    print("\n\n\nTESTING:\t" + msg + "\n\n\n")


################################################
# Simulated Environments
################################################
class Point:
    def __init__(self, params, params_train, params_val):
        self.params = params
        self.params_train = params_train
        self.params_val = params_val

    def train_loss(self):
        # Sum(Xi - Xi*)^2
        return L2_distance(self.params, self.params_train)
    
    def regularization_loss(self, P2, coefs):
        # alpha x Sum(Xi - Zi)^2
        assert len(self.params.shape) == len(coefs.shape)
        assert len(self.params.shape) == len(P2.params.shape)
        reg_loss = coefs * np.square(self.params - P2.params)
        return np.sum(reg_loss)

    def losses(self, P2, coefs):
        l2_loss = self.train_loss()
        reg_loss = self.regularization_loss(P2=P2, coefs=coefs)
        val_loss = self.validation_loss()
        return l2_loss + reg_loss, val_loss

    def validation_loss(self):
        # Sum(Xi - Xi*)^2
        return L2_distance(self.params, self.params_val)

    def gradient(self, P2, coefs):
        # grad = 2(X - X*) + 2 alpha x (X - _X)
        assert len(self.params.shape) == len(coefs.shape)
        assert len(self.params.shape) == len(P2.params.shape)
        grads = (2 * (self.params - self.params_train) +  # train loss
                 2 * coefs * (self.params - P2.params))  # reg loss
        return grads
    
    def gradient_descent(self, P2, coefs, lr):
        grads = self.gradient(P2=P2, coefs=coefs)
        self.params = self.params - lr * grads
        

class Environment:
    def __init__(self, P1, P2, lr):
        self.P1 = copy.deepcopy(P1)
        self.P2 = copy.deepcopy(P2)
        self.lr = lr
        
        self.best_P1_valid_loss = np.inf
        self.best_P2_valid_loss = np.inf
        self.P1_history = [P1.params]
        self.P2_history = [P2.params]
    
    def state(self):
        return [self.P1.params, self.P2.params,
                self.P1.params - self.P2.params]
    
    def update_P1(self, coefs):
        self.P1.gradient_descent(P2=self.P2, coefs=coefs, lr=self.lr)
        P1_train_loss, P1_val_loss = self.P1.losses(P2=self.P2, coefs=coefs)
        
        self.log_P1()
        self.log_P1_val(P1_val_loss)
        return P1_train_loss, P1_val_loss, self.state()
    
    def update_P2(self, coefs):
        self.P2.gradient_descent(P2=self.P1, coefs=coefs, lr=self.lr)
        P2_train_loss, P2_val_loss = self.P2.losses(P2=self.P1, coefs=coefs)
        
        self.log_P2()
        self.log_P2_val(P2_val_loss)
        return P2_train_loss, P2_val_loss, self.state()
    
    def log_P1(self):
        self.P1_history.append(self.P1.params)
    
    def log_P2(self):
        self.P2_history.append(self.P2.params)
        
    def log_P1_val(self, P1_val_loss):
        if P1_val_loss < self.best_P1_valid_loss:
            self.best_P1_valid_loss = P1_val_loss
    
    def log_P2_val(self, P2_val_loss):
        if P2_val_loss < self.best_P2_valid_loss:
            self.best_P2_valid_loss = P2_val_loss

    def visualize(self, figidx=1):
        stacked_P1_history = np.stack(self.P1_history)
        stacked_P2_history = np.stack(self.P2_history)
        pca = PCA(n_components=2).fit(np.concatenate(
            [stacked_P1_history, stacked_P2_history], axis=0))
        proj_P1_history = pca.transform(stacked_P1_history)
        proj_P2_history = pca.transform(stacked_P2_history)
        proj_train_val_params = pca.transform(np.stack(
            [self.P1.params_train, self.P1.params_val,
             self.P2.params_train, self.P2.params_val], axis=0))

        plt.figure(figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
        plt.scatter(proj_P1_history[0, 0], proj_P1_history[0, 1],
            marker='o', c="black", s=300, cmap=plt.get_cmap('Spectral'))
        plt.scatter(proj_P2_history[0, 0], proj_P2_history[0, 1],
            marker='o', c="black", s=300, cmap=plt.get_cmap('Spectral'))

        plt.scatter(proj_train_val_params[0, 0], proj_train_val_params[0, 1],
            marker='o', c="salmon", s=300, cmap=plt.get_cmap('Spectral'))
        plt.scatter(proj_train_val_params[1, 0], proj_train_val_params[1, 1],
            marker='o', c="grey", s=300, cmap=plt.get_cmap('Spectral'))

        plt.scatter(proj_train_val_params[2, 0], proj_train_val_params[2, 1],
            marker='o', c="salmon", s=300, cmap=plt.get_cmap('Spectral'))
        plt.scatter(proj_train_val_params[3, 0], proj_train_val_params[3, 1],
            marker='o', c="grey", s=300, cmap=plt.get_cmap('Spectral'))
        
        
        all_X1 = [proj_P1_history[0, 0],
                  proj_P2_history[0, 0],
                  proj_train_val_params[0, 0],  # P1.X1_train,
                  proj_train_val_params[1, 0],  # P1.X1_val,
                  proj_train_val_params[2, 0],  # P2.X1_train,
                  proj_train_val_params[3, 0]]  # P2.X1_val,
        
        all_X2 = [proj_P1_history[0, 1],
                  proj_P2_history[0, 1],
                  proj_train_val_params[0, 1],  # P1.X1_train,
                  proj_train_val_params[1, 1],  # P1.X1_val,
                  proj_train_val_params[2, 1],  # P2.X1_train,
                  proj_train_val_params[3, 1]]  # P2.X1_val,
        
        labels = ["P1", "P2", "P1-Train", "P1-Val", "P2-Train", "P2-Val"]
        for label, x, y in zip(labels, all_X1, all_X2):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-5, 5),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.scatter(proj_P1_history[:, 0], proj_P1_history[:, 1], c="grey", s=5, alpha=0.5)
        plt.scatter(proj_P2_history[:, 0], proj_P2_history[:, 1], c="grey", s=5, alpha=0.5)
        
        # plt.xlim(0, 110)
        # plt.ylim(0, 110)
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()
        self.summarize()
        
    def summarize(self):
        print("Best P1 Val %.2f \tBest P2 Val %.2f" % (self.best_P1_valid_loss, self.best_P2_valid_loss))
