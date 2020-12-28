'''
-----------------------Authors-------------------------
              Nayanika Ghosh - 4191 4976
                Disha Nayar - 6199 1035
-------------------------------------------------------
'''
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


class Classifier():
    '''
    This class implements 3 functions for classifiers
    Each function has one classifier model defined.
    '''
    def __init__(self):
        '''
        Load the tiktac_final and tiktak_single data files
        Call each of the models with the two datasets
        '''
        self.final_path = "datasets-part1/tictac_final.txt"
        self.single_path = 'datasets-part1/tictac_single.txt'
        self.data_final = np.loadtxt(self.final_path)
        self.data_single = np.loadtxt(self.single_path)
        # randomly shuffle the data
        self.data_final = shuffle(self.data_final)
        self.data_single = shuffle(self.data_single)
        # call functions for Final dataset
        self.linear_SVM(self.data_final, 2, "Final boards classification dataset")
        self.kNearest(self.data_final, 2, "Final boards classification dataset")
        self.MLP(self.data_final, 2, "Final boards classification dataset")
        # call functions for Intermediate - single label
        self.linear_SVM(self.data_single, 9, "Intermediate boards optimal play (single label)")
        self.kNearest(self.data_single, 9, "Intermediate boards optimal play (single label)")
        self.MLP(self.data_single, 9, "Intermediate boards optimal play (single label)")

    def linear_SVM(self, data, num_labels, str):
        '''
        This function implements linear svm
        It applies k-fold cross validation with k = 10
        :param data: input dataset
        :param num_labels: number of values that output label can take
        :param str: name of the dataset, used in printing
        :return: None
        '''
        features = data[:,:9]
        labels = data[:,9:]
        kf = KFold(n_splits=10)
        kf.get_n_splits(features)
        total_accuracy = 0.0
        con_matrix = np.zeros((num_labels, num_labels))
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            svm_model = SVC(kernel='linear')
            svm_model.fit(X_train, y_train.ravel())
            y_pred = svm_model.predict(X_test)
            acc = metrics.accuracy_score(y_pred, y_test)
            total_accuracy += acc
            con_matrix += confusion_matrix(y_test, y_pred)
        cm_normalized = con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis]
        print('Accuracy of Linear SVM on ', str, ' is ',  total_accuracy/10)
        np.set_printoptions(precision=2)
        print('Confusion Matrix of Linear SVM  on ', str, ' is \n',  cm_normalized, '\n\n')

    def kNearest(self, data, num_labels, str):
        '''
        This function implements K nearest neighbours model
        It applies k-fold cross validation with k = 10
        number of neighbors = 9
        :param data: input dataset
        :param num_labels: number of values that output label can take
        :param str: name of the dataset, used in printing
        :return: None
        '''
        features = data[:, :9]
        labels = data[:, 9:]
        kf = KFold(n_splits=10)
        kf.get_n_splits(features)
        total_accuracy = 0.0
        con_matrix = np.zeros((num_labels, num_labels))
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            knn_model = KNeighborsClassifier(n_neighbors=9)
            knn_model.fit(X_train, y_train.ravel())
            y_pred = knn_model.predict(X_test)
            acc = metrics.accuracy_score(y_pred, y_test)
            total_accuracy += acc
            con_matrix += confusion_matrix(y_test, y_pred)
        cm_normalized = con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis]
        print('Accuracy of k-Nearest Classifier on ', str, ' is ', total_accuracy / 10)
        np.set_printoptions(precision=2)
        print('Confusion Matrix of k-Nearest Classifier on ', str, ' is \n', cm_normalized, '\n\n')

    def MLP(self, data, num_labels, str):
        '''
        This function implements the MLP model
        It applies k-fold cross validation with k = 10
        :param data: input dataset
        :param num_labels: number of values that output label can take
        :param str: name of the dataset, used in printing
        :return: None
        '''
        features = data[:, :9]
        labels = data[:, 9:]
        kf = KFold(n_splits=10)
        kf.get_n_splits(features)
        total_accuracy = 0.0
        con_matrix = np.zeros((num_labels, num_labels))
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            mlp_model = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(200,100,40),
                                      random_state=1, max_iter=500).fit(X_train, y_train.ravel())
            y_pred = mlp_model.predict(X_test)
            acc3 = metrics.accuracy_score(y_pred, y_test)
            total_accuracy += acc3
            con_matrix += confusion_matrix(y_test, y_pred)
        cm_normalized = con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis]
        print('Accuracy of MLP Classifier on ', str, ' is ', total_accuracy / 10)
        np.set_printoptions(precision=2)
        print('Confusion Matrix of MLP Classifier on ', str, ' is \n', cm_normalized, '\n\n')


class Regressor():
    '''
    This class implements 3 functions for regressors
    Each function has one regressor model defined.
    '''
    def __init__(self):
        '''
        Load the tiktac_multi data files
        Call each of the models with the two datasets
        '''
        self.multi_path = 'datasets-part1/tictac_multi.txt'
        self.data_multi = np.loadtxt(self.multi_path)
        # randomly shuffle the data
        self.data_multi = shuffle(self.data_multi)
        # call functions for Intermediate - multi label
        self.linear_regressor()
        self.kNearest()
        self.MLP()

    def linear_regressor(self):
        '''
        This function implements a linear regressor model
        The weigths are calculated using normal equation
        w0 to w8 define the weigths
        It applies k-fold cross validation with k = 10
        :return: None
        '''
        X = self.data_multi[:, :9]
        y = self.data_multi[:, 9:]
        y0 = y[:, :1]
        y1 = y[:, 1:2]
        y2 = y[:, 2:3]
        y3 = y[:, 3:4]
        y4 = y[:, 4:5]
        y5 = y[:, 5:6]
        y6 = y[:, 6:7]
        y7 = y[:, 7:8]
        y8 = y[:, 8:]
        x_shape = X.shape
        x_bias = np.ones((x_shape[0], 1))
        features = np.append(x_bias, X, axis=1)
        kf = KFold(n_splits=10)
        total_accuracy = 0.0
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            # Calculate (X^TX)^-1X^T
            M1 = np.dot(np.transpose(X_train), X_train)
            M2 = np.dot(np.linalg.inv(M1), np.transpose(X_train))
            # train on y0
            y0_train, y0_test = y0[train_index], y0[test_index]
            w0 = np.dot(M2, y0_train)
            y0_pred = np.dot(X_test, w0)
            # train on y1
            y1_train, y1_test = y1[train_index], y1[test_index]
            w1 = np.dot(M2, y1_train)
            y1_pred = np.dot(X_test, w1)
            # train on y2
            y2_train, y2_test = y2[train_index], y2[test_index]
            w2 = np.dot(M2, y2_train)
            y2_pred = np.dot(X_test, w2)
            # train on y3
            y3_train, y3_test = y3[train_index], y3[test_index]
            w3 = np.dot(M2, y3_train)
            y3_pred = np.dot(X_test, w3)
            # train on y4
            y4_train, y4_test = y4[train_index], y4[test_index]
            w4 = np.dot(M2, y4_train)
            y4_pred = np.dot(X_test, w4)
            # train on y5
            y5_train, y5_test = y5[train_index], y5[test_index]
            w5 = np.dot(M2, y5_train)
            y5_pred = np.dot(X_test, w5)
            # train on y6
            y6_train, y6_test = y6[train_index], y6[test_index]
            w6 = np.dot(M2, y6_train)
            y6_pred = np.dot(X_test, w6)
            # train on y7
            y7_train, y7_test = y7[train_index], y7[test_index]
            w7 = np.dot(M2, y7_train)
            y7_pred = np.dot(X_test, w7)
            # train on y8
            y8_train, y8_test = y8[train_index], y8[test_index]
            w8 = np.dot(M2, y8_train)
            y8_pred = np.dot(X_test, w8)
            # stack all the vectors together to get y_predicted
            y_pred = np.column_stack((y0_pred, y1_pred, y2_pred, y3_pred, y4_pred,
                                      y5_pred, y6_pred, y7_pred, y8_pred))
            y_test = np.column_stack((y0_test, y1_test, y2_test, y3_test, y4_test,
                                      y5_test, y6_test, y7_test, y8_test))
            b = np.zeros_like(y_pred)
            b[np.arange(len(y_pred)), y_pred.argmax(1)] = 1
            y_pred_updated = b
            acc = 0.0
            for i in range(len(y_pred_updated)):
                acc = acc + metrics.accuracy_score(y_test[i], y_pred_updated[i])
            acc_vec = acc / len(y_pred_updated)
            total_accuracy += acc_vec

        with open("LRWeights.txt", "w") as f:
            f.write("\n".join(" ".join(map(str, x)) for x
                              in (w0.T, w1.T, w2.T, w3.T, w4.T, w5.T, w6.T, w7.T, w8.T)))
        print('Accuracy of Linear Regressor on Intermediate boards optimal play '
              '(multi label): is ', total_accuracy/10, '\n')

    def kNearest(self):
        '''
        This function implements K nearest regressor
        It applies k-fold cross validation with k = 10
        number of neighbors = 9
        :return: None
        '''
        features = self.data_multi[:, :9]
        labels = self.data_multi[:, 9:]
        kf = KFold(n_splits=10)
        kf.get_n_splits(features)
        total_accuracy = 0.0
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            knn_reg_model = KNeighborsRegressor(n_neighbors=9)
            knn_reg_model.fit(X_train, y_train)
            y_pred = knn_reg_model.predict(X_test)
            y_pred_updated = np.where(y_pred > 0.5, 1, 0)
            acc = 0.0
            for i in range(len(y_pred_updated)):
                acc = acc + metrics.accuracy_score(y_test[i], y_pred_updated[i])
            acc_vec = acc / len(y_pred_updated)
            total_accuracy += acc_vec

        knnWeights = open('knnWeights', 'wb')
        pickle.dump(knn_reg_model, knnWeights)
        print('Accuracy of k-Nearest Regressor on Intermediate boards optimal '
              'play (multi label): is ', total_accuracy / 10, '\n')

    def MLP(self):
        '''
        This function implements a mlp regressor
        It applies k-fold cross validation with k = 10
        :return: None
        '''
        features = self.data_multi[:, :9]
        labels = self.data_multi[:, 9:]
        kf = KFold(n_splits=10)
        kf.get_n_splits(features)
        total_accuracy = 0.0
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            mlp_reg_model = MLPRegressor(solver='adam', activation='relu', hidden_layer_sizes=(200, 100, 40),
                                         random_state=1, max_iter=500).fit(X_train, y_train)
            y_pred = mlp_reg_model.predict(X_test)
            y_pred_updated = np.where(y_pred > 0.5, 1, 0)
            acc = 0.0
            for i in range(len(y_pred_updated)):
                acc = acc + metrics.accuracy_score(y_test[i], y_pred_updated[i])
            acc_vec = acc / len(y_pred_updated)
            total_accuracy += acc_vec
        # save the model
        MLPWeights = open('MLPWeights', 'wb')
        pickle.dump(mlp_reg_model, MLPWeights)
        print('Accuracy of MLP Regressor on Intermediate boards optimal play '
              '(multi label): is ', total_accuracy / 10)

if __name__ == "__main__":
    print('Output for Classifiers \n')
    classifier = Classifier()
    print('Output for Regresors \n')
    regressor = Regressor()
