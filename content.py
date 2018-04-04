# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np


def sigmoid(x):
    """
    :param x: wektor wejsciowych wartosci Nx1
    :return: wektor wyjściowych wartości funkcji sigmoidalnej dla wejścia x, Nx1
    """
    return 1/(1+np.exp(-x))
    #pass

def logistic_cost_function(w, x_train, y_train):
    """
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej, a grad jej gradient po w
    """
    N = y_train.shape[0]
    sigmas = sigmoid(x_train.dot(w))
    cost = y_train*(np.log(sigmas)) + (1-y_train)*(np.log(1-sigmas))
    val = (-1/N)*np.sum(cost)
    grad = np.dot(x_train.T, sigmas-y_train) / N
    return val, grad
    #pass

def gradient_descent(obj_fun, w0, epochs, eta):
    """
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w).
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok / iteracji algorytmu
    :param eta: krok uczenia
    :return: funkcja wykonuje optymalizacje metoda gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_values jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu
    """
    func_values = []
    w = w0
    grad = obj_fun(w)[1]
    for i in range(epochs):
        w = w - (eta*grad)
        curr_val, grad = obj_fun(w)
        func_values.append(curr_val)
    func_values = np.array(func_values, ndmin = 2)
    return w, func_values.T
    #pass

def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w,x,y), gdzie x,y oznaczaja podane
    podzbiory zbioru treningowego (mini-batche)
    :param x_train: dane treningowe wejsciowe NxM
    :param y_train: dane treningowe wyjsciowe Nx1
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini-batcha
    :return: funkcja wykonuje optymalizacje metoda stochastycznego gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_values jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu. Wartosci
    funkcji do func_values sa wyliczane dla calego zbioru treningowego!
    """
    M = x_train.shape[0]/mini_batch
    M = int(M)
    w = w0
    func_values = []
    for i in range(epochs):
        for m in range(M):
            x_sub = x_train[m*mini_batch: (m+1)*mini_batch]
            y_sub = y_train[m*mini_batch: (m+1)*mini_batch]
            grad = obj_fun(w, x_sub, y_sub)[1]
            w = w - (eta*grad)
        val = obj_fun(w, x_train, y_train)[0]
        func_values.append(val)
    func_values = np.array(func_values, ndmin = 2)
    return w, func_values.T
    #pass

def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej z regularyzacja l2,
    a grad jej gradient po w
    """
    val, grad = logistic_cost_function(w, x_train, y_train)
    w_nozero = w[1:w.shape[0]]
    val = val + ((regularization_lambda/2)*(np.linalg.norm(w_nozero)**2))
    w_derivative = w.copy()
    w_derivative[0] = 0 #pochodna wyrazu wolnego
    grad = grad + regularization_lambda*w_derivative
    return val, grad
    #pass

def prediction(x, w, theta):
    """
    :param x: macierz obserwacji NxM
    :param w: wektor parametrow modelu Mx1
    :param theta: prog klasyfikacji z przedzialu [0,1]
    :return: funkcja wylicza wektor y o wymiarach Nx1. Wektor zawiera wartosci etykiet ze zbioru {0,1} dla obserwacji z x
     bazujac na modelu z parametrami w oraz progu klasyfikacji theta
    """
    y = []
    sigmas = sigmoid(x.dot(w))
    for sigma in sigmas:
        y.append(sigma>theta)
    y = np.array(y)
    return y
    #pass

def f_measure(y_true, y_pred):
    """
    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet przewidzianych przed model Nx1
    :return: funkcja wylicza wartosc miary F
    """
    tp = 0
    fp = 0
    fn = 0
    for i in range(y_true.shape[0]):
        if(y_true[i]==1 and y_pred[i]==1):
            tp +=1
        if(y_pred[i]==0 and y_true[i]==1):
            fn +=1
        if(y_pred[i]==1 and y_true[i]==0):
            fp +=1
    return (2*tp)/((2*tp)+fp+fn)        
    #pass

def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    :param x_train: ciag treningowy wejsciowy NxM
    :param y_train: ciag treningowy wyjsciowy Nx1
    :param x_val: ciag walidacyjny wejsciowy Nval x M
    :param y_val: ciag walidacyjny wyjsciowy Nval x 1
    :param w0: wektor poczatkowych wartosci parametrow
    :param epochs: liczba epok dla SGD
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini batcha
    :param lambdas: lista wartosci parametru regularyzacji lambda, ktore maja byc sprawdzone
    :param thetas: lista wartosci progow klasyfikacji theta, ktore maja byc sprawdzone
    :return: funckja wykonuje selekcje modelu. Zwraca krotke (regularization_lambda, theta, w, F), gdzie regularization_lambda
    to najlpszy parametr regularyzacji, theta to najlepszy prog klasyfikacji, a w to najlepszy wektor parametrow modelu.
    Dodatkowo funkcja zwraca macierz F, ktora zawiera wartosci miary F dla wszystkich par (lambda, theta). Do uczenia nalezy
    korzystac z algorytmu SGD oraz kryterium uczenia z regularyzacja l2.
    """
    best_measure = 0
    best_lambda = lambdas[0]
    best_theta = thetas[0]
    best_w = w0
    F = []
    for l in lambdas:
        fun = lambda w,x_train,y_train: regularized_logistic_cost_function(w, x_train, y_train, l)
        w = stochastic_gradient_descent(fun, x_train, y_train, w0, epochs, eta, mini_batch)[0]
        for theta in thetas:
            curr_measure = f_measure(y_val, prediction(x_val, w, theta))
            F.append(curr_measure)
            if curr_measure>best_measure:
                best_measure = curr_measure
                best_lambda = l
                best_theta = theta
                best_w = w
    F = np.array(F).reshape(len(lambdas),len(thetas))
    return best_lambda, best_theta, best_w, F
    #pass