import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from numpy import ndarray
from typing import Optional, cast,  List
from joblib import Parallel, delayed
from functools import partial
from scipy.linalg import eigh, pinv, qr
from scipy.stats import pearsonr
from scipy.sparse import block_diag, identity, vstack, spmatrix
from scipy.sparse.linalg import eigsh
from utils.concat_dataset import concat_dataset
import mne
from metabci.base import generate_cca_references, generate_filterbank, FilterBankSSVEP
from utils.speller_config import *
from metabci.model_selection import (set_random_seeds, generate_kfold_indices, match_kfold_indices)
import pandas as pd
from sklearn.model_selection import train_test_split


def _ged_wong( 
    Z: ndarray,
    D: Optional[ndarray] = None,
    P: Optional[ndarray] = None,
    n_components=1,
    method="type1",
):
    if method != "type1" and method != "type2":
        raise ValueError("not supported method type")

    A = Z
    if D is not None:
        A = D.T @ A
    if P is not None:
        A = P.T @ A
    A = A.T @ A
    if method == "type1":
        B = Z
        if D is not None:
            B = D.T @ Z
        B = B.T @ B
        if isinstance(A, spmatrix) or isinstance(B, spmatrix):
            D, W = eigsh(A, k=n_components, M=B)
        else:
            D, W = eigh(A, B)
    elif method == "type2":
        if isinstance(A, spmatrix):
            D, W = eigsh(A, k=n_components)
        else:
            D, W = eigh(A)

    D_exist = cast(ndarray, D)
    ind = np.argsort(D_exist)[::-1]
    D_exist, W = D_exist[ind], W[:, ind]
    return D_exist[:n_components], W[:, :n_components]

def _ecca_feature(
    X: ndarray,
    templates: ndarray,
    Yf: ndarray,
    Us: Optional[ndarray] = None,
    n_components: int = 1,
):
    if Us is None:
        Us_array, _ = zip(
            *[_scca_kernel(templates[i], Yf[i]) for i in range(len(templates))]
        )
        Us = np.stack(Us_array)
    rhos = []
    for Xk, Y, U3 in zip(templates, Yf, Us):
        rho_list = []
        # 14a, 14d
        U1, V1 = _scca_kernel(X, Y)
        a = U1[:, :n_components].T @ X
        b = V1[:, :n_components].T @ Y
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        a = U1[:, :n_components].T @ X
        b = U1[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # 14b
        U2, _ = _scca_kernel(X, Xk)
        a = U2[:, :n_components].T @ X
        b = U2[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        # 14c
        a = U3[:, :n_components].T @ X
        b = U3[:, :n_components].T @ Xk
        a, b = np.reshape(a, (-1)), np.reshape(b, (-1))
        rho_list.append(pearsonr(a, b)[0])
        rho = np.array(rho_list)
        rho = np.sum(np.sign(rho) * (rho**2))
        rhos.append(rho)
    return rhos

def _scca_kernel(X: ndarray, Yf: ndarray):
    """Standard CCA (sCCA).

    This is an time-consuming implementation due to GED.

    X: (n_channels, n_samples)
    Yf: (n_harmonics, n_samples)
    """
    n_components = min(X.shape[0], Yf.shape[0])
    Q, R = qr(Yf.T, mode="economic")
    P = Q @ Q.T
    Z = X.T
    _, U = _ged_wong(Z, None, P, n_components=n_components)  # U for X
    V = pinv(R) @ Q.T @ X.T @ U  # V for Yf
    return U, V

class ECCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1, n_jobs: Optional[int] = None):
        self.n_components = n_components
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: ndarray, Yf: ndarray):

        self.classes_ = np.unique(y)
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        self.templates_ = np.stack(
            [np.mean(X[y == label], axis=0) for label in self.classes_]
        )

        Yf = np.reshape(Yf, (-1, *Yf.shape[-2:]))
        Yf = Yf - np.mean(Yf, axis=-1, keepdims=True)
        self.Yf_ = Yf
        self.Us_, self.Vs_ = zip(
            *[
                _scca_kernel(self.templates_[i], self.Yf_[i])
                for i in range(len(self.classes_))
            ]
        )
        self.Us_, self.Vs_ = np.stack(self.Us_), np.stack(self.Vs_)
        return self

    def transform(self, X: ndarray):
        X = np.reshape(X, (-1, *X.shape[-2:]))
        X = X - np.mean(X, axis=-1, keepdims=True)
        templates = self.templates_
        Yf = self.Yf_
        Us = self.Us_
        n_components = self.n_components
        rhos = Parallel(n_jobs=self.n_jobs)(
            delayed(partial(_ecca_feature, Us=Us, n_components=n_components))(
                a, templates, Yf
            )
            for a in X
        )
        rhos = np.stack(rhos)
        return rhos

    def predict(self, X: ndarray):
        rhos = self.transform(X)
        labels = self.classes_[np.argmax(rhos, axis=-1)]
        return labels

class FBECCA(FilterBankSSVEP, ClassifierMixin):
    def __init__(
        self,
        filterbank: List[ndarray],
        n_components: int = 1,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.n_components = n_components
        self.filterweights = filterweights
        self.n_jobs = n_jobs
        super().__init__(
            filterbank,
            ECCA(n_components=n_components, n_jobs=1),
            filterweights=filterweights,
            n_jobs=n_jobs,
        )

    def fit(self, X: ndarray, y: ndarray, Yf: Optional[ndarray] = None):  # type: ignore[override]
        self.classes_ = np.unique(y)
        super().fit(X, y, Yf=Yf)
        return self

    def predict(self, X: ndarray):
        features = self.transform(X)
        if self.filterweights is None:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            features = np.mean(features, axis=1)
        labels = self.classes_[np.argmax(features, axis=-1)]
        return labels

def get_freq(event):
    return FREQS[event - 1]

def get_phase(event):
    return PHASES[event - 1]

def main():

    wp = [(11,92),(22,92),(34,92),(46,92),(58,92),(70,92),(82,92)]
    ws = []
    filterbank = generate_filterbank(wp, ws, srate=250)

    subjects = ['best_recording', 'wan']
    raw = concat_dataset(subjects, 5)
    raw.filter(11, 17, method='iir')
    events = mne.find_events(raw)
    events = events[events[:,2] != 99 ]
    print(events)
    print(type(events))

    freq_list = [get_freq(int(event)) for event in events[:,2]]
    phase_list = [get_phase(int(event)) for event in events[:,2]]

    Yf = generate_cca_references(freqs=freq_list, srate=250, T=0.2, phases=phase_list, n_harmonics=5)

    filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
    epochs = mne.Epochs(raw=raw,events=events, baseline=None, tmin=-0.2, tmax=2.0, reject=None, reject_by_annotation=False)
    X = epochs.get_data()
    y = events[:,2]
    # estimator=ECCA(n_components = 1, n_jobs=-1)
    estimator=FBECCA(filterbank=filterbank, n_components=1, filterweights=np.array(filterweights), n_jobs=-1)
    accs = []

    meta = pd.DataFrame(
                            {
                                "subject": 1,
                                "session": 1,
                                "run": 1,
                                "event": 1,
                                "trial_id": 1,
                                "dataset": 1,
                            }, index=[0]
                        )

    # 6-fold cross validation
    # set_random_seeds(38)
    # kfold = 6
    # indices = generate_kfold_indices(meta, kfold=kfold)

    # for k in range(kfold):
    #     train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
    #     # merge train and validate set
    #     train_ind = np.concatenate((train_ind, validate_ind))
    #     p_labels = estimator.fit(X=X[train_ind],y=y[train_ind],f=Yf).predict(X[test_ind])
    #     accs.append(np.mean(p_labels==y[test_ind]))
    # print(np.mean(accs))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = estimator.fit(X_train, y_train, Yf)
    preds = model.predict(X_test)
    acc = np.mean(preds==y_test)
    print(acc)

if __name__ == "__main__":
    main()




# import sys
# import numpy as np
# from brainda.algorithms.decomposition import ECCA
# from brainda.algorithms.decomposition.base import generate_filterbank,

# wp=[(5,90)]
# ws=[(3,92)]
# filterbank = generate_filterbank(wp,ws,srate=250,order=15,rp=0.5)
# filterweights = [(idx_filter+1) ** (-1.25) + 0.25 for idx_filter in range(5)]
# estimator=ECCA(n_components = 1, n_jobs=-1)
# accs = []
# for k in range(kfold):
#     train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
#     # merge train and validate set
#     train_ind = np.concatenate((train_ind, validate_ind))
#     p_labels = estimator.fit(X=X[train_ind],y=y[train_ind],f=Yf).predict(X[test_ind])
#     accs.append(np.mean(p_labels==y[test_ind]))
#     print(np.mean(accs))