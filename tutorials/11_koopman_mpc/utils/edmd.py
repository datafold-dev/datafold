from sklearn.cluster import KMeans

from datafold import (
    EDMD,
    DMDControl,
    GaussianKernel,
    TSCDataFrame,
    TSCIdentity,
    TSCPolynomialFeatures,
    TSCRadialBasis,
)

from .model import Predictor, PredictResult


class EDMD_Identity(Predictor):
    def _init_predictor(self):
        return EDMD(
            dict_steps=[
                ("id", TSCIdentity()),
            ],
            dmd_model=DMDControl(),
            include_id_state=False,
        )

    def fit(self, X_tsc: TSCDataFrame):
        X = X_tsc[self.state_cols]
        U = X_tsc[self.input_cols]
        self._predictor.fit(X, U=U)

    def predict(self, initial_conds, control_input, t):
        pred = self._predictor.predict(initial_conds, U=control_input, time_values=t)

        return PredictResult(
            control_input, initial_conds, pred, self.state_cols, self.input_cols
        )


class EDMD_RBF(Predictor):
    def __init__(self, num_rbfs, eps, *args, **kwargs):
        self.num_rbfs = num_rbfs
        self.eps = eps

        self.rbf = self._init_rbf()
        self.centers = None
        self.kmeans = None

        super().__init__(*args, **kwargs)

    def _init_predictor(self):
        return EDMD(
            dict_steps=[
                ("rbf", self.rbf),
            ],
            dmd_model=DMDControl(),
            include_id_state=True,
        )

    def _init_rbf(self):
        rbf = TSCRadialBasis(
            kernel=GaussianKernel(epsilon=self.eps), center_type="fit_params"
        )
        return rbf

    def _init_centers(self, X_tsc):
        X = X_tsc[self.state_cols + self.input_cols].to_numpy()
        km = KMeans(n_clusters=self.num_rbfs)
        km.fit(X)

        centers = km.cluster_centers_

        self.kmeans = km
        self.centers = centers
        return centers

    # def _init_centers(self, X_tsc):
    #     N = X_tsc.shape[0]
    #     centers = np.arange(0, N)
    #     center_sample = np.sort(np.random.choice(centers, size=self.num_rbfs,
    #                                              replace=False))

    #     centers = X_tsc.iloc[center_sample].values
    #     self.centers = centers
    #     return centers

    def fit(self, X_tsc: TSCDataFrame):
        X = X_tsc[self.state_cols]
        U = X_tsc[self.input_cols]

        centers = self._init_centers(X_tsc)

        self._predictor.fit(
            X,
            U=U,
            rbf__centers=centers[:, :-1],
        )

    def predict(self, initial_conds, control_input, t):
        pred = self._predictor.predict(
            initial_conds[self.state_cols],
            U=control_input,
            # control_input=np.atleast_2d(control_input).T,
            time_values=t,
        )

        return PredictResult(
            control_input, initial_conds, pred, self.state_cols, self.input_cols
        )


class EDMD_RBF_v2(EDMD_RBF):
    def __init__(self, *args, **kwargs):
        kwargs["input_cols"] = []
        super().__init__(*args, **kwargs)

    def _init_predictor(self):
        return EDMD(
            dict_steps=[
                ("rbf", self.rbf),
            ],
            dmd_model=DMDControl(),
            include_id_state=True,
        )

    def fit(self, X_tsc: TSCDataFrame):
        X = X_tsc[self.state_cols]
        U = X_tsc[self.input_cols]

        centers = self._init_centers(X_tsc)

        self._predictor.fit(
            X,
            U=U,
            rbf__centers=centers,
        )

    def predict(self, initial_conds, t):
        pred = self._predictor.predict(initial_conds[self.state_cols], time_values=t)

        return PredictResult(None, initial_conds, pred, self.state_cols, None)


class EDMD_Polynomial(EDMD_Identity):
    def __init__(self, n_degrees, *args, **kwargs):
        self.n_degrees = n_degrees

        super().__init__(*args, **kwargs)

    def _init_polynomial(self):
        return TSCPolynomialFeatures(degree=self.n_degrees)

    def _init_predictor(self) -> EDMD:
        dict_steps = [
            ("polynomial", self._init_polynomial()),
        ]
        return EDMD(dict_steps=dict_steps, include_id_state=True)
