import abc
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
import tqdm
from sklearn.base import BaseEstimator
from sklearn.utils import check_scalar

from datafold.dynfold.base import TSCPredictMixin, TSCTransformerMixin
from datafold.dynfold.dmd import DMDBase, PretrainedDMD
from datafold.pcfold import TSCDataFrame

try:
    import torch

    IS_TORCH_IMPORT = True
except ImportError:
    torch = object
    IS_TORCH_IMPORT = False


class DictLearningMethod(BaseEstimator, TSCTransformerMixin, metaclass=abc.ABCMeta):
    """Abtract base class for dictionary learning methods.

    An implementing class should store the linear system matrix during fit.
    The ``transform`` method maps the original time series to the learnt dictionary functions.

    Parameters
    ----------
    sys_type
        Whether the system matrix represents a flow map or vector field (differential).
    """

    def __init__(
        self,
        sys_type=Literal["flowmap", "differential"],
    ) -> None:
        self.sys_type = sys_type
        self.system_matrix_: np.ndarray


try:

    class EarlyStopping:
        def __init__(self, patience, delta=0):
            self.patience = patience
            self.delta = delta
            self.counter = 0
            self.best_score = None
            self.early_stop = False

        def update(self, val_loss) -> None:
            if self.best_score is None:
                self.best_score = val_loss
            elif val_loss > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_loss
                self.counter = 0

        def is_stop(self) -> bool:
            return self.early_stop

    class FeedforwardNN(DictLearningMethod):
        """
        Layered feedforward neural network to learn a dictionary.

        This is the training algorithm used in the original EDMD-DL in :cite:t:`li-2017`.


        Parameters
        ----------
        hidden_size
            The number of weights per hidden layer.

        n_hidden_layer
            The number of hidden layers.

        n_dict_elements
            The number weights on the output, corresponding to the number of
            dictionary functions.

        include_id_state
            Whether to incldue the original state. If set to False, then it is advised to set
            ``loss_with_inverse_map=True`` to avoid trivial zero solution (see :cite:`li-2017`
            for details).

        loss_with_inverse_map
            Whether to include the reconstruction error from the dictionary to the
            original state.

        learning_rate
            The (initial) learning rate of the network. A scheduler can be set in
            ``fit_params``.

        n_epochs
            The number of epochs to perform. It is possible to set early stopping in
            ``fit_params``.

        batch_size: int = 50
            The batch size of the data to use per iteration.

        loss_regularization
            The regularization used within the loss function.

        sys_regularization
            The regularization used to compute the system matrix.

        use_dtype
            The data type used for the network. This becomes more relevant if GPU training is
            supported (implementation required).

        random_state
            A seed passed to PyTorch.
        """

        class _TorchTrainDataset(torch.utils.data.Dataset):
            def __init__(self, X_now, psi_next):
                self.X_now = X_now
                self.psi_next = psi_next

            def __len__(self):
                return self.X_now.shape[0]

            def __getitem__(self, i):
                return self.X_now[i], self.psi_next[i]

        class _TorchValidatationDataset(torch.utils.data.Dataset):
            def __init__(self, X_now, psi_now, psi_next):
                self.X_now = X_now
                self.psi_now = psi_now
                self.psi_next = psi_next

            def __len__(self):
                return self.X_now.shape[0]

            def __getitem__(self, i):
                return self.X_now[i], self.psi_now[i], self.psi_next[i]

        class _intern_FNN(torch.nn.Module):
            def __init__(
                self,
                input_size,
                hidden_size,
                n_hidden_layers,
                output_size,
                activation_function=torch.nn.Tanh(),
                bias=True,
                use_dtype=torch.float64,
            ):
                super().__init__()

                layers = []
                layer_sizes = (
                    [input_size] + [hidden_size] * (n_hidden_layers + 1) + [output_size]
                )
                layers = list()

                for i in range(len(layer_sizes) - 1):
                    layers.append(
                        torch.nn.Linear(
                            layer_sizes[i],
                            layer_sizes[i + 1],
                            bias=bias,
                            dtype=use_dtype,
                        )
                    )

                    if i != len(layer_sizes) - 2:
                        # only use activation functions for the hidden layers
                        layers.append(activation_function)

                self.model = torch.nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

            def sample_init_weights(self, X):
                Xm, Xp = X.tsc.shift_matrices(snapshot_orientation="row")

                import swimnetworks
                from sklearn.pipeline import Pipeline

                dict_steps = [
                    ("l1", swimnetworks.Dense(layer_width=100, activation="tanh")),
                    ("l2", swimnetworks.Dense(layer_width=100, activation="tanh")),
                    ("l3", swimnetworks.Dense(layer_width=100, activation="tanh")),
                    ("l4", swimnetworks.Dense(layer_width=100, activation="tanh")),
                ]
                pipeline = Pipeline(dict_steps).fit(Xm, Xp)

                with torch.no_grad():
                    for i in range(4):
                        self.model[i * 2].weight.copy_(
                            torch.Tensor(pipeline[i].weights.T)
                        )
                        self.model[i * 2].bias.copy_(
                            torch.Tensor(pipeline[i].biases[0].T)
                        )

                # from sklearn.decomposition import PCA

                # pca = PCA(n_components=22).fit(pipeline.predict(Xm))
                # with torch.no_grad():
                #     self.model[8].weight.copy_(torch.Tensor(pca.components_))
                # self.model[8].bias.copy_() not sure...j

            @property
            def n_params(self) -> int:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def __init__(
            self,
            hidden_size=10,
            n_hidden_layer=1,
            n_dict_elements=10,
            include_id_state=True,
            loss_with_inverse_map=False,
            learning_rate: float = 1e-4,
            n_epochs: int = 100,
            batch_size: int = 50,
            loss_regularization=0.0,
            sys_regularization=0.0,
            use_dtype=torch.float64,
            random_state: Optional[int] = None,
        ) -> None:
            super().__init__(sys_type="flowmap")

            if not IS_TORCH_IMPORT:
                raise ImportError(
                    "To use this function the Python package `torch` is required. "
                    "Install with `pip install torch`"
                )

            # TODO: make sure that different dtypes are used correctly (other than float64)
            # TODO: checkpoint (with path) best (i.e. lowest validation loss model) also
            #   provide an option (classmethod?) where model is saved - this can be used to
            #   continue / restart method (i.e. some init-parameters relating the NN are
            #   ignored but training is used)
            # TODO: test training with GPU - no transfer to GPU at all is implemented yet
            #   (for scaling it to GPU clusters, possibly PyTorch lightning is the best option,
            #   however, this may require re-writing also the PyTorch stuff)
            # TODO: avoid transitions between numpy/pytorch (later GPU devices)

            self.hidden_size = hidden_size
            self.n_hidden_layer = n_hidden_layer
            self.n_dict_elements = n_dict_elements
            self.include_id_state = include_id_state
            self.loss_with_inverse_map = loss_with_inverse_map
            self.learning_rate = learning_rate
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.sys_regularization = sys_regularization
            self.loss_regularization = loss_regularization
            self.use_dtype = use_dtype
            self.random_state = random_state

        def get_feature_names_out(self, input_features=None):
            psi = ["const"] + [f"psi{i}" for i in range(self.n_dict_elements)]
            # return list(self.feature_names_in_) + ["const"] + psi
            if self.include_id_state:
                psi = list(self.feature_names_in_) + psi
            return psi

        @property
        def n_params(self) -> int:
            return self.method.n_params

        def _to_torch(self, X):
            if torch.is_tensor(X):
                return X

            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()

            if isinstance(X, np.ndarray):
                return torch.tensor(X).to(self.use_dtype)
            else:
                raise RuntimeError(f"{type(X)=} is not supported")

        def _validate_parameter(self):
            check_scalar(
                self.n_dict_elements, name="n_dict_elements", target_type=int, min_val=1
            )

            check_scalar(self.n_epochs, name="n_epochs", target_type=int, min_val=1)

            check_scalar(
                self.sys_regularization,
                name="sys_regularization",
                target_type=(int, float),
                min_val=0.0,
            )

            check_scalar(
                self.loss_regularization,
                name="loss_regularization",
                target_type=(int, float),
                min_val=0.0,
            )

        def _augment_id_const(
            self, X, psi, *, time_indices_from: Optional[TSCDataFrame] = None
        ):
            if self.include_id_state:
                # augment dictionary with full state (ID) and psi
                psi = torch.cat([X, torch.ones(X.shape[0], 1), psi], dim=1)
            else:
                psi = torch.cat([torch.ones(X.shape[0], 1), psi], dim=1)

            if time_indices_from is not None:
                # if set returns the data in TSCDataFrame
                # note that this converts a pytorch dataset into Numpy
                psi = TSCDataFrame.from_same_indices_as(
                    indices_from=time_indices_from,
                    values=psi.detach().numpy(),
                    except_columns=self.get_feature_names_out(),
                )
            return psi

        def _compute_system_matrix(
            self, Xtorch, Xtsc, *, to_torch=False, return_shift_matrices=False
        ):
            with torch.no_grad():
                # use current mapping of the neural network
                psi = self.method(Xtorch)

            psi = self._augment_id_const(Xtorch, psi, time_indices_from=Xtsc)
            psi_now, psi_next = psi.tsc.shift_matrices(snapshot_orientation="row")

            # Eq. 5 in https://arxiv.org/pdf/1707.00225.pdf
            # Note: factor actually does not change the result if regulariation is zero
            # to match the source with same scale of regularization influence the factor
            # is included
            # TODO: this is solved in multiple places (mainly dmd.py) there should be a general
            #  class that performs this, especially if different sorts of regularization are
            #  implemented.
            factor = 1 / Xtorch.shape[0]

            G = factor * (psi_now.T @ psi_now)
            A = factor * (psi_now.T @ psi_next)

            if self.sys_regularization > 0.0:
                G.ravel()[:: G.shape[1] + 1] += self.sys_regularization

            K = np.linalg.lstsq(
                G,
                A,
                rcond=None,
            )[0]

            if to_torch:
                K = torch.tensor(K)

            if self.loss_with_inverse_map:
                inverse_map = np.linalg.lstsq(
                    psi.to_numpy(), Xtorch.detach().numpy(), rcond=None
                )[0]
                inverse_map = torch.tensor(inverse_map).to(torch.float64)
            else:
                inverse_map = None

            if return_shift_matrices:
                return K, psi_now, psi_next, inverse_map
            else:
                return K

        def loss_func(
            self, _lambda, psi_now, system_matrix, psi_next, inverse_map, X_now
        ):
            loss = torch.mean(torch.square(psi_next - (psi_now @ system_matrix)))

            if _lambda > 0.0:
                reg = _lambda * torch.norm(system_matrix)
                loss += reg

            if self.loss_with_inverse_map:
                loss += torch.mean(torch.square(psi_now @ inverse_map - X_now))

            return loss

        def _train_epoch(self, system_matrix, inverse_map, trainloader, optimizer):
            current_train_loss = 0

            # Step 2: Run the training loop of neural network
            # Iterate over the DataLoader for training data
            for X_now, psi_next in trainloader:
                # standard Pytorch training loop
                optimizer.zero_grad()

                # perform forward pass (with storing gradient information)
                _psi_now_raw = self.method(X_now)
                _psi_now_aug = self._augment_id_const(X_now, _psi_now_raw)

                # TODO: not sure if it is better to "dynamically"
                #   (i.e. per iteration) update the model for psi_next?
                #    -- this means the target function changes between batches
                # X_next = psi_next[:, :2]
                # with torch.no_grad():
                #    psi_next = self._augment_id_const(X_next, self.method(X_next))

                train_loss = self.loss_func(
                    _lambda=self.loss_regularization,
                    psi_now=_psi_now_aug,
                    system_matrix=system_matrix,
                    psi_next=psi_next,
                    inverse_map=inverse_map,
                    X_now=X_now,
                )

                # perform backward pass and optimization
                train_loss.backward()
                optimizer.step()

                current_train_loss += train_loss.item()

            return current_train_loss

        def _validate_network(
            self, X, Xtorch_val, X_val, system_matrix, inverse_map, early_stopping
        ):
            current_val_loss = 0

            with torch.no_grad():
                # use current mapping of the neural network
                psi = self.method(Xtorch_val)

            psi = self._augment_id_const(Xtorch_val, psi, time_indices_from=X_val)
            psi_now_aug, psi_next_aug = psi.tsc.shift_matrices(
                snapshot_orientation="row"
            )

            val_dataset = self._TorchValidatationDataset(
                X_now=psi_now_aug[:, : X.shape[1]],
                psi_now=self._to_torch(psi_now_aug),
                psi_next=self._to_torch(psi_next_aug),
            )

            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
            )

            for _X_now, _psi_now, _psi_next in val_dataloader:
                val_loss = self.loss_func(
                    _lambda=self.loss_regularization,
                    psi_now=_psi_now,
                    system_matrix=system_matrix,
                    psi_next=_psi_next,
                    inverse_map=inverse_map,
                    X_now=_X_now,
                )

                current_val_loss += val_loss.item()

            if early_stopping is not None:
                early_stopping.update(val_loss=current_val_loss)

            return early_stopping, current_val_loss

        def fit(self, X: TSCDataFrame, *, y=None, **fit_params):
            """Fit the model.

            Parameter
            ---------
            X
                Training time series.
            y
                ignored

            **fit_params
            - ``X_val`` a time series for validation (may be required by other
                ``fit_params`` features)
            - ``early_stopping`` keyword arguments passed to EarlyStopping
            - ``lr_scheduler`` a callable setting a PyTorch scheduler with
                ``scheduler(optimizer)``
            - ``record_losses`` a bool to indicate whether to store the training and validation
                (only if ``X_val`` is provided) losses
            - ``with_tqdm`` a bool to indicate whether to monitor the iterative training with a
                tqdm progress bar
            """
            self._validate_parameter()

            self._setup_feature_attrs_fit(X, n_features_out=self.n_dict_elements)

            (
                X_val,
                early_stopping_kwargs,
                lr_scheduler,
                record_losses,
                with_tqdm,
            ) = self._read_fit_params(
                attrs=[
                    ("X_val", None),
                    ("early_stopping", None),
                    ("lr_scheduler", None),
                    ("record_losses", False),
                    ("with_tqdm", True),
                ],
                fit_params=fit_params,
            )

            is_validation_available = X_val is not None

            if self.random_state is not None:
                torch.manual_seed(self.random_state)

            self.method = self._intern_FNN(
                input_size=X.shape[1],
                hidden_size=self.hidden_size,
                n_hidden_layers=self.n_hidden_layer,
                output_size=self.n_dict_elements,
                use_dtype=self.use_dtype,
            )

            # if False:
            #     self.method.sample_init_weights(X=X)

            # these two cases are required during the training loop
            X_all_torch = self._to_torch(X)
            X_now_torch = self._to_torch(
                X.tsc.shift_matrices(snapshot_orientation="row")[0]
            )

            if is_validation_available:
                Xtorch_val = self._to_torch(X_val)

            optimizer = torch.optim.Adam(
                self.method.parameters(), lr=self.learning_rate
            )

            if lr_scheduler is not None:
                lr_scheduler = lr_scheduler(optimizer)

            if record_losses:
                self.fit_losses_ = np.zeros(self.n_epochs) * np.nan
                if is_validation_available:
                    self.val_losses_ = np.zeros(self.n_epochs) * np.nan

            if early_stopping_kwargs is not None:
                if not is_validation_available:
                    raise ValueError(
                        "early stopping parameters are provided but no validation data X_val"
                    )
                early_stopping = EarlyStopping(**early_stopping_kwargs)
            else:
                early_stopping = None

            require_val_loss = (lr_scheduler is not None) or (
                early_stopping is not None
            )

            if not require_val_loss and is_validation_available:
                warnings.warn(
                    "Validation set (X_val) is provided but no feature it is not "
                    "required in the current training setting.",
                    stacklevel=1,
                )

            if with_tqdm:
                all_epochs = tqdm.tqdm(range(self.n_epochs))
            else:
                all_epochs = range(self.n_epochs)

            # number of iterations of alternating optimiation steps.
            for epoch in all_epochs:
                system_matrix, _, psi_next, inverse_map = self._compute_system_matrix(
                    X_all_torch, X, to_torch=True, return_shift_matrices=True
                )

                train_data_nn = self._TorchTrainDataset(
                    X_now=X_now_torch,
                    psi_next=self._to_torch(psi_next),
                )

                trainloader = torch.utils.data.DataLoader(
                    train_data_nn,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=1,
                )

                # train network
                current_train_loss = self._train_epoch(
                    system_matrix,
                    inverse_map,
                    trainloader=trainloader,
                    optimizer=optimizer,
                )

                if record_losses:
                    self.fit_losses_[epoch] = current_train_loss

                # evaluate validation loss
                if require_val_loss:
                    early_stopping, current_val_loss = self._validate_network(
                        X, Xtorch_val, X_val, system_matrix, inverse_map, early_stopping
                    )

                    if record_losses:
                        self.val_losses_[epoch] = current_val_loss

                # learning rate scheduler
                if lr_scheduler is not None:
                    try:
                        lr_scheduler.step(current_val_loss)
                    except TypeError:
                        # some schedulers (e.g. StepLR) are not dependent on loss information
                        lr_scheduler.step()

                # stop training if early stopping indicates
                if early_stopping is not None and early_stopping.is_stop():
                    break

                # update progress bar
                if with_tqdm:
                    tqdm_output = dict(loss=current_train_loss)

                    if is_validation_available:
                        tqdm_output["vloss"] = current_val_loss
                    if lr_scheduler is not None:
                        tqdm_output["lr"] = optimizer.param_groups[0]["lr"]

                    all_epochs.set_postfix(**tqdm_output)

            if record_losses and early_stopping_kwargs is not None:
                # truncate loss records in case there was an early stop during training
                self.fit_losses_ = self.fit_losses_[~np.isnan(self.fit_losses_)]
                if is_validation_available:
                    self.val_losses_ = self.val_losses_[~np.isnan(self.val_losses_)]

            # setup final system matrix in attribute
            self.system_matrix_ = self._compute_system_matrix(
                X_all_torch, X, to_torch=False
            )
            self.system_matrix_ = self.system_matrix_.T
            return self

        def transform(self, X: TSCDataFrame) -> TSCDataFrame:
            self._validate_feature_input(X=X, direction="transform")

            Xtorch = torch.from_numpy(X.to_numpy()).to(dtype=self.use_dtype)
            psi = self.method(Xtorch)
            X_transform = self._augment_id_const(X=Xtorch, psi=psi, time_indices_from=X)
            return X_transform

except AttributeError as e:
    if IS_TORCH_IMPORT:
        raise e


class DMDDictLearning(TSCTransformerMixin, TSCPredictMixin):
    """Learn a dictionary from data and then perform a dynamic mode decomposition.

    Combined with :py:class:`EDMD` (as the `dmd_method`) this enables EDMD-DL
    (cf. :cite:t:`li-2017`), where the dictionary is learnt from the available data.
    It improves upon the classical EDMD method because there is no need to set the
    observables explicitly. Instead it allows learning a flexible set of observables.
    This class requires specifying a method that performs the actual dictionary learning.
    The standard case, as described in :cite:t:`li-2017`, is to use a feedforward neural
    network (see :class:`FeedforwardNN`) but in principle other learning algorithms can be
    implemented.

    Parameters
    ----------

    learning_model
        The method to learn the dictionary.

    References
    ----------

    :cite:t:`li-2017`

    """

    def __init__(
        self,
        learning_model: DictLearningMethod,
        sys_mode: Literal["matrix", "spectral"] = "spectral",
        sys_type: Literal["flowmap", "differential"] = "flowmap",
        is_diagonalize: bool = False,
    ):
        self.learning_model = learning_model
        self.sys_mode = sys_mode
        self.sys_type = sys_type
        self.is_diagonalize = is_diagonalize
        self.dmd_model: DMDBase

    def fit(
        self, X: TSCDataFrame, *, y=None, U=None, **fit_params
    ) -> "DMDDictLearning":
        """Train dictionary functions and perform dynamic mode decomposition.

        Parameters
        ----------
        X
            Training time series.

        y
            ignored (this could be extended to account for different set of observables
            to reconstruct to)

        U
            ignored (this could be extended to account for control input)

        Returns
        -------
        DMDDictLearning
            fitted model
        """
        self.learning_model = self.learning_model.fit(X, **fit_params)

        # TODO: this can be avoided, as it is only used to setup some standard attributes
        #  in the pretrained DMD
        X_dict = self.learning_model.transform(X)

        self.dmd_model = PretrainedDMD.from_available_system_matrix(
            sys_type=self.sys_type,
            sys_mode=self.sys_mode,
            system_matrix=self.learning_model.system_matrix_,
            is_diagonalize=self.is_diagonalize,
        )

        self.dmd_model = self.dmd_model.fit(X=X_dict, U=None)
        return self
