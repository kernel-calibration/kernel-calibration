from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from sklearn.datasets import load_boston
from functools import partial

class RegressionDataModule(LightningDataModule):
    """Datamodule for regression datasets.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_dir: str = "data/",
        random_seed: Optional[int] = 1,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        batch_size: int = 64,
        test_batch_size: int = 128,
        normalize: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        normalize_y: Optional[bool] = False,
    ):
        super().__init__()

        assert np.isclose(sum(train_val_test_split), 1), f"Train_val_test_split must sum to 1. Got {train_val_test_split} with sum {sum(train_val_test_split):0.5f}."

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.input_size: int = regression_shapes[dataset_name][0]
        self.label_size: int = regression_shapes[dataset_name][1]

        self.setup()

    @property
    def dataset_type(self) -> str:
        return "regression"

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            loader_fun = regression_load_funs[self.hparams.dataset_name]
            X, y = loader_fun(self.hparams.data_dir)
            if self.hparams.normalize:
                # print(f'X.shape = {X.shape}')
                # std = X.std(axis=0)
                # print(f'std.shape = {std.shape}')
                # # print(f'std = {std}')
                # zeros = ~np.isclose(std, 0.)
                # num_zeros = np.sum(1*zeros)
                # print(f'num of zeros = {num_zeros}')
                # new_X = X[:, zeros]
                # print(f'new_X.shape = {new_X.shape}')
                # X = (new_X - new_X.mean(axis=0)) / new_X.std(axis=0)
                # old_shape = regression_shapes[self.hparams.dataset_name]
                # regression_shapes[self.hparams.dataset_name] = (old_shape[0] - int(num_zeros), 1)

                # self.input_size = regression_shapes[self.hparams.dataset_name][0]
                # self.label_size = regression_shapes[self.hparams.dataset_name][1]

                std = X.std(axis=0)
                zeros = np.isclose(std, 0.)
                X[:, ~zeros] = (X[:, ~zeros] - X[:, ~zeros].mean(axis=0)) / X[:, ~zeros].std(axis=0)
                X[:, zeros] = 0.

                # X = (X - X.mean(axis=0)) / X.std(axis=0)
                # X = np.nan_to_num(X)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            if self.hparams.normalize_y:
                print('Normalizing y!')
                y = y - np.median(y, axis=0)
            dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
            lengths = [int(len(X) * p) for p in self.hparams.train_val_test_split]
            lengths[-1] += len(X) - sum(lengths)  # fix any rounding errors
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(self.hparams.random_seed),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True
        )


def _load_boston(data_dir):
    """
    Attribute Information:
    1. CRIM: per capita crime rate by town
    2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
    3. INDUS: proportion of non-retail business acres per town
    4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    5. NOX: nitric oxides concentration (parts per 10 million)
    6. RM: average number of rooms per dwelling
    7. AGE: proportion of owner-occupied units built prior to 1940
    8. DIS: weighted distances to five Boston employment centres
    9. RAD: index of accessibility to radial highways
    10. TAX: full-value property-tax rate per $10,000
    11. PTRATIO: pupil-teacher ratio by town
    12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    13. LSTAT: % lower status of the population
    14. MEDV: Median value of owner-occupied homes in $1000's
    """
    X, y = load_boston(return_X_y=True)
    return X, y


def _load_powerplant(data_dir):
    """
    attribute information:
    features consist of hourly average ambient variables
    - temperature (t) in the range 1.81 c and 37.11 c,
    - ambient pressure (ap) in the range 992.89-1033.30 millibar,
    - relative humidity (rh) in the range 25.56% to 100.16%
    - exhaust vacuum (v) in teh range 25.36-81.56 cm hg
    - net hourly electrical energy output (ep) 420.26-495.76 mw
    the averages are taken from various sensors located around the
    plant that record the ambient variables every second.
    the variables are given without normalization.
    """
    data_file = os.path.join(data_dir, "uci/power-plant/Folds5x2_pp.xlsx")
    data = pd.read_excel(data_file)
    x = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return x, y


def _load_concrete(data_dir):
    """
    Summary Statistics:
    Number of instances (observations): 1030
    Number of Attributes: 9
    Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
    Missing Attribute Values: None
    Name -- Data Type -- Measurement -- Description
    Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
    Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
    Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
    Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
    Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
    Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
    Age -- quantitative -- Day (1~365) -- Input Variable
    Concrete compressive strength -- quantitative -- MPa -- Output Variable
    ---------------------------------
    """
    data_file = os.path.join(data_dir, "uci/concrete/Concrete_Data.xls")
    data = pd.read_excel(data_file)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_yacht(data_dir):
    """
    Attribute Information:
    Variations concern hull geometry coefficients and the Froude number:
    1. Longitudinal position of the center of buoyancy, adimensional.
    2. Prismatic coefficient, adimensional.
    3. Length-displacement ratio, adimensional.
    4. Beam-draught ratio, adimensional.
    5. Length-beam ratio, adimensional.
    6. Froude number, adimensional.
    The measured variable is the residuary resistance per unit weight of displacement:
    7. Residuary resistance per unit weight of displacement, adimensional.
    """
    data_file = os.path.join(data_dir, "uci/yacht/yacht_hydrodynamics.data")
    data = pd.read_csv(data_file, delim_whitespace=True, header=None)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_energy_efficiency(data_dir):
    """
    Data Set Information:
    We perform energy analysis using 12 different building shapes simulated in
    Ecotect. The buildings differ with respect to the glazing area, the
    glazing area distribution, and the orientation, amongst other parameters.
    We simulate various settings as functions of the afore-mentioned
    characteristics to obtain 768 building shapes. The dataset comprises
    768 samples and 8 features, aiming to predict two real valued responses.
    It can also be used as a multi-class classification problem if the
    response is rounded to the nearest integer.
    Attribute Information:
    The dataset contains eight attributes (or features, denoted by X1...X8) and two responses (or outcomes, denoted by y1 and y2). The aim is to use the eight features to predict each of the two responses.
    Specifically:
    X1    Relative Compactness
    X2    Surface Area
    X3    Wall Area
    X4    Roof Area
    X5    Overall Height
    X6    Orientation
    X7    Glazing Area
    X8    Glazing Area Distribution
    y1    Heating Load
    y2    Cooling Load
    """
    data_file = os.path.join(data_dir, "uci/energy-efficiency/ENB2012_data.xlsx")
    data = pd.read_excel(data_file)
    X = data.values[:, :-4]
    y_heating = data.to_numpy()[:, -4]
    y_cooling = data.to_numpy()[:, -3]  # There are two dead columns in the end, remove them here
    return X, y_cooling


def _load_wine(data_dir):
    """
    Attribute Information:
    For more information, read [Cortez et al., 2009].
    Input variables (based on physicochemical tests):
    1 - fixed acidity
    2 - volatile acidity
    3 - citric acid
    4 - residual sugar
    5 - chlorides
    6 - free sulfur dioxide
    7 - total sulfur dioxide
    8 - density
    9 - pH
    10 - sulphates
    11 - alcohol
    Output variable (based on sensory data):
    12 - quality (score between 0 and 10)
    """
    data_file = os.path.join(data_dir, 'uci/wine-quality/winequality-red.csv')
    data = pd.read_csv(data_file, sep=";")
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_kin8nm(data_dir):
    """
    This is data set is concerned with the forward kinematics of an 8 link robot arm. Among the existing variants of
     this data set we have used the variant 8nm, which is known to be highly non-linear and medium noisy.
    Original source: DELVE repository of data. Source: collection of regression datasets by Luis Torgo
    (ltorgo@ncc.up.pt) at http://www.ncc.up.pt/~ltorgo/Regression/DataSets.html Characteristics: 8192 cases,
    9 attributes (0 nominal, 9 continuous).
    Input variables:
    1 - theta1
    2 - theta2
    ...
    8 - theta8
    Output variable:
    9 - target
    """
    data_file = os.path.join(data_dir, 'kin8nm/dataset_2175_kin8nm.csv')
    data = pd.read_csv(data_file)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_naval(data_dir):
    """
    http://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
    Input variables:
    1 - Lever position(lp)[]
    2 - Ship speed(v)[knots]
    3 - Gas Turbine shaft torque(GTT)[kNm]
    4 - Gas Turbine rate of revolutions(GTn)[rpm]
    5 - Gas Generator rate of revolutions(GGn)[rpm]
    6 - Starboard Propeller Torque(Ts)[kN]
    7 - Port Propeller Torque(Tp)[kN]
    8 - HP Turbine exit temperature(T48)[C]
    9 - GT Compressor inlet air temperature(T1)[C]
    10 - GT Compressor outlet air temperature(T2)[C]
    11 - HP Turbine exit pressure(P48)[bar]
    12 - GT Compressor inlet air pressure(P1)[bar]
    13 - GT Compressor outlet air pressure(P2)[bar]
    14 - Gas Turbine exhaust gas pressure(Pexh)[bar]
    15 - Turbine Injecton Control(TIC)[ %]
    16 - Fuel flow(mf)[kg / s]
    Output variables:
    17 - GT Compressor decay state coefficient.
    18 - GT Turbine decay state coefficient.
    """
    data = pd.read_csv(os.path.join(data_dir, "uci/naval/data.txt"), delim_whitespace=True, header=None)
    X = data.to_numpy()[:, :-2]
    y_compressor = data.to_numpy()[:, -2]
    y_turbine = data.to_numpy()[:, -1]
    return X, y_turbine


def _load_protein(data_dir):
    """
    Physicochemical Properties of Protein Tertiary Structure Data Set
    Abstract: This is a data set of Physicochemical Properties of Protein Tertiary Structure.
    The data set is taken from CASP 5-9. There are 45730 decoys and size varying from 0 to 21 armstrong.

    Output variable:
        RMSD-Size of the residue.

    Input variables:
        F1 - Total surface area.
        F2 - Non polar exposed area.
        F3 - Fractional area of exposed non polar residue.
        F4 - Fractional area of exposed non polar part of residue.
        F5 - Molecular mass weighted exposed area.
        F6 - Average deviation from standard exposed area of residue.
        F7 - Euclidian distance.
        F8 - Secondary structure penalty.
        F9 - Spacial Distribution constraints (N,K Value).
    """
    data_file = os.path.join(data_dir, "uci/protein/CASP.csv")
    data = pd.read_csv(data_file, sep=",")
    X = data.to_numpy()[:, 1:]
    y = data.to_numpy()[:, 0]
    return X, y


def _load_crime(data_dir):
    data = pd.read_csv(os.path.join(data_dir, 'uci/crime/communities_processed.csv'))
    X = data.values[:, :-1].astype(float)
    y = data.values[:, -1].astype(float)
    return X, y


def _load_superconductivity(data_dir):
    data = pd.read_csv(os.path.join(data_dir, 'uci/superconductivity/train.csv'))
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    return X, y


def _load_mpg(data_dir):
    data = pd.read_csv(os.path.join(data_dir, 'uci/mpg/auto-mpg.data'), sep='\s+', header=None,
                       names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year',
                              'origin', 'car name'])
    data = data.replace('?',
                        np.nan)  # There are some missing data denoted as ?
    data = data.drop('car name', axis=1)  # Uninformative feature
    data = data.dropna(axis=0)

    # Transform the origin into one-hot encoding
    origin = pd.get_dummies(data.origin, prefix='origin')
    data = data.drop('origin', axis=1).join(origin)
    X = data.to_numpy().astype(np.float)[:, 1:]
    y = data.to_numpy().astype(np.float)[:, 0]
    return X, y


def _load_blog_feedback(data_dir):
    data = pd.read_csv(os.path.join(data_dir, 'uci/blog/blogData_train.csv'), header=None)
    X = data.to_numpy()[:, :280]
    y = data.to_numpy()[:, 280]
    # The log scale is more appropriate because the data is very skewed, this is the experiment setup in https://arxiv.org/pdf/1905.02928.pdf
    y = np.log(y + 1)
    return X, y


def _load_medical_expenditure(data_dir):
    """ Preprocess the medical expenditure dataset, the preprocessing is based on http://www.stat.uchicago.edu/~rina/jackknife/get_meps_data.ipynb"""
    reader = np.load(os.path.join(data_dir, 'medical-expenditure/meps_data.npz'))
    return reader['X'], reader['y']


def _load_forest_fires(data_dir):
    data = pd.read_csv(os.path.join(data_dir, 'uci/forest-fires/forestfires.csv'))
    data.isnull().values.any()
    month = pd.get_dummies(data.month, prefix='origin')
    data = data.drop(['month', 'day'], axis=1).join(month)

    y = data['area'].to_numpy().astype(np.float)
    y = np.log(1 + y)  # Because the dataset is skewed toward zero, transform it by log (1+x) (same as original paper)
    data = data.drop('area', axis=1)
    X = data.to_numpy()[:, :-1].astype(np.float)
    return X, y


def _load_facebook_comment1(data_dir):
    data = pd.read_csv(os.path.join(data_dir, 'uci/facebook/Features_Variant_1.csv'), header=None)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    y = np.log(1 + y)
    return X, y


def _load_cropyield(data_dir, crop_type):
    assert crop_type in ['Barley', 'Maize', 'Millet', 'Rice, paddy', 'Wheat', 'Sorghum']
    data = pd.read_csv(os.path.join(data_dir, 'cropyield/cropyield_processed.csv'))
    data = data[data["crop"] == crop_type]
    X = data.drop("crop", axis=1).values
    y = data["yield"].values
    return X, y


def _load_county_cropyield(data_dir, crop_type):
    assert crop_type in ['Wheat']
    data = pd.read_csv(os.path.join(data_dir, 'cropyield/county_cropyield_processed.csv'))
    data = data[data["crop"] == crop_type]
    data = data[data['Year'] != 1990] # Exclude first year for visualization purposes
    X = data.drop(["crop", "FIPS", "yield"], axis=1).values
    y = data["yield"].values
    return X, y


def _load_facebook_comment2(data_dir):
    data = pd.read_csv(os.path.join(data_dir, 'uci/facebook/Features_Variant_2.csv'), header=None)
    X = data.to_numpy()[:, :-1]
    y = data.to_numpy()[:, -1]
    y = np.log(1 + y)
    return X, y


regression_load_funs = {
    "blog": _load_blog_feedback,
    "boston": _load_boston,
    "concrete": _load_concrete,
    "crime": _load_crime,
    "energy-efficiency": _load_energy_efficiency,
    "fb-comment1": _load_facebook_comment1,
    "fb-comment2": _load_facebook_comment2,
    "forest-fires": _load_forest_fires,
    "kin8nm": _load_kin8nm,
    "medical-expenditure": _load_medical_expenditure,
    "mpg": _load_mpg,
    "naval": _load_naval,
    "power-plant": _load_powerplant,
    "protein": _load_protein,
    "superconductivity": _load_superconductivity,
    "wine": _load_wine,
    "yacht": _load_yacht,
    "county_crop_wheat": partial(_load_county_cropyield, crop_type="Wheat"),
    "crop_rice": partial(_load_cropyield, crop_type="Rice, paddy"),
    "crop_barley": partial(_load_cropyield, crop_type="Barley"),
    "crop_maize": partial(_load_cropyield, crop_type="Maize"),
    "crop_millet": partial(_load_cropyield, crop_type="Millet"),
    "crop_wheat": partial(_load_cropyield, crop_type="Wheat"),
    "crop_sorghum": partial(_load_cropyield, crop_type="Sorghum"),
}

regression_shapes = {
    "blog": (280, 1),
    "boston": (13, 1),
    "concrete": (8, 1),
    "crime": (102, 1),
    "energy-efficiency": (6, 1),
    "fb-comment1": (53, 1),
    "fb-comment2": (53, 1),
    "forest-fires": (21, 1),
    "kin8nm": (8, 1),
    "medical-expenditure": (107, 1),
    "mpg": (9, 1),
    "naval": (16, 1),
    "power-plant": (4, 1),
    "protein": (9, 1),
    "superconductivity": (81, 1),
    "wine": (11, 1),
    "yacht": (6, 1),
    "county_crop_wheat": (75, 1),
    "crop_rice": (7, 1),
    "crop_barley": (7, 1),
    "crop_maize": (7, 1),
    "crop_millet": (7, 1),
    "crop_wheat": (7, 1),
    "crop_sorghum": (7, 1),
}

# string = "{"
# for name, fun in regression_load_funs.items():
#     string += f'"{name}": ({fun("data")[0].shape[-1]}, 1), \n'
# string += "}"

if __name__ == "__main__":
    # data_module = RegressionDataModule("power-plant", data_dir="kernel-experiments/data")
    # data_module.setup()
    # train_loader = data_module.train_dataloader()

    data_module = RegressionDataModule("crime", data_dir="data", normalize=False)
    data_module.setup()
    next(iter(data_module.train_dataloader()))[0]

    # print(len(data_module.data_train) // 0.7)
