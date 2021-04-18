import numpy as np
from itertools import combinations
from numba import njit
from numba import prange
from numba import vectorize
import matplotlib.pyplot as plt

class MiniRocketImage():
    """使用 MINIRocket 转换原始信号为许多特征值，使用这些特征值构建一张特征图
    """
    def __init__(self, num_features=10000, max_dilations_per_kernel=32, random_state=None):
        """num_features=10000, max_dilations_per_kernel=32, mode='fixed', random_state=None
        """
        super(MiniRocketImage, self).__init__()
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = (
            np.int32(random_state) if isinstance(random_state, int) else None
        )
        self.num_kernels = 84
        self.indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype=np.int32)
    
    def fit(self, X, y=None):
        """Fits dilations and biases to input time series.
        X: array_like, shape:(n_instances, channels, n_timepoints)
        """
        X = X[:, 0, :].astype(np.float32)
        _, n_timepoints = X.shape # for soil disturbance, 96

        self.parameters = _fit(X, self.indices, n_timepoints, self.num_kernels, self.num_features, \
            self.max_dilations_per_kernel, self.random_state)
        return self
    
    def transform(self, X):
        """Transforms input time series.
        """
        X = X[:, 0, :].astype(np.float32)
        return _transform(X, self.indices, self.parameters)
    
    def fit_transform(self, X):
        X = X[:, 0, :].astype(np.float32)
        _, n_timepoints = X.shape # for soil disturbance, 96
        self.parameters = _fit(X, self.indices, n_timepoints, self.num_kernels, self.num_features, \
            self.max_dilations_per_kernel, self.random_state)
        return _transform(X, self.indices, self.parameters)

def _fit(X, indices, n_timepoints, num_kernels, num_features, max_dilations_per_kernel, random_state):
    dilations, num_features_per_dilation = _fit_dilation(n_timepoints, num_kernels, num_features, max_dilations_per_kernel)
    biases = _fit_biases(X, indices, dilations, num_features_per_dilation, random_state)
    return dilations, num_features_per_dilation, biases

def _quantiles(n):
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )

def _fit_dilation(n_timepoints, num_kernels, num_features, max_dilations_per_kernel):
    """Fits dilations per kernel & features per dilations
    """
    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    ) # 每个 kernel 实际上最多只能对应 true_max_dilations_per_kernel 个dilations

    multiplier = num_features_per_kernel / true_max_dilations_per_kernel 
    # 实际每个 kernel 需要的 feature 数与最多 dilations 数的比值

    max_exponent = np.log2((n_timepoints - 1) / (9 - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    ) # 计算 dilations 与其对应的 feature 数
        
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
        np.int32
    )  # 与以下的 remainder 一起，使得每个 dilations 对应的 feature 数加起来达到期望的数目（10000）

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


@njit(
    "Tuple((float32[:],float32[:,:]))(float32[:],float32[:],int32,int32,int32)",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _transform_without_bias(A, G, dilation, padding, n_timepoints):
    """不适用 bias 计算卷积结果
    """
    # A = -_X  # A = alpha * X = -X
    # G = _X + _X + _X  # G = gamma * X = 3X

    C_alpha = np.zeros(n_timepoints, dtype=np.float32)
    C_alpha[:] = A

    C_gamma = np.zeros((9, n_timepoints), dtype=np.float32)
    C_gamma[9 // 2] = G

    start = dilation
    end = n_timepoints - padding

    for gamma_index in range(9 // 2):

        C_alpha[-end:] = C_alpha[-end:] + A[:end]
        C_gamma[gamma_index, -end:] = G[:end]

        end += dilation

    for gamma_index in range(9 // 2 + 1, 9):

        C_alpha[:-start] = C_alpha[:-start] + A[start:]
        C_gamma[gamma_index, :-start] = G[start:]

        start += dilation

    # C = C_alpha + C_gamma[index[0]] + C_gamma[index[1]] + C_gamma[index[2]]
    return C_alpha, C_gamma


@njit(
    "float32[:](float32[:,:],int32[:,:],int32[:],int32[:],optional(int32))",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases(X, indices, dilations, num_features_per_dilation, seed):

    if seed is not None:
        np.random.seed(seed)

    n_instances, n_timepoints = X.shape
    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0
    max_match = -np.inf
    min_match = np.inf

    for dilation_index in range(num_dilations): # 为每个 dilation 分配特征

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2
        num_features_this_dilation = num_features_per_dilation[dilation_index]
        
        for kernel_index in range(num_kernels):
            for example_index in range(n_instances):
                _X = X[example_index]

                A = -_X  # A = alpha * X = -X
                G = _X + _X + _X  # G = gamma * X = 3X
                C_alpha, C_gamma = _transform_without_bias(A, G, dilation, padding, n_timepoints)
                
                index_0, index_1, index_2 = indices[kernel_index]
                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
                
                max_match = max(C) if max(C) > max_match else max_match
                min_match = min(C) if min(C) < min_match else min_match

        fixed_min_match = min_match + 0.25 * (max_match-min_match)
        fixed_max_match = min_match + 0.75 * (max_match-min_match)

        for kernel_index in range(num_kernels):
            feature_index_end = feature_index_start + num_features_this_dilation
            biases[feature_index_start:feature_index_end] = np.linspace(fixed_min_match, fixed_max_match, \
                num_features_this_dilation)

            feature_index_start = feature_index_end

    return biases


@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0


@njit(
    "Tuple((float32[:,:],float32[:,:,:]))(float32[:,:],int32[:,:],Tuple((int32[:],int32[:],float32[:])))",
    fastmath=True,
    parallel=True,
    cache=True,
)
def _transform(X, indices, parameters):

    n_instances, n_timepoints = X.shape
    dilations, num_features_per_dilation, biases = parameters
    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((n_instances, num_features), dtype=np.float32)
    fimages = np.zeros((n_instances, num_kernels, np.sum(num_features_per_dilation)), dtype=np.float32)

    for example_index in prange(n_instances):

        _X = X[example_index]

        A = -_X  # A = alpha * X = -X
        G = _X + _X + _X  # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]
            num_features_former_dilation = np.sum(num_features_per_dilation[:dilation_index])
            
            C_alpha, C_gamma = _transform_without_bias(A, G, dilation, padding, n_timepoints)

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        ppv = _PPV(C, biases[feature_index_start + feature_count]).mean()
                        features[example_index, feature_index_start + feature_count] = ppv
                        fimages[example_index, kernel_index, num_features_former_dilation + feature_count] = ppv
                else:
                    for feature_count in range(num_features_this_dilation):
                        ppv = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()
                        features[example_index, feature_index_start + feature_count] = ppv
                        fimages[example_index, kernel_index, num_features_former_dilation + feature_count] = ppv

                feature_index_start = feature_index_end

    return features, fimages


if __name__ == '__main__':
    x = np.random.rand(100, 1, 96).astype(np.float32)
    rocket = MiniRocketImage(num_features=10000, max_dilations_per_kernel=32)
    rocket.fit(x)
    y1, y2 = rocket.transform(x)
    print(y1.shape)
    print(y2[0])
    plt.figure(figsize=(5, 5))
    plt.imshow(y2[0], cmap='rainbow', origin='lower')
    plt.title('MiniRocket Plot', fontsize=16)
    plt.tight_layout()
    plt.show()
