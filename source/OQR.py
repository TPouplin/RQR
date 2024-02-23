import torch


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)

def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x).float()
    L = GaussianKernelMatrix(y, s_y).float()
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().to(K.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC



def compute_coverages_and_avg_interval_len(y, y_lower, y_upper):
    """
    computes prediction interval length and the coverage indicators.
    Parameters
    ----------
    y - n response variables
    y_lower - n upper bounds of the intervals that should contain y with a pre-specified probability
    y_upper - n lower bounds of the intervals that should contain y with a pre-specified probability

    Returns
    -------
    interval_lengths - a vector of size n that contains the length of each interval
    coverage_indicators - a vector of size n that contains a differential indicator that equals (approximately) 1
                            if the interval contains y, and (approximately) 0, otherwise.
    """

    interval_lengths = (y_upper - y_lower)

    coverage_indicators = torch.nn.Tanh()(50 * torch.min(y - y_lower, y_upper - y))

    coverage_indicators = (coverage_indicators + 1) / 2

    return coverage_indicators, interval_lengths


def pearsons_corr2d(x, y):
    """

    Parameters
    ----------
    x - matrix of size n_vectors x n_elements
    y - matrix of size n_vectors x n_elements

    Returns
    -------
    A vector of size n_vectors where the i-th element is the correlation between x[i, :], y[i,:]
    """
    vx = x - torch.mean(x, dim=1).reshape(len(x), 1)
    vy = y - torch.mean(y, dim=1).reshape(len(x), 1)

    return torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)))


def independence_penalty(y, pred_l, pred_u, pearsons_corr_multiplier=1, hsic_multiplier=0, y_multiplier=100):

    """
    Computes the independence penalty given the true label and the prediced upper and lower quantiles.
    Parameters
    ----------
    y - the true label of a feature vector.
    pred_l - the predicted lower bound
    pred_u - the prediced upper bound
    pearsons_corr_multiplier - multiplier of R_corr
    hsic_multiplier - multiplier of R_HSIC
    y_multiplier - multiplier of y for numeric stability

    Returns
    -------
    The independence penalty R
    """

    if pearsons_corr_multiplier == 0 and hsic_multiplier == 0:
        return 0
    
    is_in_interval, interval_sizes = compute_coverages_and_avg_interval_len(y.view(-1) * y_multiplier,
                                                                            pred_l * y_multiplier,
                                                                            pred_u * y_multiplier)
    

    # print("is_in_interval", is_in_interval.shape)
    # print("interval_sizes", interval_sizes.shape)
    # print("y", y.shape)
    
    # print("pred_l", pred_l.shape)
    # print("pred_u", pred_u.shape)
    
    partial_interval_sizes = interval_sizes[abs(torch.min(is_in_interval, dim=1)[0] -
                                                torch.max(is_in_interval, dim=1)[0]) > 0.05, :]
    partial_is_in_interval = is_in_interval[abs(torch.min(is_in_interval, dim=1)[0] -
                                                torch.max(is_in_interval, dim=1)[0]) > 0.05, :]
    
    
    if partial_interval_sizes.shape[0] > 0 and pearsons_corr_multiplier != 0:
        corrs = pearsons_corr2d(partial_interval_sizes, partial_is_in_interval)
        pearsons_corr_loss = torch.mean((torch.abs(corrs)))
        if pearsons_corr_loss.isnan().item():
            pearsons_corr_loss = 0
    else:
        pearsons_corr_loss = 0
    
    hsic_loss = 0
    if partial_interval_sizes.shape[0] > 0 and hsic_multiplier != 0:
        n = partial_is_in_interval.shape[1]
        data_size_for_hsic = 512
        for i in range(partial_is_in_interval.shape[0]):

            v = partial_is_in_interval[i, :].reshape((n,1))
            l = partial_interval_sizes[i, :].reshape((n,1))
            v = v[:data_size_for_hsic]
            l = l[:data_size_for_hsic]
            if torch.max(v) - torch.min(v) > 0.05:  # in order to not get hsic = 0
                curr_hsic = torch.abs(torch.sqrt(HSIC(v, l)))
            else:
                curr_hsic = 0

            hsic_loss += curr_hsic
        hsic_loss = hsic_loss / partial_interval_sizes.shape[0]
    
    penalty = pearsons_corr_loss * pearsons_corr_multiplier + hsic_loss * hsic_multiplier

    return penalty