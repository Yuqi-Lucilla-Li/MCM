import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy
import functools
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import time
from lifelines import KaplanMeierFitter
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns




random_seed = 2025
torch.manual_seed(random_seed)
np.random.seed(random_seed)

H_value = 800
M_value = 1000

# --------------------------- Section 1: DGP --------------------------------
class DataSimulator(object):
    def __init__(self, beta_Z, beta_T1, beta_T0, sigma_1, sigma_0, beta_E1, beta_E0, H, M):
        self.beta_Z = beta_Z
        self.beta_T1 = beta_T1
        self.beta_T0 = beta_T0
        self.sigma_1 = sigma_1
        self.sigma_0 = sigma_0
        self.beta_E1 = beta_E1
        self.beta_E0 = beta_E0
        self.H = H
        self.M = M
    def sample_T(self, N, mu, sigma):
        upper_bound = (np.log(self.H) - mu) / sigma
        upper_bound_quantile = stats.norm.cdf(upper_bound)
        U = np.random.rand(N) * upper_bound_quantile
        return np.exp(stats.norm.ppf(U) * sigma + mu)
    def mean_time(self, mu, sigma):
        return np.exp(mu + sigma**2 / 2) * stats.norm.cdf((np.log(self.H) - mu - sigma**2) / sigma) / stats.norm.cdf((np.log(self.H) - mu) / sigma)
    def simulate(self, N, P_bin, P_cont):
        X = np.vstack(
            [np.ones(N)] + [stats.bernoulli.rvs(0.5, size = N) for i in range(P_bin)] + [stats.norm.rvs(size = N) for i in range(P_cont)]
        ).T
        Z = np.random.rand(N) < scipy.special.expit(X @ self.beta_Z)
        E1 = np.random.rand(N) < scipy.special.expit(X @ self.beta_E1)  # event probability
        E0 = np.random.rand(N) < scipy.special.expit(X @ self.beta_E0)
        C = np.random.rand(N) * self.M
        T1NC = self.sample_T(N, X @ self.beta_T1, self.sigma_1)
        T0NC = self.sample_T(N, X @ self.beta_T0, self.sigma_0)
        T1 = E1 * T1NC + (1 - E1) * self.M
        T0 = E0 * T0NC + (1 - E0) * self.M
        E = Z * E1 + (1 - Z) * E0
        T = Z * T1 + (1 - Z) * T0
        Y = functools.reduce(np.minimum, [T, C, self.H])
        S = (T < C) * (T < self.H)  # \delta, event indicator
        return {
            'X': X[:,1:],
            'Z': Z * 1,
            'E1': E1 * 1,
            'E0': E0 * 1,
            'E1_prob': scipy.special.expit(X @ self.beta_E1),
            'E0_prob': scipy.special.expit(X @ self.beta_E0),
            'T1': T1,
            'T0': T0,
            'T1_mean': self.mean_time(X @ self.beta_T1, self.sigma_1),
            'T0_mean': self.mean_time(X @ self.beta_T0, self.sigma_0),
            'T1_mean_lognorm': X @ self.beta_T1,
            'T0_mean_lognorm': X @ self.beta_T0,
            'C': C,
            'E': E,
            'T': T,
            'Y': Y,
            'S': S * 1,
            'ITE_cure_prob': scipy.special.expit(X @ self.beta_E0) - scipy.special.expit(X @ self.beta_E1),  # event(0) - event(1) = cure prob ITE
            'ITE_mean_time': self.mean_time(X @ self.beta_T1, self.sigma_1) - self.mean_time(X @ self.beta_T0, self.sigma_0)
        }


data_simulator = DataSimulator(
    beta_Z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]),  # 21
    beta_E1=np.array([-0.5, 0.5, -0.7, 0.5, -0.02475476, 0.05566316, 0.0277398, -0.16407168, -0.03701199,
                      0.08182949, -0.146312, -0.01942051, 0.02267608, 0.02247047,
                      0.01787867, 0.04254193, -0.09974565, -0.120529, 1.0, -0.8, 0.5]),  # 21
    beta_E0= np.array([0.5, 0.5, -0.2, 0.3, -0.14226733, -0.02932832,  0.01070134,  0.015169, -0.00839642,
                       0.01942336,  0.02702149, -0.00027359,  0.086567,  0.09919373,
                      0.01821063, -0.07707085,  0.12590933,  0.02647048, 1.0, -1.3, 0.4]),  # 21
    beta_T1 = np.array([3, 1, -0.2, -0.5, 0.8, -0.04759329,  0.06774229,  0.0749607, -0.00141549,  0.0545323 ,
       -0.00485808,  0.11326782,  0.05510337, -0.00372984,  0.04671354,
       -0.00964139,  0.0292803,  0.13637939, -0.5, 0.3, 0.5]),  # 21
    beta_T0 = np.array([2, 1, -0.2, -0.5, 0.8, -0.04759329,  0.06774229,  0.0749607, -0.00141549,  0.0545323 ,
       -0.00485808,  0.11326782,  0.05510337, -0.00372984,  0.04671354,
       -0.00964139,  0.0292803,  0.13637939, -0.5, 0.3, 0.5]),  # 21
    sigma_1 = 1,
    sigma_0 = 1,
    H = H_value,
    M = M_value
)

# paras = np.random.normal(loc=0, scale=np.sqrt(0.005), size=14)


data = data_simulator.simulate(10000, 10, 10)

print("end")

X_df = pd.DataFrame(data['X'], columns=[f'X_{i}' for i in range(data['X'].shape[1])])
data_dict = {k: v for k, v in data.items() if k != 'X'}
df = pd.DataFrame(data_dict)
data_df = pd.concat([df, X_df], axis=1)


# ----------------- Dataset Description ------------
data_df['ITE_cure_prob'].mean()



df_train, df_test = train_test_split(data_df, test_size=0.65, random_state=random_seed)
# df_val, df_test = train_test_split(df_temp, test_size=50/65, random_state=random_seed)

df_train.to_csv("mcm_data.csv")
# ----------------------- Section 2: Feature Selection ----------------------
from scipy.stats import kendalltau

X_features = [f'X_{i}' for i in range(20)]
Y_train = df_train['Y']

##### here we can compare with a feature selection approach ########
# weight_array = np.zeros(len(X_features))
#
# for i, feature in enumerate(X_features):
#     tau, p_value = kendalltau(df_train[feature], Y_train)
#     if p_value < 0.05:
#         weight_array[i] = 1
#
#
# print(weight_array)


# ---------------------------- Matching ---------------------------------------------

def Effect_Estimation(match_S, match_Y, time_horizon):
    probabilities = []
    expected_mean = []
    race_mean = []
    kmf = KaplanMeierFitter()

    for i in range(match_S.shape[0]):
        event_indicators = match_S[i]
        survival_times = match_Y[i]

        kmf.fit(survival_times, event_observed=event_indicators)
        S_H = kmf.survival_function_at_times(time_horizon).values[0]
        probabilities.append(S_H)

        time_points = kmf.survival_function_.index
        survival_probs = kmf.survival_function_['KM_estimate'].values
        valid_indices = time_points <= time_horizon
        time_points = time_points[valid_indices]
        survival_probs = survival_probs[valid_indices]

        A = np.trapz(survival_probs, time_points)
        if S_H < 1.0:
            cond_mean = (A - S_H * time_horizon) / (1 - S_H)
        else:
            cond_mean = np.nan

        expected_mean.append(cond_mean)
        race_mean.append(A)

    return np.array(probabilities), np.array(expected_mean), np.array(race_mean)


def matching_and_estimation(features, weights, treated_indices, control_indices, S_test, Y_test, H, k,  labels=None):
    nn_control = NearestNeighbors(n_neighbors=k, metric='minkowski', metric_params={'w': weights}, p=2)
    nn_treat = NearestNeighbors(n_neighbors=k, metric='minkowski', metric_params={'w': weights}, p=2)

    treated_features = features[treated_indices]
    control_features = features[control_indices]
    nn_control.fit(control_features)
    nn_treat.fit(treated_features)

    distances_ct, indices_ct = nn_control.kneighbors(treated_features)
    distances_tc, indices_tc = nn_treat.kneighbors(control_features)
    distances_tt, indices_tt = nn_treat.kneighbors(treated_features)
    distances_cc, indices_cc = nn_control.kneighbors(control_features)

    # Map local indices to global indices
    control_indices_for_treat = control_indices[indices_ct]
    treat_indices_for_control = treated_indices[indices_tc]
    treat_indices_for_treat = treated_indices[indices_tt]
    control_indices_for_control = control_indices[indices_cc]

    # Prepare data for Kaplan-Meier estimation
    treat_control_S = np.array([S_test[idx] for idx in treat_indices_for_control])
    control_treat_S = np.array([S_test[idx] for idx in control_indices_for_treat])
    treat_treat_S = np.array([S_test[idx] for idx in treat_indices_for_treat])
    control_control_S = np.array([S_test[idx] for idx in control_indices_for_control])

    # Perform effect estimation
    probs_treat_control, means_treat_control, races_treat_control = Effect_Estimation(treat_control_S, Y_test[treat_indices_for_control], H)
    probs_control_treat, means_control_treat, races_control_treat = Effect_Estimation(control_treat_S, Y_test[control_indices_for_treat], H)
    probs_treat_treat, means_treat_treat, races_treat_treat = Effect_Estimation(treat_treat_S, Y_test[treat_indices_for_treat], H)
    probs_control_control, means_control_control, races_control_control = Effect_Estimation(control_control_S,Y_test[control_indices_for_control],H)

    # Compile results
    probs = {
        "treat_control": probs_treat_control,
        "control_treat": probs_control_treat,
        "treat_treat": probs_treat_treat,
        "control_control": probs_control_control
    }
    means = {
        "treat_control": means_treat_control,
        "control_treat": means_control_treat,
        "treat_treat": means_treat_treat,
        "control_control": means_control_control
    }
    races = {
        "treat_control": races_treat_control,
        "control_treat": races_control_treat,
        "treat_treat": races_treat_treat,
        "control_control": races_control_control
    }

    return probs, means, races


# to compare if patients in a matching group have closer ground truth
def knn_matching_treated(df, X_features, label, k=30, weights=None):

    df_treated = df[df['Z'] == label].reset_index(drop=True)
    features_treated = df_treated[X_features].values  

    nn_euclidean = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_euclidean.fit(features_treated)

    if weights is not None:
        nn_weighted = NearestNeighbors(n_neighbors=k, metric='minkowski', metric_params={'w': weights}, p=2)
        nn_weighted.fit(features_treated)

    def get_matching_stats(nn_model):

        matched_T1 = []
        matched_E1 = []

        for i in range(len(df_treated)):
            _, neighbors = nn_model.kneighbors(features_treated[i].reshape(1, -1), n_neighbors=k)
            matched_group = df_treated.iloc[neighbors[0]]

            matched_T1.append(matched_group['T1_mean'].mean())
            matched_E1.append(matched_group['E1_prob'].mean())

        return np.array(matched_T1), np.array(matched_E1)

    matched_T1_euclidean, matched_E1_euclidean = get_matching_stats(nn_euclidean)

    if weights is not None:
        matched_T1_weighted, matched_E1_weighted = get_matching_stats(nn_weighted)
    else:
        matched_T1_weighted = matched_E1_weighted = None

    df_treated['T1_diff_euclidean'] = df_treated['T1_mean'] - matched_T1_euclidean
    df_treated['E1_diff_euclidean'] = df_treated['E1_prob'] - matched_E1_euclidean

    if weights is not None:
        df_treated['T1_diff_weighted'] = df_treated['T1_mean'] - matched_T1_weighted
        df_treated['E1_diff_weighted'] = df_treated['E1_prob'] - matched_E1_weighted

    return df_treated


# df_matched = knn_matching_treated(df_test, X_features, 1, k=30, weights=beta_weight)
#
# columns_of_interest = ['T1_mean', 'E1_prob', 'T1_diff_weighted', 'E1_diff_weighted', 'T1_diff_euclidean', 'E1_diff_euclidean']
# df_matched_filtered = df_matched[columns_of_interest]
#
# plt.figure(figsize=(8, 5))
# sns.kdeplot(df_matched_filtered['T1_diff_weighted'], fill=True, label='T1_diff_weighted Density')
# sns.kdeplot(df_matched_filtered['T1_diff_euclidean'], fill=True, label='T1_diff_euclidean Density')
# plt.xlabel('Event probability')
# plt.ylabel('Density')
# plt.title('Density Plot beta weight')
# plt.legend()
# plt.grid()
# plt.show()


# ---------- Matching Process ------------------
df_test = df_test.reset_index(drop=True)
features = df_test[X_features].values
labels = df_test['Z'].values  # Assuming 'Z' is the treatment indicator
original_indices = df_test.index.values

treated_indices = np.where(labels == 1)[0]
control_indices = np.where(labels == 0)[0]

S_test = df_test['S'].values
Y_test = df_test['Y'].values

beta_1 = np.array([0.385, -0.598, 0.298, 0.063, -0.065, 0.1, -0.151, 0.113, 0.395, -0.111, -0.014, 0.055, 0.015, -0.007, 0.008, -0.224, -0.119, 0.963, -0.616, 0.54])
beta_0 = np.array([0.36, -0.102, 0.26, -0.018, 0.101, -0.045, -0.087, 0.221, 0.104, 0.071, 0.004, 0.083, 0.058, -0.051, -0.152, 0.092, 0.051, 1.044, -1.217, 0.455])
gamma_1 = np.array([1, -0.316, -0.43, 0.821, -0.082, 0.101, 0.06, -0.052, 0.065, -0.032, 0.144, 0.063, 0.004, 0.019, 0.022, -0.024, 0.157, -0.438, 0.303, 0.526])
gamma_0 = np.array([1.092, -0.245, -0.488, 0.693, -0.052, 0.019, 0.044, 0.058, 0.022, -0.085, 0.126, 0.047, 0.009, 0.069, 0.002, 0.001, 0.12, -0.474, 0.267, 0.493])

beta_weight = (np.abs(beta_1) + np.abs(beta_0)) / 2
gamma_weight = (np.abs(gamma_1) + np.abs(gamma_0)) / 2

beta_weight_normalized = beta_weight / np.linalg.norm(beta_weight, ord=2)
gamma_weight_normalized = gamma_weight / np.linalg.norm(gamma_weight, ord=2)

combined_weight = (beta_weight_normalized + gamma_weight_normalized) / 2
equal_weight = np.ones(20)


# ------------ Compare matching approaches -------------------
weight_list = [beta_weight, gamma_weight, combined_weight, equal_weight]
weight_name = ['cure_weight', 'time_weight', 'combined_weight', 'Euclidean']

df_compare_cure = df_test[['ITE_cure_prob']].copy()
df_compare_time = df_test[['ITE_mean_time']].copy()

list_cure_diff = []
list_time_diff = []

for i in range(len(weight_list)):
    weight = weight_list[i]
    name = weight_name[i]
    print(name)

    probs, means, races = matching_and_estimation(
        features=features,
        treated_indices=treated_indices,
        control_indices=control_indices,
        S_test=S_test,
        Y_test=Y_test,
        H=H_value,
        k=25,
        weights=weight,
        labels=labels
    )


    # ----------------- Result 1: Cure probability ------------------------
    esti_cure = np.concatenate((probs['treat_treat'], probs['control_control']))
    esti_cure_cf = np.concatenate((probs['control_treat'], probs['treat_control']))
    cure_indices = np.concatenate((treated_indices, control_indices))
    wknn_cure_esti = np.column_stack((cure_indices, esti_cure, esti_cure_cf))
    wknn_cure_esti = wknn_cure_esti[wknn_cure_esti[:, 0].argsort()]

    wknn_cure_esti = np.column_stack((wknn_cure_esti, df_test['ITE_cure_prob'].values))

    # Summarize the estimated results
    column_names = ['Index', 'Estimated_cure_prob', 'Estimated_CF_cure_prob', 'ITE_cure_prob']
    df_wknn_cure_esti = pd.DataFrame(wknn_cure_esti, columns=column_names)

    df_wknn_cure_esti['Estimated_cure_ITE'] = np.nan

    df_wknn_cure_esti.loc[treated_indices, 'Estimated_cure_ITE'] = (
        df_wknn_cure_esti.loc[treated_indices, 'Estimated_cure_prob'] -
        df_wknn_cure_esti.loc[treated_indices, 'Estimated_CF_cure_prob']
    )

    df_wknn_cure_esti.loc[control_indices, 'Estimated_cure_ITE'] = (
        df_wknn_cure_esti.loc[control_indices, 'Estimated_CF_cure_prob'] -
        df_wknn_cure_esti.loc[control_indices, 'Estimated_cure_prob']
    )

    ite_differences = df_wknn_cure_esti['Estimated_cure_ITE'] - df_wknn_cure_esti['ITE_cure_prob']
    ite_diff_abs = ite_differences.abs()
    ite_mae = ite_diff_abs.mean()
    ite_mae_std = ite_diff_abs.std()

    print('Cure Effect')
    print(f"ITE MAE: {ite_mae:.4f}")
    print(f"ITE Std: {ite_mae_std:.4f}")

    # --------------------------------- Result 2: Time Effect ----------------------
    esti_time = np.concatenate((means['treat_treat'], means['control_control']))
    esti_time_cf = np.concatenate((means['control_treat'], means['treat_control']))
    time_indices = np.concatenate((treated_indices, control_indices))
    wknn_time_esti = np.column_stack((time_indices, esti_time, esti_time_cf))
    wknn_time_esti = wknn_time_esti[wknn_time_esti[:, 0].argsort()]


    wknn_time_esti = np.column_stack((wknn_time_esti, df_test['ITE_mean_time'].values))

    column_names_time = ['Index', 'Estimated_time', 'Estimated_CF_time', 'ITE_mean_time']
    df_wknn_time_esti = pd.DataFrame(wknn_time_esti, columns=column_names_time)


    df_wknn_time_esti['Estimated_time_ITE'] = np.nan

    df_wknn_time_esti.loc[treated_indices, 'Estimated_time_ITE'] = (
        df_wknn_time_esti.loc[treated_indices, 'Estimated_time'] -
        df_wknn_time_esti.loc[treated_indices, 'Estimated_CF_time']
    )

    df_wknn_time_esti.loc[control_indices, 'Estimated_time_ITE'] = (
        df_wknn_time_esti.loc[control_indices, 'Estimated_CF_time'] -
        df_wknn_time_esti.loc[control_indices, 'Estimated_time']
    )

    ite_time_differences = df_wknn_time_esti['Estimated_time_ITE'] - df_wknn_time_esti['ITE_mean_time']
    ite_time_abs = ite_time_differences.abs()
    ite_time_mae = ite_time_abs.mean()
    ite_time_mae_std = ite_time_abs.std()

    print('Time Effect')
    print(f"ITE MAE: {ite_time_mae:.4f}")
    print(f"ITE bias Std: {ite_time_mae_std:.4f}")


    # -------- compare results --------
    df_compare_cure[name] = df_wknn_cure_esti['Estimated_cure_ITE'].copy()
    df_compare_time[name] = df_wknn_time_esti['Estimated_time_ITE'].copy()

    list_cure_diff.append(ite_differences)
    list_time_diff.append(ite_time_differences)

    print('end')





plt.figure(figsize=(10, 6))
legend_labels = []

for col in df_compare_time.columns:
    sns.kdeplot(df_compare_time[col], label=col)

    mean_val = df_compare_time[col].mean()
    std_val = df_compare_time[col].std()

    legend_labels.append(f"{col} (μ={mean_val:.3f}, σ={std_val:.3f})")

plt.xlim(-150,200)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Density Plot of Estimated time ITE with Mean and Std")

plt.legend(legend_labels, loc='upper left')
plt.grid()
plt.show()
