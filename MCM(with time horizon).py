import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors
from torch.distributions.normal import Normal
from lifelines import KaplanMeierFitter
from sklearn.neighbors import NearestNeighbors


random_seed = 2024
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# hyperparameters
n_samples = 20000
epochs = 10000
learning_rate = 0.001
t = 80
k = 25 # number of neighbors
polynomial_degree = 1

# --------------------------- Section 1: Data Generating Process -----------------------
def generate_features(n_samples, n_features=20):
    binary_features = np.random.binomial(1, 0.5, size=(n_samples, n_features // 2))
    normal_features = np.random.randn(n_samples, n_features // 2)
    return np.hstack((binary_features, normal_features))


def assign_treatment(X1):
    p_Z = expit(0 + X1)
    Z = np.random.binomial(1, p_Z, size=(X1.shape[0], 1))
    return Z


def cure_prob_treat(X, beta):
    X_poly = np.hstack((np.ones((X.shape[0], 1)), X))
    P_cure = 1 / (1 + np.exp(-X_poly @ beta))
    return P_cure


def cure_prob_control(X, beta):
    # non-linear
    # baseline + covariates
    X_poly = np.hstack((np.ones((X.shape[0], 1)), X ** polynomial_degree))
    P_cure = 1 / (1 + np.exp(-X_poly @ beta))
    return P_cure


n_features = 20
X = generate_features(n_samples, n_features)
X1 = X[:, 0].reshape(-1, 1)
Z = assign_treatment(X1)
X_combined = np.hstack((Z, X))
feature_names = ['Z'] + [f'X{i + 1}' for i in range(n_features)]

beta_treated = np.zeros(n_features + 1)
beta_control = np.zeros(n_features + 1)

norm_idx = n_features // 2 + 1

beta_treated[0] = 2.0  # intercept
beta_treated[1] = 1.5
beta_treated[2] = -1
beta_treated[3] = 0.8
# beta_treated[norm_idx] = 0.5
# beta_treated[norm_idx+1] = -0.3

beta_control[0] = 1.0  # intercept
beta_control[1] = 1.0
beta_control[2] = -2
beta_control[3] = 0.3
# beta_control[norm_idx] = 0.3
# beta_control[norm_idx+1] = -0.8

# 1. Cure probability
# only input covariates
P_cure_treated = cure_prob_treat(X_combined[Z.flatten() == 1][:, 1:], beta_treated)
P_cure_control = cure_prob_control(X_combined[Z.flatten() == 0][:, 1:], beta_control)
P_cure = np.zeros(n_samples)
P_cure[Z.flatten() == 1] = P_cure_treated.flatten()
P_cure[Z.flatten() == 0] = P_cure_control.flatten()

cure_status = np.random.binomial(1, P_cure)

# 2. Survival time
gamma_treat = np.zeros(n_features + 1)
gamma_control = np.zeros(n_features + 1)

# Define gamma for treated group
gamma_treat[0] = 1.0  # intercept for treated group
gamma_treat[1] = -0.2
gamma_treat[2] = 0.6
gamma_treat[norm_idx + 3] = 0.4
gamma_treat[norm_idx + 4] = -0.5

# Define gamma for control group
gamma_control[0] = 0.7 # intercept for control group
gamma_control[1] = -0.3
gamma_control[2] = 0.2
gamma_control[norm_idx + 3] = 0.2
gamma_control[norm_idx + 4] = -0.2

# Assign survival time based on treatment group
mu_treat = (4.0 + X_combined[Z.flatten() == 1] @ gamma_treat)
mu_control = (4.0 + X_combined[Z.flatten() == 0] @ gamma_control)
mu = np.zeros(n_samples)
mu[Z.flatten() == 1] = mu_treat
mu[Z.flatten() == 0] = mu_control


sigma = 0.5

log_survival_times_treat = np.random.normal(mu_treat, sigma)
log_survival_times_control = np.random.normal(mu_control, sigma)

# Combine survival times for all patients
log_survival_times = np.zeros(n_samples)
log_survival_times[Z.flatten() == 1] = log_survival_times_treat
log_survival_times[Z.flatten() == 0] = log_survival_times_control

survival_times = np.exp(log_survival_times)

plt.figure(figsize=(8, 5))
sns.kdeplot(survival_times, bw_adjust=0.5)
plt.title("Density Plot of Survival Times")
plt.xlabel("Survival Times")
plt.ylabel("Density")
plt.show()

# time horizon
b = np.percentile(survival_times, 90)
survival_times[cure_status == 1] = np.inf


# check the consistency: for non-cured patients, event occurs before time horizon
while True:
    # Identify inconsistent indices for the entire dataset
    inconsistent_indices = (cure_status == 0) & (survival_times > b)

    # Split inconsistent indices into treat and control groups
    inconsistent_indices_treat = inconsistent_indices & (Z.flatten() == 1)
    inconsistent_indices_control = inconsistent_indices & (Z.flatten() == 0)

    if not (inconsistent_indices_treat.any() or inconsistent_indices_control.any()):
        break

    # Resample survival times for the treated group
    if inconsistent_indices_treat.any():
        # Map inconsistent indices to treat group subset
        treat_group_indices = np.where(Z.flatten() == 1)[0]
        resample_indices_treat = np.isin(treat_group_indices, np.where(inconsistent_indices_treat)[0])
        log_survival_times_resampled_treat = np.random.normal(
            mu_treat[resample_indices_treat], sigma
        )
        survival_times[inconsistent_indices_treat] = np.exp(log_survival_times_resampled_treat)

    # Resample survival times for the control group
    if inconsistent_indices_control.any():
        # Map inconsistent indices to control group subset
        control_group_indices = np.where(Z.flatten() == 0)[0]
        resample_indices_control = np.isin(control_group_indices, np.where(inconsistent_indices_control)[0])
        log_survival_times_resampled_control = np.random.normal(
            mu_control[resample_indices_control], sigma
        )
        survival_times[inconsistent_indices_control] = np.exp(log_survival_times_resampled_control)


# 3. Censoring time
censoring_times = np.random.uniform(0, b, size=n_samples)

# S=1 indicates failure
Y = np.where(cure_status == 1, censoring_times, np.minimum(survival_times, censoring_times))
S = np.where((cure_status == 1) | ((cure_status == 0) & (censoring_times < survival_times)), 0, 1)


# Counterfactual Features --------
Z_cf = 1 - Z
X_cf_combined = np.hstack((Z_cf, X))

P_cure_cf_treated = cure_prob_treat(X_cf_combined[Z.flatten() == 0][:, 1:], beta_treated)
P_cure_cf_control = cure_prob_control(X_cf_combined[Z.flatten() == 1][:, 1:], beta_control)
P_cure_cf = np.zeros(n_samples)
P_cure_cf[Z.flatten() == 0] = P_cure_cf_treated.flatten()
P_cure_cf[Z.flatten() == 1] = P_cure_cf_control.flatten()

mu_cf = np.zeros(n_samples)
mu_cf[Z_cf.flatten() == 0] = 4.0 + X_cf_combined[Z_cf.flatten() == 0] @ gamma_control
mu_cf[Z_cf.flatten() == 1] = 4.0 + X_cf_combined[Z_cf.flatten() == 1] @ gamma_treat



data = {name: X_combined[:, i + 1] for i, name in enumerate(feature_names[1:])}
data.update({
    'Z': X_combined[:, 0],
    'Z_cf': Z_cf.flatten(),
    'cure status': cure_status,
    'E': 1 - cure_status,  # E=1 indicates event
    'Survival Time': survival_times,
    'Censoring Time': censoring_times,
    'Y': Y,
    'S': S,
    'P_cure': P_cure,
    'P_cure_cf': P_cure_cf,
    'mu': mu,
    'mu_cf': mu_cf
})

df = pd.DataFrame(data)
df_untreated = df[df['Z'] == 0].drop(columns=['Z'])
df_treated = df[df['Z'] == 1].drop(columns=['Z'])

df_S1 = df[df['S'] == 1]
Y_S1 = df_S1['Y'].values
b_esti = Y_S1.max()
print(f"Estimated b (b_esti): {b_esti:.2f}")


# Ground truth
df['cure_ITE'] = 0  # Initialize the column
df.loc[df['Z'] == 1, 'cure_ITE'] = df.loc[df['Z'] == 1, 'P_cure'] - df.loc[df['Z'] == 1, 'P_cure_cf']
df.loc[df['Z'] == 0, 'cure_ITE'] = df.loc[df['Z'] == 0, 'P_cure_cf'] - df.loc[df['Z'] == 0, 'P_cure']
cure_ATE = df['cure_ITE'].mean()

print("cure ATE:")
print(round(cure_ATE, 3))

# consider exp(mu) or exp(mu+sigma^2) as the truth?
df['time_ITE'] = 0  # Initialize the column
df.loc[df['Z'] == 1, 'time_ITE'] = df.loc[df['Z'] == 1, 'mu'] - df.loc[df['Z'] == 1, 'mu_cf']
df.loc[df['Z'] == 0, 'time_ITE'] = df.loc[df['Z'] == 0, 'mu_cf'] - df.loc[df['Z'] == 0, 'mu']


plt.figure(figsize=(8, 5))
sns.kdeplot(df_untreated['Survival Time'], bw_adjust=0.5, label='Control Group', shade=True)
sns.kdeplot(df_treated['Survival Time'], bw_adjust=0.5, label='Treated Group', shade=True)

plt.title("Density Plot of Survival Times: Treated vs Control Groups")
plt.xlabel("Survival Time")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

print("finish DGP")

# ----------------------------- Section 2: Mixture cure Model ----------------------------
# Define the neural network models and loss functions
class CureModel(nn.Module):
    def __init__(self, input_dim):
        super(CureModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class AFTModel(nn.Module):
    def __init__(self, input_dim):
        super(AFTModel, self).__init__()
        self.linear_mu = nn.Linear(input_dim, 1)
        self.sigma = nn.Parameter(torch.zeros(1) + 0.1)

    def forward(self, x):
        mu = self.linear_mu(x)
        std = torch.exp(self.sigma)
        return mu, std


def compute_survival_functions(mu, sigma, y):
    log_y = torch.log(y)
    normal_dist = Normal(mu, sigma)
    F_y = 1 - normal_dist.cdf(log_y)
    f_y = torch.exp(normal_dist.log_prob(log_y)) / y
    return F_y, f_y


class CombinedLoss(nn.Module):
    def __init__(self, cure_model, survival_model, b_esti, l2_reg=1e-4):
        super(CombinedLoss, self).__init__()
        self.cure_model = cure_model
        self.survival_model = survival_model
        self.eps = 1e-8
        self.l2_reg = l2_reg

    def forward(self, x, s, y):
        cure_prob = self.cure_model(x)
        mu, sigma = self.survival_model(x)

        F_y, f_y = compute_survival_functions(mu, sigma, y)
        F_c = 1 - y / b_esti
        f_c = 1 / b_esti

        term1 = ((1 - cure_prob) * f_y * F_c) ** s.float()
        term2 = ((1 - cure_prob) * F_y * f_c + cure_prob * f_c) ** (1 - s.float())
        likelihood = term1 * term2
        nll = -torch.log(likelihood + self.eps).mean()

        l2_loss = sum(torch.sum(param ** 2) for param in self.cure_model.parameters()) + \
                  sum(torch.sum(param ** 2) for param in self.survival_model.parameters())

        total_loss = nll + self.l2_reg * l2_loss

        if torch.isnan(total_loss):
            print("NaN detected!")

        return total_loss


def prepare_train_val_test_sets_by_group(df_treated, df_control, train_ratio=0.375, val_ratio=0.125, test_ratio=0.5):
    """
    Split treated and control groups into training, validation, and test sets.

    :param df_treated: DataFrame for treated group
    :param df_control: DataFrame for control group
    :param train_ratio: Proportion of the data to be used for training
    :param val_ratio: Proportion of the data to be used for validation
    :param test_ratio: Proportion of the data to be used for testing
    :return: Train, validation, and test sets for treated, control, and combined groups
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    def split_group(df):
        n_samples = len(df)
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        indices = np.random.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        return df.iloc[train_indices], df.iloc[val_indices], df.iloc[test_indices]

    # Split treated and control groups
    train_treated, val_treated, test_treated = split_group(df_treated)
    train_control, val_control, test_control = split_group(df_control)

    # Combine treated and control groups for mixed sets
    train_combined = pd.concat([train_treated, train_control]).sample(frac=1, random_state=42)
    val_combined = pd.concat([val_treated, val_control]).sample(frac=1, random_state=42)
    test_combined = pd.concat([test_treated, test_control]).sample(frac=1, random_state=42)

    return (train_treated, val_treated, test_treated,
            train_control, val_control, test_control,
            train_combined, val_combined, test_combined)

# Prepare datasets for treated and untreated groups
(train_treated, val_treated, test_treated,
 train_control, val_control, test_control,
 train_combined, val_combined, test_combined) = prepare_train_val_test_sets_by_group(df_treated, df_untreated)

def train_models(data, epochs, learning_rate):

    X_train, X_val, cure_train, cure_val, Y_train, Y_val, S_train, S_val = data[:8]

    X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    cure_tensor_train = torch.tensor(cure_train, dtype=torch.float32).unsqueeze(1)
    Y_tensor_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    S_tensor_train = torch.tensor(S_train, dtype=torch.float32).unsqueeze(1)

    X_tensor_val = torch.tensor(X_val, dtype=torch.float32)
    cure_tensor_val = torch.tensor(cure_val, dtype=torch.float32).unsqueeze(1)
    Y_tensor_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)
    S_tensor_val = torch.tensor(S_val, dtype=torch.float32).unsqueeze(1)

    cure_model = CureModel(X_train.shape[1])
    survival_model = AFTModel(X_train.shape[1])
    combined_loss = CombinedLoss(cure_model, survival_model, b_esti, l2_reg=1e-4)

    optimizer_cure = optim.Adam(cure_model.parameters(), lr=learning_rate)
    optimizer_survival = optim.Adam(survival_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        cure_model.train()
        survival_model.train()

        optimizer_cure.zero_grad()
        optimizer_survival.zero_grad()

        loss = combined_loss(X_tensor_train, S_tensor_train, Y_tensor_train)
        loss.backward()

        optimizer_cure.step()
        optimizer_survival.step()

        if (epoch + 1) % 20 == 0:
            cure_model.eval()
            survival_model.eval()
            with torch.no_grad():
                val_loss = combined_loss(X_tensor_val, S_tensor_val, Y_tensor_val)
                print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    return cure_model, survival_model


# Prepare treated group datasets
X_train_treated = train_treated.filter(regex='^X').values
X_val_treated = val_treated.filter(regex='^X').values
X_test_treated = test_treated.filter(regex='^X').values

cure_train_treated = train_treated['cure status'].values
cure_val_treated = val_treated['cure status'].values
cure_test_treated = test_treated['cure status'].values

Y_train_treated = train_treated['Y'].values
Y_val_treated = val_treated['Y'].values
Y_test_treated = test_treated['Y'].values

S_train_treated = train_treated['S'].values
S_val_treated = val_treated['S'].values
S_test_treated = test_treated['S'].values

# Prepare control group datasets
X_train_control = train_control.filter(regex='^X').values
X_val_control = val_control.filter(regex='^X').values
X_test_control = test_control.filter(regex='^X').values

cure_train_control = train_control['cure status'].values
cure_val_control = val_control['cure status'].values
cure_test_control = test_control['cure status'].values

Y_train_control = train_control['Y'].values
Y_val_control = val_control['Y'].values
Y_test_control = test_control['Y'].values

S_train_control = train_control['S'].values
S_val_control = val_control['S'].values
S_test_control = test_control['S'].values

# Prepare combined datasets
# Include only "Z" and "X*" columns, exclude "Z_cf" and other non-feature columns
X_train_combined = train_combined.filter(regex='^X').values
X_val_combined = val_combined.filter(regex='^X').values
X_test_combined = test_combined.filter(regex='^X').values


cure_train_combined = train_combined['cure status'].values
cure_val_combined = val_combined['cure status'].values
cure_test_combined = test_combined['cure status'].values

Y_train_combined = train_combined['Y'].values
Y_val_combined = val_combined['Y'].values
Y_test_combined = test_combined['Y'].values

S_train_combined = train_combined['S'].values
S_val_combined = val_combined['S'].values
S_test_combined = test_combined['S'].values


# Train and validate models for treated group
data_treated = (X_train_treated, X_val_treated, cure_train_treated, cure_val_treated, Y_train_treated, Y_val_treated, S_train_treated, S_val_treated)
model_treated = train_models(data_treated, epochs, learning_rate)

# Train and validate models for control group
data_control = (X_train_control, X_val_control, cure_train_control, cure_val_control, Y_train_control, Y_val_control, S_train_control, S_val_control)
model_control = train_models(data_control, epochs, learning_rate)

# Train and validate models for combined group
data_combined = (X_train_combined, X_val_combined, cure_train_combined, cure_val_combined, Y_train_combined, Y_val_combined, S_train_combined, S_val_combined)
model_combined = train_models(data_combined, epochs, learning_rate)


cure_model_treated, survival_model_treated = model_treated
cure_model_untreated, survival_model_untreated = model_control
cure_model_combined, survival_model_combined = model_combined



def evaluate_models(cure_model, survival_model, X_val, cure_val, Y_val, S_val, t):
    """
    Evaluate the cure model and survival model on the validation set.
    """
    cure_model.eval()
    survival_model.eval()
    with torch.no_grad():
        # Ensure all inputs are tensors
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32)
        if not isinstance(cure_val, torch.Tensor):
            cure_val = torch.tensor(cure_val, dtype=torch.float32)
        if not isinstance(Y_val, torch.Tensor):
            Y_val = torch.tensor(Y_val, dtype=torch.float32)
        if not isinstance(S_val, torch.Tensor):
            S_val = torch.tensor(S_val, dtype=torch.float32)

        # Get predictions from models
        cure_pred_val = cure_model(X_val).squeeze()  # Ensure correct shape
        mu_val, sigma_val = survival_model(X_val)

        # Compute the AUC for the cure model predictions
        auc_val = roc_auc_score(cure_val.numpy().flatten(), cure_pred_val.numpy().flatten())
        print(f"AUC-ROC for Cure Probability: {auc_val:.3f}")

        # Compute the Concordance Index for survival predictions
        normal_dist = Normal(mu_val, sigma_val)
        log_t = torch.log(torch.tensor([t]).float())  # Use the same data type as mu, sigma
        S_t = 1 - normal_dist.cdf(log_t).squeeze()

        mcm_survival = cure_pred_val + (1 - cure_pred_val) * S_t
        event_observed = S_val.numpy().flatten()
        c_index = concordance_index(Y_val.numpy().flatten(), mcm_survival.numpy(), event_observed=event_observed)
        print(f"Concordance Index: {c_index:.3f}")

        # Model coefficients
        estimated_beta = cure_model.linear.weight.data.numpy().flatten()
        beta_bias = cure_model.linear.bias.data.numpy()
        print(f"Estimated beta: {estimated_beta}")
        print(f"Estimated beta bias: {beta_bias}")

        estimated_gamma = survival_model.linear_mu.weight.data.numpy().flatten()
        print(f"Estimated gamma: {estimated_gamma}")

        for name, param in survival_model.named_parameters():
            print(f"{name}: {param}")

    return cure_pred_val, auc_val, c_index, estimated_beta, estimated_gamma


cure_pred_val_treat, auc_val_treat, c_index_treat, esti_beta_treat, esti_gamma_treat = evaluate_models(
    cure_model_treated, survival_model_treated,
    X_val_treated, cure_val_treated, Y_val_treated, S_val_treated, t
)
cure_pred_val_control, auc_val_control, c_index_control, esti_beta_control, esti_gamma_control = evaluate_models(
    cure_model_untreated, survival_model_untreated,
    X_val_control, cure_val_control, Y_val_control, S_val_control, t
)
cure_pred_val_combined, auc_val_combined, c_index_combined, esti_beta_combined, esti_gamma_combined = evaluate_models(
    cure_model_combined, survival_model_combined,
    X_val_combined, cure_val_combined, Y_val_combined, S_val_combined, t
)


print("finish training")

# -------------------------------- Section 3: Matching ------------------------------------

def plot_difference_distribution(differences, k, title):
    mean_diff = np.mean(differences)
    var_diff = np.var(differences)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(differences, bw_adjust=0.5)
    plt.title(title)
    plt.xlabel("Difference")
    plt.ylabel("Density")

    # Annotate mean and variance on the plot
    plt.text(x=0.05, y=plt.ylim()[1] * 0.8, s=f"Mean: {mean_diff:.4f}\nVariance: {var_diff:.4f}", fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.show()


def calculate_individual_event_probabilities(match_S, match_Y, time_horizon):
    probabilities = []
    censor_rates = []
    kmf = KaplanMeierFitter()

    for i in range(match_S.shape[0]):
        event_indicators = match_S[i]
        survival_times = match_Y[i]

        kmf.fit(survival_times, event_observed=event_indicators)
        probability_event_occurs = 1 - kmf.survival_function_at_times(time_horizon).values[0]
        probabilities.append(probability_event_occurs)

        censor_rate = (match_S.shape[0] - event_indicators.sum()) / match_S.shape[0]
        censor_rates.append(censor_rate)


    return np.array(probabilities), np.array(censor_rates)


# features, labels and indices from Test set
df_test = test_combined
df_test['Z'] = 1 - df_test['Z_cf']
df_test = df_test.reset_index(drop=True)

features = df_test.filter(regex='^X').values
labels = df_test['Z'].values  # Assuming 'Z' is the treatment indicator
original_indices = df_test.index.values  # original indices

# weight by average coefficients
beta_weight = (np.abs(esti_beta_treat) + np.abs(esti_beta_control)) / 2
gamma_weight = (np.abs(esti_gamma_treat) + np.abs(esti_gamma_control)) / 2

beta_weight_normalized = beta_weight / np.linalg.norm(beta_weight, ord=2)
gamma_weight_normalized = gamma_weight / np.linalg.norm(gamma_weight, ord=2)
combined_weight = (beta_weight_normalized + gamma_weight_normalized) / 2


treated_indices = np.where(labels == 1)[0]
control_indices = np.where(labels == 0)[0]

# ------------ 1. Matching for Cure probability (weighted by beta) --------------------
nn_control = NearestNeighbors(n_neighbors=k, metric='minkowski', metric_params={'w': combined_weight}, p=2)
nn_treat = NearestNeighbors(n_neighbors=k, metric='minkowski', metric_params={'w': combined_weight}, p=2)

# nn_control = NearestNeighbors(n_neighbors=k, metric='minkowski', p=2)
# nn_treat = NearestNeighbors(n_neighbors=k, metric='minkowski', p=2)

# ------ PS Matching --------
# from sklearn.linear_model import LogisticRegression
# logit_model = LogisticRegression()
# logit_model.fit(features, labels)
# propensity_scores = logit_model.predict_proba(features)[:, 1]
# nn_control = NearestNeighbors(n_neighbors=k, metric='euclidean')
# nn_treat = NearestNeighbors(n_neighbors=k, metric='euclidean')
#
# nn_control.fit(propensity_scores[control_indices].reshape(-1, 1))
# nn_treat.fit(propensity_scores[treated_indices].reshape(-1, 1))
#
# distances_ct, indices_ct = nn_control.kneighbors(propensity_scores[treated_indices].reshape(-1, 1))
# distances_tc, indices_tc = nn_treat.kneighbors(propensity_scores[control_indices].reshape(-1, 1))
# distances_tt, indices_tt = nn_treat.kneighbors(propensity_scores[treated_indices].reshape(-1, 1))
# distances_cc, indices_cc = nn_control.kneighbors(propensity_scores[control_indices].reshape(-1, 1))

# --------------------------
treated_features = features[treated_indices]
control_features = features[control_indices]

nn_control.fit(control_features)
nn_treat.fit(treated_features)

distances_ct, indices_ct = nn_control.kneighbors(treated_features)
distances_tc, indices_tc = nn_treat.kneighbors(control_features)

distances_tt, indices_tt = nn_treat.kneighbors(treated_features)
distances_cc, indices_cc = nn_control.kneighbors(control_features)

# ---------------------------
control_indices_for_treat = control_indices[indices_ct]
treat_indices_for_control = treated_indices[indices_tc]

treat_indices_for_treat = treated_indices[indices_tt]
control_indices_for_control = control_indices[indices_cc]

# estimation
treat_control_S = np.array([S_test_combined[indices] for indices in treat_indices_for_control])
control_treat_S = np.array([S_test_combined[indices] for indices in control_indices_for_treat])

treat_treat_S = np.array([S_test_combined[indices] for indices in treat_indices_for_treat])
control_control_S = np.array([S_test_combined[indices] for indices in control_indices_for_control])

time_horizon = b_esti  # Adjust as needed

# Estimate probabilities for each matching group
probs_treat_control, rates_treat_control = calculate_individual_event_probabilities(treat_control_S, Y_test_combined[treat_indices_for_control], time_horizon)
probs_control_treat, rates_control_treat = calculate_individual_event_probabilities(control_treat_S, Y_test_combined[control_indices_for_treat], time_horizon)
probs_treat_treat, rates_treat_treat = calculate_individual_event_probabilities(treat_treat_S, Y_test_combined[treat_indices_for_treat], time_horizon)
probs_control_control, rates_control_control = calculate_individual_event_probabilities(control_control_S, Y_test_combined[control_indices_for_control], time_horizon)

# sort by original index
esti_event = np.concatenate((probs_treat_treat, probs_control_control))
esti_event_cf = np.concatenate((probs_control_treat, probs_treat_control))
cure_indices = np.concatenate((treated_indices, control_indices))
wknn_cure_esti = np.column_stack((cure_indices, esti_event, esti_event_cf))
wknn_cure_esti = wknn_cure_esti[wknn_cure_esti[:, 0].argsort()]

true_event_probs = 1 - df_test['P_cure'].values
true_cf_event_probs = 1 - df_test['P_cure_cf'].values
wknn_cure_esti = np.column_stack((wknn_cure_esti, true_event_probs, true_cf_event_probs))

column_names = ['Index', 'Estimated event prob', 'Estimated CF event prob', 'True Event Prob', 'True CF Event Prob']
df_wknn_cure_esti = pd.DataFrame(wknn_cure_esti, columns=column_names)


df_wknn_cure_esti['Estimated cure ITE'] = np.nan

df_wknn_cure_esti.loc[treated_indices, 'Estimated cure ITE'] = (
    df_wknn_cure_esti.loc[treated_indices, 'Estimated CF event prob'] -
    df_wknn_cure_esti.loc[treated_indices, 'Estimated event prob']
)

df_wknn_cure_esti.loc[control_indices, 'Estimated cure ITE'] = (
    df_wknn_cure_esti.loc[control_indices, 'Estimated event prob'] -
    df_wknn_cure_esti.loc[control_indices, 'Estimated CF event prob']
)

df_wknn_cure_esti['True cure ITE'] = np.nan

df_wknn_cure_esti.loc[treated_indices, 'True cure ITE'] = (
    df_wknn_cure_esti.loc[treated_indices, 'True CF Event Prob'] -
    df_wknn_cure_esti.loc[treated_indices, 'True Event Prob']
)

df_wknn_cure_esti.loc[control_indices, 'True cure ITE'] = (
    df_wknn_cure_esti.loc[control_indices, 'True Event Prob'] -
    df_wknn_cure_esti.loc[control_indices, 'True CF Event Prob']
)

ite_differences = df_wknn_cure_esti['Estimated cure ITE'] - df_wknn_cure_esti['True cure ITE']
ite_diff_mean = ite_differences.mean()
ite_diff_std = ite_differences.std()

# Calculate mean square error (MSE) and standard deviation of squared errors
ite_squared_errors = ite_differences**2
ite_mse = ite_squared_errors.mean()
ite_squared_error_std = ite_squared_errors.std()

ate_esti = df_wknn_cure_esti['Estimated cure ITE'].mean()
# Print the results
print(f"ATE: {ate_esti:.4f}")
print(f"ITE bias Mean: {ite_diff_mean:.4f}")
print(f"ITE bias Std: {ite_diff_std:.4f}")
print(f"ITE Mean Square Error: {ite_mse:.4f}")
print(f"Squared Error Std: {ite_squared_error_std:.4f}")


differences = df_wknn_cure_esti['Estimated CF event prob'] - df_wknn_cure_esti['True CF Event Prob']
title = f"Distribution of Differences between CF K-M prob and CF True prob"
plot_difference_distribution(differences, k, title)


