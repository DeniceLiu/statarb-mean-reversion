# src/main.py

from src.ou_model import (
    download_log_data,
    train_ou_model,
    print_z_scores,
    regression_ou_params,
)
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
tickers = "GLD SIL"

# 1. log data
train = download_log_data(tickers, "2022-01-01", "2023-12-31")
test_retrain = download_log_data(tickers, "2024-01-01", "2024-05-30")
test_final = download_log_data(tickers, "2024-06-01", "2024-10-10")

# 2. spread plot
spread = train["GLD"] - train["SIL"]
plt.figure(figsize=(10, 5))
plt.plot(spread)
plt.axhline(spread.mean(), linestyle='--', color='r', label='Mean')
plt.title("GLD - SIL Log Price Spread (Training)")
plt.xlabel("Date")
plt.ylabel("Log Price Spread")
plt.legend()
plt.tight_layout()
plt.show()

# 3. correlation check
print("\nCorrelation on Final Test Data:")
print(test_final.corr())

# 4. train ou model on train data
model = train_ou_model(train)
model.check_fit()
model.L = None
_ = model.description()
model.L = 0.2
print("\n--- Model Description (Training) ---")
print(model.description().to_string())

print("\n--- Entry/Liquidation Z-Scores (Training) ---")
print_z_scores(model)

# 5. optimal levels on training data
fig1 = model.plot_levels(data=train, stop_loss=True)
fig1.set_figheight(10)
fig1.set_figwidth(15)
plt.tight_layout()
plt.show()

# 6. optimal levels on retrain data
fig2 = model.plot_levels(data=test_retrain, stop_loss=True)
fig2.set_figheight(10)
fig2.set_figwidth(15)
plt.tight_layout()
plt.show()

# 7. retrain on test period
model.fit_to_assets(data=test_retrain.to_numpy())
model.check_fit()
model.L = None
_ = model.description()

model.L = 0.2
print("\n--- Model Description After Retraining ---")
print(model.description().to_string())

print("\n--- Entry/Liquidation Z-Scores (After Retraining) ---")
print_z_scores(model)

# 8. optimal levels on final data
fig3 = model.plot_levels(data=test_final, stop_loss=True)
fig3.set_figheight(10)
fig3.set_figwidth(15)
plt.tight_layout()
plt.show()

# 9. regression-based OU parameter estimation
train["Spread"] = train["GLD"] - train["SIL"]
mu, theta = regression_ou_params(train['Spread'])

print("\n--- OU Parameters Estimated via Regression ---")
print(f"mu = {mu:.4f}, theta = {theta:.4f}")
