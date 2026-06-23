#!/usr/bin/env python
# coding: utf-8

# In[23]:


import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device


# In[24]:


# 1. Analiza błędu w sieci neuronowej

import torch
import torch.nn as nn
import torch.optim as optim

class AmazingNet(nn.Module):
    def __init__(self,activation_function,layer_count: int = 4,input_dim: int = 10,hidden_dim: int = 64):
        super().__init__()
        self.activation_function = activation_function
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.linears = nn.ModuleList()

        self.linears.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.layer_count):
            layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)

        self.linears.append(nn.Linear(self.hidden_dim, 1))

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = layer(x)
            x = self.activation_function(x)
        x = self.linears[-1](x)
        return x


# In[25]:


import numpy as np

model = AmazingNet(activation_function=nn.ReLU())
model.to(device)  

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

X = np.random.randn(1000, 10)
y = np.sum(X**2, axis=1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

X_tensor = X_tensor.to(device)  
y_tensor = y_tensor.to(device)  

epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()

    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model_fixed.pth")


# In[26]:


# 2. Armageddon

class AsteroidProfiler(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# In[27]:


import umtools.datasets
import matplotlib.pyplot as plt

X, y = umtools.datasets.load_asteroid_data()

X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
X_tensor = X_tensor.to(device)  
y_tensor = y_tensor.to(device) 

model_a = AsteroidProfiler()
model_a.to(device) 

criterion = nn.MSELoss()
optimizer_a = optim.SGD(model_a.parameters(), lr=0.01)

epochs_a = 500
for epoch in range(epochs_a):
    optimizer_a.zero_grad()
    predictions_a = model_a(X_tensor)
    loss = criterion(predictions_a, y_tensor)
    loss.backward()
    optimizer_a.step()

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Prawdziwe dane z czujników (Cel)', color='gray', alpha=0.5)
plt.plot(X, model_a(X_tensor).cpu().detach().numpy(), color='red', linewidth=2, label='Predykcja Modelu A (SGD)')
plt.title('Przewidywanie uskoku - 500 epok (SGD)')
plt.xlabel('Głębokość czujnika (X)')
plt.ylabel('Własności fizyczne (y)')
plt.legend()
plt.savefig('asteroid_prediction_1.png')
plt.close()


# In[28]:


from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor.cpu(), y_tensor.cpu())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model_b = AsteroidProfiler()
model_b.to(device)

optimizer_b = optim.Adam(model_b.parameters(), lr=0.005)

epochs_b = 500
for epoch in range(epochs_b):
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer_b.zero_grad()
        predictions_b = model_b(batch_X)
        loss = criterion(predictions_b, batch_y)
        loss.backward()
        optimizer_b.step()

X_sorted = np.sort(X)
X_sorted_tensor = torch.tensor(X_sorted, dtype=torch.float32).view(-1, 1).to(device)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Prawdziwe dane z czujników (Cel)', color='gray', alpha=0.5)
plt.plot(X_sorted, model_b(X_sorted_tensor).cpu().detach().numpy(), color='green', linewidth=3, label='Predykcja Modelu B (Adam + Batche)')
plt.title('Przewidywanie uskoku - 500 epok (Adam + Mini-Batches)')
plt.xlabel('Głębokość czujnika (X)')
plt.ylabel('Własności fizyczne (y)')
plt.legend()
plt.savefig('asteroid_prediction_2.png')
plt.close()

torch.save(model_b, "asteroid_profiler.pt")


# In[39]:


# 3. Optuna

import torch.nn as nn
from umtools.datasets import make_spirals

X, y = make_spirals(n_samples=1000, noise=0.2, n_classes=10)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)

class DynamicMLP(nn.Module):
    n_classes: int = 10

    def __init__(self, n_layers, hidden_dim, activation_function):
        super().__init__()

        activations = {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid()
        }
        self.act_fn = activations[activation_function]

        self.layers = nn.ModuleList()
        input_dim = 2  

        self.layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, self.n_classes))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.act_fn(self.layers[i](x))

        x = self.layers[-1](x)
        return x


# In[40]:


import optuna

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 5) 
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128) 
    activation = trial.suggest_categorical("activation", ["ReLU", "Tanh", "Sigmoid"]) 
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "NAdam"]) 
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 128)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DynamicMLP(n_layers, hidden_dim, activation_function=activation).to(device)
    criterion = nn.CrossEntropyLoss()

    opt_class = getattr(optim, optimizer_name)
    optimizer = opt_class(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(2000):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        final_outputs = model(X_tensor)
        final_loss = criterion(final_outputs, y_tensor)

    return final_loss.item()


#study = optuna.create_study(direction="minimize")
#study.optimize(objective, n_trials=40)
#study.trials_dataframe().to_csv("study_results.csv", index=False)
#best_params = study.best_params

#best_dataset = TensorDataset(X_tensor, y_tensor)
#best_dataloader = DataLoader(best_dataset, batch_size=best_params["batch_size"], shuffle=True)

#best_model = DynamicMLP(
#    n_layers=best_params["n_layers"], 
#    hidden_dim=best_params["hidden_dim"], 
#    activation_function=best_params["activation"]
#).to(device)

#criterion = nn.CrossEntropyLoss()
#best_opt_class = getattr(optim, best_params["optimizer"])
#best_optimizer = best_opt_class(best_model.parameters(), lr=best_params["learning_rate"])

#best_model.train()

#for epoch in range(5000):
#    for batch_X, batch_y in best_dataloader:
#        best_optimizer.zero_grad()
#        outputs = best_model(batch_X)
#        loss = criterion(outputs, batch_y)
#        loss.backward()
#        best_optimizer.step()
#
#    if (epoch + 1) % 500 == 0:
#        print(f"Epoka [{epoch+1}/5000], Obecna Strata (Loss): {loss.item():.4f}")

#torch.save(best_model.state_dict(), "best_model.pth")


# In[ ]:

# 4. Ile ludzi na miasteczku

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from umtools.datasets import load_miasteczko
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_test = load_miasteczko()

X_train = X_train.set_index('id')
X_test = X_test.set_index('id')

y_train_raw = X_train['studenty_ms'].values
X_train_raw = X_train.drop(columns=['studenty_ms'])
X_test_raw = X_test.copy()

cols_to_drop = []
for col in X_train_raw.select_dtypes(include=['object', 'string']).columns:
    try:
        dates_train = pd.to_datetime(X_train_raw[col], format='%Y-%m-%d')
        dates_test = pd.to_datetime(X_test_raw[col], format='%Y-%m-%d')
        
        X_train_raw[col + '_month'] = dates_train.dt.month.astype(str)
        X_train_raw[col + '_dayofweek'] = dates_train.dt.dayofweek.astype(str)
        
        X_test_raw[col + '_month'] = dates_test.dt.month.astype(str)
        X_test_raw[col + '_dayofweek'] = dates_test.dt.dayofweek.astype(str)
        
        cols_to_drop.append(col)
    except ValueError:
        pass

X_train_raw = X_train_raw.drop(columns=cols_to_drop)
X_test_raw = X_test_raw.drop(columns=cols_to_drop)

num_cols = X_train_raw.select_dtypes(include=['number']).columns
X_train_raw[num_cols] = X_train_raw[num_cols].fillna(X_train_raw[num_cols].median())
X_test_raw[num_cols] = X_test_raw[num_cols].fillna(X_train_raw[num_cols].median())

X_train_raw = pd.get_dummies(X_train_raw, drop_first=True, dtype=float)
X_test_raw = pd.get_dummies(X_test_raw, drop_first=True, dtype=float)
X_train_raw, X_test_raw = X_train_raw.align(X_test_raw, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

ms_test_df = pd.DataFrame(X_test_scaled, columns=X_test_raw.columns, index=X_test_raw.index)
ms_test_df.to_pickle("ms_test.pkl")

X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_train_raw, dtype=torch.float32).view(-1, 1).to(device)

class MiasteczkoNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train_scaled.shape[1]
model = MiasteczkoNet(input_dim).to(device) 

criterion = nn.MSELoss() 
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

model.train()
for epoch in range(1500):
    optimizer.zero_grad()
    
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 300 == 0:
        print(f"Epoka [{epoch+1}/1500], Strata MSE: {loss.item():.4f}")

torch.save(model, "miasteczko.pt")


