# project# Import the necessary Python libraries
import pandas as pd              
import numpy as np                
import matplotlib.pyplot as plt    
from sklearn.svm import SVC        
from sklearn.preprocessing import StandardScaler 
df = pd.read_csv("vHoneyNeonic_v02.csv")
df['decline_flag'] = df['totalprod'] < df['totalprod'].median()
X = df[['nAllNeonic', 'numcol']].fillna(0)  
X = df[['nAllNeonic', 'numcol']].fillna(0)  
y = df['decline_flag'].fillna(0)           
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = SVC(kernel='linear')   
model.fit(X_scaled, y)         
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)  
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')  
plt.title("SVM Decision Boundary: Pesticide Use vs Bee Colony Decline")
plt.xlabel("Pesticide Use (Standardized)")
plt.ylabel("Number of Colonies (Standardized)")
plt.grid(True)
plt.tight_layout()
plt.show()
