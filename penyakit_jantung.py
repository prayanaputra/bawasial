import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from collections import Counter

# %%
##tentukan libary yang akan digunakan
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
##load dataset
heart_data = pd.read_csv("heart_cleveland_upload.csv")

# %%
heart_data.head()

# %%
##pisahkan data atribut dengan label
X = heart_data.drop(columns='condition', axis=1)
Y = heart_data['condition']

# %%
print(X)

# %%
print(Y)

# %%
##pisahkan data training dan data testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %%
##membuat label training
model = LogisticRegression()

# %%
model.fit(X_train, Y_train)

# %%
##evaluasi model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# %%
print('akurasi data training :', training_data_accuracy)

# %%
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# %%
print('akurasi data testing :', test_data_accuracy)

# %%
##buat model prediksi
input_data = (61, 1, 0, 134, 234, 0, 0, 145, 0, 2.6, 1, 2, 0)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print('Pasien Tidak Terkena Penyakit Jantung')
else:
  print('Pasien Terkena Penyakit Jantung')

# %%
##simpan model
import pickle

# %%
filename = 'penyakit_jantung.sav'
pickle.dump(model, open(filename, 'wb'))

# %%
import numpy as np
import seaborn as sns
import matplotlib
import wordcloud
import sklearn
import imblearn
import pandas as pd

# Print version information
print(f"NumPy version: {np.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Wordcloud version: {wordcloud.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Imbalanced-learn version: {imblearn.__version__}")
print(f"Pandas version: {pd.__version__}")



