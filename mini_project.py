#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kagglehub

# Download latest version
directory_path = kagglehub.dataset_download("vinicius150987/titanic3")

print("directory_path to dataset files:", directory_path)


# In[2]:


import os
files_path = os.listdir(directory_path)
files_paths = { file_path.split('.')[0]: os.path.join(directory_path, file_path) for file_path in files_path }
files_paths


# In[3]:


import pandas as pd

df = pd.read_excel(files_paths['titanic3'])
display(df.head(10))
print(df.info())
df["embarked"].unique()


# In[4]:


df["ticket"].nunique(), df["fare"].nunique()
for col in df.select_dtypes(include=['O']).columns:
    print(f"{col}: ", df[col].unique())


# In[5]:


df['boat'] = pd.to_numeric(df['boat'], errors='coerce')
df['boat'].unique()


# In[6]:


mask = df['ticket'] != 'LINE'
df = df[mask]

def clean_ticket(ticket):
    if isinstance(ticket, str):
        if ticket.isdigit():
            return int(ticket)
        return int(ticket.split(' ')[-1])
    return ticket


df['ticket'] = df['ticket'].apply(clean_ticket)
df['ticket'].unique()


# In[7]:


df.info()
df['ticket'].unique()


# In[8]:


def getTitle(name: str):
    temp = name.split(',')[1]
    title = temp.split('.')[0]
    title = title.strip()
    return title

df['title'] = df['name'].apply(getTitle)
df = df.drop(columns=['name'])


# In[9]:


mask = (df['title'] == 'Miss') | (df['title'] == 'Mlle')
df.loc[mask, 'title'] = 'Miss'


# In[10]:


mask = (df['title'] == 'Mrs') | (df['title'] == 'Mme')
df.loc[mask, 'title'] = 'Mrs'


# In[11]:


mask = (df['parch'] > 2) & (df['title'] == "Ms")
df.loc[mask, 'title'] = 'Mrs'


# In[12]:


import seaborn as sns

series = df['age']
print(series)
sns.boxplot(x='age', data=df)


# In[13]:


q1 = series.quantile(0.25)
q3 = series.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
mask = (series < lower_bound) | (series > upper_bound)
df = df[~mask]
df.info()


# In[14]:


# put outliers in upper whisker
import numpy as np
series = df['age']
q1 = series.quantile(0.25)
q3 = series.quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
df['age'] = np.where(df['age'] > upper_bound, upper_bound, df['age'])
sns.boxplot(x='age', data=df)


# In[15]:


for col in df.columns:
    print(f"{col}: ", (df[col].isnull().sum() / df.shape[0]) * 100)


# In[16]:


from matplotlib import pyplot as plt

for col in df.select_dtypes(include=['number']).columns:
    sns.histplot(data=df, x=col, kde=True)
    plt.show()


# In[17]:


df = df.drop(columns=['cabin', 'boat', 'body', 'home.dest', 'embarked', 'fare'])
print(df.info())
display(df['title'].value_counts())
display(df['sibsp'].value_counts())
display(df['parch'].value_counts())
df['title'].value_counts()


# In[18]:


col_with_nans = [col for col in df.columns if df[col].isna().sum() > 0]
num_cols = df[col_with_nans].select_dtypes(include=['number']).columns.tolist()
num_cols


# In[19]:


from sklearn.preprocessing import LabelBinarizer

binarizer = LabelBinarizer()
encoder = binarizer.fit_transform(df[['sex']])
df['ismale'] = encoder.squeeze()
df = df.drop(columns=['sex'])
df.columns


# In[20]:


df['title'].unique()
def army_levels(title):
    if title == 'Col':
        return 1
    elif title == 'Major':
        return 2
    else:
        return 0
def nobility(noble):
    if noble in ['Dona', 'Jonkheer', 'the Countess', 'Don', 'Sir', 'Lady']:
        return 1
    else:
        return 0

#df = df[df["title"] not in ['Dona', 'Jonkheer', 'the Countess', 'Don', 'Sir', 'Lady']]

df['army_levels'] = df['title'].apply(army_levels)
df['nobility_levels'] = df['title'].apply(nobility)


# In[21]:


df.drop(columns=['title'],inplace=True)
df.info()


# In[22]:


from lightgbm import LGBMRegressor

for col in num_cols:
    df_copy = df.copy()
    nan_idx_labels = df_copy.index[df_copy[col].isna()]
    df_copy['nan_idx'] = 0
    df_copy.loc[nan_idx_labels, 'nan_idx'] = 1

    train = df_copy[df_copy['nan_idx'] == 0]
    test = df_copy[df_copy['nan_idx'] == 1]

    X_train = train.drop(columns=[col, 'nan_idx'])
    y_train = train[col]
    X_test = test.drop(columns=[col, 'nan_idx'])

    model = LGBMRegressor()
    model.fit(X_train, y_train)
    df.loc[nan_idx_labels, col] = model.predict(X_test)
df.info()


# In[23]:


# df['age'].fillna(df['age'].mean(),inplace=True)


# In[24]:


df.isna().sum().sum()


# In[25]:


from sklearn.preprocessing import StandardScaler

stander = StandardScaler()
df['age'] = stander.fit_transform(df[['age']]).squeeze()


# In[26]:


df['age']


# In[27]:


X = df.drop(columns='survived')
y = df['survived']
X.shape , y.shape


# In[28]:


# Model Selection & Model Performance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[29]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# In[30]:


# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)


# In[31]:


log_reg.fit(X_train, y_train)


# In[32]:


y_pred = log_reg.predict(X_test)


# In[33]:


# Logistic Regression Performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

lr_acc = accuracy_score(y_test, y_pred)
lr_prec = precision_score(y_test, y_pred)
lr_rec = recall_score(y_test, y_pred)
lr_f1_score = f1_score(y_test, y_pred)

print("Accuracy:", lr_acc)
print("Precision:", lr_prec)
print("Recall:", lr_rec)
print("F1 Score:", lr_f1_score)


# In[34]:


# Logistic Regression Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# In[35]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[36]:


# Initialize Decision Tree
dt = DecisionTreeClassifier()


# In[37]:


dt.fit(X_train, y_train)


# In[38]:


y_pred = dt.predict(X_test)


# In[39]:


# Decision Tree Performance
dt_acc = accuracy_score(y_test, y_pred)
dt_prec = precision_score(y_test, y_pred)
dt_rec = recall_score(y_test, y_pred)
dt_f1_score = f1_score(y_test, y_pred)

print("Accuracy:", dt_acc)
print("Precision:", dt_prec)
print("Recall:", dt_rec)
print("F1 Score:", dt_f1_score)


# In[40]:


# Decision Tree Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()


# In[41]:


# Plot Decision Tree (Too many features made the plot too big)
# dt_plot = plot_tree(decision_tree=dt, 
#              feature_names=X.columns,
#             class_names=["Not Survived","Survived"],
#              filled=True,
#              rounded=True)
#plt.savefig('tree.pdf', format='pdf')
#plt.show()


# In[42]:


# Final Comparison
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
lr_scores = [lr_acc, lr_prec, lr_rec, lr_f1_score]
dt_scores = [dt_acc, dt_prec, dt_rec, dt_f1_score]

x = np.arange(len(metrics)) 
width = 0.35 

fig, ax = plt.subplots(figsize=(8, 6))
bars1 = ax.bar(x - width/2, lr_scores, width, label="Logistic Regression")
bars2 = ax.bar(x + width/2, dt_scores, width, label="Decision Tree")

# Labels & formatting
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend()

# Add numbers on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom")

plt.show()

# In[43]:
import joblib
joblib.dump(log_reg, "titanic_model.pkl")
