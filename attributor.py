import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis, linregress
import matplotlib.pyplot as plt

import seaborn as sns


# Load metadata
metadata = pd.read_csv('data/SPGC-metadata-2018-07-18.csv')

# Load KLD scores
kld_scores = pd.read_csv('data/KLDscores.csv')
extra_controls = pd.read_csv('data/extra_controls.csv')

data = pd.merge(metadata, kld_scores, left_on='id', right_on="filename", how='inner')

data = pd.merge(data, extra_controls, on='id', how='inner')
print(data.head())


indexes  = {}

def standard_deviation( kld_list):
  kld_array = np.array(eval(kld_list))
  return np.std(kld_array)

def avg_kld( kld_list):
  kld_array = np.array(eval(kld_list))
  return np.mean(kld_array)

def median_kld(kld_list):
  kld_array = np.array(eval(kld_list))
  return np.median(kld_array)

def total_positive_changes(kld_list):
  kld_array = np.diff(np.array(eval(kld_list)))
  positive_changes = kld_array[kld_array > 0]
  return np.sum(positive_changes)

def total_negative_changes(kld_list):
  kld_array = np.diff(np.array(eval(kld_list)))
  negative_changes = kld_array[kld_array < 0]
  return np.sum(negative_changes)

def coefficient_of_variation(kld_list):
  kld_array = np.array(eval(kld_list))
  mean = np.mean(kld_array)
  std = np.std(kld_array)
  return std / mean if mean != 0 else 0

def slope_of_trend( kld_list):
  kld_array = np.array(eval(kld_list))
  sections = np.arange(len(kld_array))
  slope, intercept, r_value, p_value, std_err = linregress(sections, kld_array)
  return slope

def calculate_iqr(kld_values):
    kld_array = np.array(eval(kld_values))
    q75, q25 = np.percentile(kld_array, [75, 25])
    iqr = q75 - q25
    return iqr

# Calculate Skewness
def calculate_skewness(kld_values):
    kld_array = np.array(eval(kld_values))
    return skew(kld_array)

# Calculate Kurtosis
def calculate_kurtosis(kld_values):
    kld_array = np.array(eval(kld_values))
    return kurtosis(kld_array)

# Calculate Slope, Intercept, and R-squared of KLD Trend
def calculate_slope_intercept(kld_values):
    kld_array = np.array(eval(kld_values))
    x = np.arange(len(kld_array))
    slope, intercept, r_value, p_value, std_err = linregress(x, kld_array)
    return intercept
def calculate_rsquare(kld_values):
    kld_array = np.array(eval(kld_values))
    x = np.arange(len(kld_array))
    slope, intercept, r_value, p_value, std_err = linregress(x, kld_array)
    return r_value**2


# Calculate First Derivative of KLD
def calculate_first_derivative(kld_values):
    kld_array = np.array(eval(kld_values))
    return np.diff(kld_array)[0]


def subject_score(row):
  score = 0
  weights = {
    "subj2_comedy": 4,
    "subj2_romance": 22,
    "subj2_thriller": 4,
    "subj2_western": 8,
    "subj2_horror": 23,
    "subj2_history": 12,
    "subj2_others": 12,
  }
  for key in weights.keys():
    score += weights[key] * row[indexes[key]]
  return score

def get_author_lived(row):
  birth = row[indexes["authoryearofbirth"]]
  death = row[indexes["authoryearofdeath"]]
  return (death - birth)


for i, key in enumerate(data):
  indexes[key] = i

def get_indexes():
  for i, key in enumerate(data):
    indexes[key] = i
  return indexes

lang_index = {"['en', 'myn']": 1, "['zh', 'en']": 2, "['en']": 3}
def is_multilang(val):
  l = eval(val)
  return 1 if len(l) > 1 else 0

author_lived = []
for d in data.values:
  author_lived.append(get_author_lived(d))
data["author_lived"] = author_lived

subjects = []
author_count = {}

for d in data.values:
  subjects.append(d[indexes["subjects"]])
  author_count[d[indexes["author"]]] = author_count.get(d[indexes["author"]], 0) + 1

yod_mul_numbook= []
for d in data.values:
  yod_mul_numbook.append(d[indexes["authoryearofdeath"]] / author_count[d[indexes["author"]]])
#print(len(set(subjects)))
#print(author_count)
data["yod_mul_numbook"] = yod_mul_numbook

def get_vowel_count(str):
  count = 0
      
  # Creating a set of vowels
  vowel = set("aeiouAEIOU")
    
  # Loop to traverse the alphabet
  # in the given string
  for alphabet in str:
    
      # If alphabet is present
      # in set vowel
      if alphabet in vowel:
          count = count + 1
  return count

def get_author_count(author):
  return author_count.get(author, 1)

def is_multiauth(author):
  return 1 if author_count.get(author, 1) > 1 else 0
def get_author_len(author):
  return (get_vowel_count(author) / len(author)) * get_author_count(author)

# Clean data

#print(data.isnull().sum())
#print(np.isinf(data).sum())


#data = data.replace([np.inf, -np.inf], np.nan).dropna()
data = data.dropna()


#kld_scores["kld_values"] = kld_scores["kld_values"].apply(eval)
#print(kld_scores)
def get_line_number(percent, total):
    return (percent * total) // 100

def get_partitioning(sentences):
  total = len(sentences)
  if len(sentences) == 1:
    return [sentences, sentences, sentences]
  
  if len(sentences) == 2:
    return [[sentences[0]], [sentences[1]], [sentences[1]]]
  if len(sentences) == 3:
    return [[sentences[0]], [sentences[1]], [sentences[2]]]
  if len(sentences) == 4:
    return [[sentences[0]], [sentences[1], sentences[2]], sentences[3]]
  else:
    intro_start, intro_end = 0, get_line_number(15, total)
    body_start, body_end = intro_end, get_line_number(80, total)
    conclusion_start, conclusion_end = body_end, total
    return [sentences[intro_start:intro_end], sentences[body_start:body_end], sentences[conclusion_start: conclusion_end]]

def diff_first_and_last_reveal(kld_list):
  kld_array = np.array(eval(kld_list))
  intro, body, last = get_partitioning(kld_array)
  #return abs(total_positive_changes(list(intro).__repr__()) - total_positive_changes(list(last).__repr__()))
  return abs(sum(intro) - sum(last))

def first_reveal_sum_abv_avg(kld_list):
  kld_array = eval(kld_list)
  intro, body, last = get_partitioning(kld_array)
  #return abs(total_positive_changes(list(intro).__repr__()) - total_positive_changes(list(last).__repr__()))
  return sum(intro)/np.mean(kld_array)
def last_reveal_sum_abv_avg(kld_list):
  kld_array = eval(kld_list)
  intro, body, last = get_partitioning(kld_array)
  #return abs(total_positive_changes(list(intro).__repr__()) - total_positive_changes(list(last).__repr__()))
  return 1 if sum(last) > np.mean(kld_array) else 0
def body_reveal_sum_abv_avg(kld_list):
  kld_array = eval(kld_list)
  intro, body, last = get_partitioning(kld_array)
  #return abs(total_positive_changes(list(intro).__repr__()) - total_positive_changes(list(last).__repr__()))
  return 1 if sum(body) > np.mean(kld_array) else 0

def kurt_structure(kld_list):
  kld_array = np.array(eval(kld_list))
  intro, body, last = get_partitioning(kld_array)
  i = kurtosis(intro)
  b = kurtosis(body)
  l = kurtosis(last)
  val = 0
  if i > b and i > l:
    val = 1
  elif b > i and b > l:
    val = -1
  else:
    val = 0

  return val

def last_sentiment(kld_list):
  kld_array = np.array(eval(kld_list))
  intro, body, last = get_partitioning(kld_array)
  return sum(intro) - sum(last)

def body_sentiment(kld_list):
  kld_array = np.array(eval(kld_list))
  intro, body, last = get_partitioning(kld_array)
  return sum(intro) - sum(last)

data['std_kld'] = data['kld_values'].apply(standard_deviation)
data['diff_first_last'] = data['kld_values'].apply(diff_first_and_last_reveal)
data['first_reveal_sum'] = data['kld_values'].apply(first_reveal_sum_abv_avg)
data['last_reveal_sum'] = data['kld_values'].apply(last_reveal_sum_abv_avg)
data['body_reveal_sum'] = data['kld_values'].apply(body_reveal_sum_abv_avg)
data['kurt_structure'] = data['kld_values'].apply(kurt_structure)
data['avg_kld'] = data['kld_values'].apply(avg_kld)
data['median_kld'] = data['kld_values'].apply(median_kld)
data['total_positive_changes'] = data['kld_values'].apply(total_positive_changes)
data['total_negative_changes'] = data['kld_values'].apply(total_negative_changes)
data['cv_kld'] = data['kld_values'].apply(coefficient_of_variation)
data['iqr'] = data['kld_values'].apply(calculate_iqr)
data['first_derivative'] = data['kld_values'].apply(calculate_first_derivative)
data['kurt'] = data['kld_values'].apply(calculate_kurtosis)
data['sqew'] = data['kld_values'].apply(calculate_skewness)
data['slope_intercept'] = data['kld_values'].apply(calculate_slope_intercept)
data['slope_kld'] = data['kld_values'].apply(slope_of_trend)
data['rsq'] = data['kld_values'].apply(calculate_rsquare)
#data['author_count'] = data['author'].apply(get_author_count)
data['author_len'] = data['author'].apply(get_author_len)
data['multiauth'] = data['author'].apply(is_multiauth)


max_min_sentiment= []
newFrequentReveal = []
slope_emotion = []
dynamic_information = []
firstLastAvgEmotion = []
skewAndVolatile = []
kurt_emotion = []
easeOfReading = []
firstEmotion = []

for d in data.values:
  indexes = get_indexes()
  kld = eval(d[indexes["kld_values"]])
  max_min_sentiment.append((max(kld) * min(kld)) * d[indexes["sentiment_vol"]])
  newFrequentReveal.append(d[indexes["total_positive_changes"]]/ (d[indexes['wordcount']]/d[indexes["speed"]]))
  slope_emotion.append(d[indexes["slope_kld"]] / d[indexes["sentiment_vol"]])
  firstLastAvgEmotion.append(d[indexes["diff_first_last"]] * d[indexes["sentiment_avg"]])
  skewAndVolatile.append(d[indexes["sqew"]] * d[indexes["sentiment_vol"]])
  kurt_emotion.append(d[indexes["kurt"]] / d[indexes["sentiment_vol"]])
  easeOfReading.append(((d[indexes["avg_kld"]])*(d[indexes['wordcount']]/d[indexes["speed"]]))/d[indexes["sentiment_vol"]])
  firstEmotion.append(d[indexes["slope_intercept"]]*d[indexes['sentiment_vol']]) # Captures sentiment volatility
  #dynamic_information.append()
#print(len(set(subjects)))
#print(author_count)
data["max_min_sentiment"] = max_min_sentiment
data["newFrequentRevealOverTime"] = newFrequentReveal
data["slope_emotion"] = slope_emotion
data["firstLastAvgEmotion"] = firstLastAvgEmotion
data["skewAndVolatile"] = skewAndVolatile
data["kurt_emotion"] = kurt_emotion
data["easeOfReading"] = easeOfReading
data["firstEmotion"] = firstEmotion



X = data[[
  #'std_kld', 
          #'slope_kld',
           #"sentiment_avg",
           #"sentiment_vol", 
           #"subj2_comedy",
    "subj2_romance",
    "subj2_thriller",
    "subj2_western",
    "subj2_horror",
    "subj2_history",
    "subj2_others",
    #"authoryearofbirth",
    #"authoryearofdeath",
    #"multiauth",
    #"yod_mul_numbook",
    #"author_lived"
    #"multilang"
    #"wordcount"
    #"author_len",
    #'author_count',
    'max_min_sentiment',
    #"std_dev_vol",
    #"total_positive_changes",
    "slope_emotion",
    "iqr",
    "first_derivative",
    "kurt",
    #"sqew",
    #"slope_intercept",
    "rsq",
    #"diff_first_last",
    "firstLastAvgEmotion",
    "first_reveal_sum",
    "last_reveal_sum",
    "body_reveal_sum",
    "skewAndVolatile",
    #"kurt_structure",
    #"kurt_emotion",
    "easeOfReading",
    "firstEmotion",
    "newFrequentRevealOverTime",
    ]]
y = data['downloads']


new = data[[
  "downloads",
  'std_kld', #1
  'slope_kld', 
  "total_positive_changes",
  #"total_negative_changes",
   # "slope_emotion",
    "iqr",
    "first_derivative", #1
    "kurt",
    "sqew",
    "slope_intercept", #1
    "rsq",
  ]
].copy()

new[[
  'std_kld', #1
  'slope_kld', 
  "total_positive_changes",
  #"total_negative_changes",
   # "slope_emotion",
    "iqr",
    "first_derivative", #1
    "kurt",
    "sqew",
    "slope_intercept", #1
    "rsq",
]]  = 1/new[[
  'std_kld', #1
  'slope_kld', 
  "total_positive_changes",
  #"total_negative_changes",
   # "slope_emotion",
    "iqr",
    "first_derivative", #1
    "kurt",
    "sqew",
    "slope_intercept", #1
    "rsq",
]]
'''new[[
  'std_kld', #1
  'slope_kld', 
  "total_positive_changes",
  #"total_negative_changes",
   # "slope_emotion",
    "iqr",
    "first_derivative", #1
    "kurt",
    "sqew",
    "slope_intercept", #1
    "rsq",
]]  *= 1000000'''



print(new.head())
sns.pairplot(data=new)
plt.savefig("inverse_result.pdf")


pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)

# Add constant to X for intercept
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X)
results = model.fit()

# Print summary of regression results
print(results.summary())




vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)




# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform LASSO with cross-validation
lasso = LassoCV(cv=5).fit(X_scaled, y)

# Extract the coefficients
lasso_coef = lasso.coef_

# Create a DataFrame to see which variables are selected
lasso_results = pd.DataFrame({'Variable': X.columns, 'Coefficient': lasso_coef})
print(lasso_results[lasso_results['Coefficient'] != 0])