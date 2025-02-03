
#  Процентный список пропущенных данных
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
missing_data_percent = df.isnull().mean()*100
missing_data_percent = missing_data_percent.astype(int)
missing_data_percent = missing_data_percent[missing_data_percent > 0].sort_values(inplace = False)
for column, percent in missing_data_percent.items():
  print(f'{column}:{percent}%')


# Гистограмма пропущенных данных
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/content/sberbank.csv')
missing = df.isnull().sum(axis=1)

plt.figure(figsize=(10, 6))
plt.hist(missing, color = 'blue', edgecolor = 'black',
         bins = range(int(missing.max())+ 2))
plt.title('Гистограмма количества пропущенных значений в строках')
plt.xlabel('Количество пропущенных значений')
plt.ylabel('Число строк')

plt.show()


#  Отбрасывание записей
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
missing_row = df.isnull().sum(axis=1)
threshold = 35
df_less_missing_rows = df[missing_row <= threshold]
print(df_less_missing_rows)


# Отбрасывание признаков
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
missing_data_percent = df.isnull().mean()*100
print(missing_data_percent)
cols_to_drop = df.drop(columns=['hospital_beds_raion'])
print(cols_to_drop.head())


# Внесение недостающих значений
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')

med = df['life_sq'].median()
print(med)
df['life_sq'] = df['life_sq'].fillna(med)

df['life_sq']


# Cразу для всех числовых признаков:
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')

numeric = df.select_dtypes(include=['number']).columns
df[numeric] = df[numeric].fillna(df[numeric].median())
print(df.isnull().sum())


# Для категориальных признаков:
import pandas as pd
import numpy as np

df = pd.read_csv('/content/sberbank.csv')
categori_col = df.select_dtypes(include=['object', 'category']).columns
for col in categori_col:
  value = df[col].mode()
  df[col] = df[col].fillna(value)
  print(df.isnull().sum())


#  Описательная статистика
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
df['life_sq'].describe()


# Неинформативные признаки
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
low_information_cols = []
for col in df.columns:
    value = df[col].mode()[0]
    count = (df[col] == value).sum()

    if count/len(df) > 0.95:
        low_information_cols.append({
            'Name':col,
            'Mode':value,
            'Percent':count/len(df)*100
        })
low_information_df = pd.DataFrame(low_information_cols)
print(low_information_df)


# Если после анализа причин получения повторяющихся значений вы пришли к выводу, что признак не несет полезной информации, используйте drop().
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
low_information_cols = []
for col in df.columns:
    if col in df.columns:
      low_information_cols.append(col)
df = df.drop(columns=low_information_cols)
print(df.columns)


# Дубликаты записей
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
df_dedupped = df.drop('id', axis=1).drop_duplicates()

# сравниваем формы старого и нового наборов
print(df.shape)
print(df_dedupped.shape)


# Другой распространенный способ вычисления дубликатов: по набору ключевых признаков.
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room']

df.fillna(-999).groupby(key)['id'].count().sort_values(ascending=False).head(20)


# Что делать с дубликатами?
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room']
df_dedupped2 = df.drop_duplicates(subset=key)

print(df.shape)
print(df_dedupped2.shape)

# Разные регистры символов
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
df['sub_area_lower'] = df['sub_area'].str.lower()
df['sub_area_lower'].value_counts(dropna=False)

# Опечатки
import pandas as pd
import difflib
df = pd.read_csv('/content/sberbank.csv')

df_city_ex = pd.DataFrame(data={'city': ['torontoo', 'toronto', 'tronto', 'vancouver', 'vancover', 'vancouvr', 'montreal', 'calgary']})
correct_cities = ['toronto', 'vancouver', 'montreal', 'calgary']

def correct_city(city_name):
  matches = difflib.get_close_matches(city_name, correct_cities, n=1, cutoff=0.8)
  return matches
df_city_ex['city_corrected'] = df_city_ex['city'].apply(correct_city)
result = []

for city in correct_cities:
    corrected_count = df_city_ex[df_city_ex['city_corrected'] == city].shape[0]
    incorrect_count = df_city_ex[df_city_ex['city'] != city].shape[0]

    result.append(pd.DataFrame({'city': [city], 'corrected_count': [corrected_count], 'incorrect_count': [incorrect_count]}))

result_df = pd.concat(result, ignore_index=True)

print(result_df)


#  Адреса
import pandas as pd

df = pd.read_csv('/content/sberbank.csv')
df_add_ex = pd.DataFrame(['123 MAIN St Apartment 15', '123 Main Street Apt 12   ', '543 FirSt Av', '  876 FIRst Ave.'], columns=['address'])

df_add_ex['address_std'] = df_add_ex['address'].str.lower()
df_add_ex['address_std'] = df_add_ex['address_std'].str.strip()
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\.', '')
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bstreet\\b', 'st')
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bapartment\\b', 'apt')
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bav\\b', 'ave')

df_add_ex