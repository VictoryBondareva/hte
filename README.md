# hte
Пример использования

```python
#Генерация датасета для тестирования
def mu0(x):
    return (-1)*np.array(x)**2 - 3 + np.random.normal(0, 0.5)
def mu1(x):
    return np.array(x)**2 + np.random.normal(0, 0.5)

def generate_simple_example(n, e):
    X_sample = np.random.sample(n)
    dataset = pd.DataFrame(X_sample)
    Y_0 = np.zeros((len(X_sample)))
    Y_1 = np.zeros((len(X_sample)))
    for i, x in enumerate(X_sample): 
        Y_0[i] = mu0(x)
        Y_1[i] = mu1(x)

    T = np.zeros(len(X_sample)) #столбец в котором 1 - леченные, 0 - не леченные
    while T.any() < 1.0:
        T = bernoulli.rvs(e, size = (len(X_sample)))
    
    dataset['T'] = pd.Series(T, index = dataset.index)

    res_Y = []
    for b, y0, y1 in zip(T, Y_0, Y_1):
        res_Y.append(y0 if b == 0 else y1)

    dataset['Y'] = pd.Series(res_Y, index = dataset.index)
    return dataset
print('Генерация...')
dataset = generate_simple_example(100, 0.1)

#Тестирование
def predict_simple_example(dataset = generate_simple_example(100, 0.1)):
    hte = HTE()
    # print(dataset)
    hte.fit(dataset, kNN = int(np.sqrt(len(dataset))), is_treatment = 'T')
    res_predict_data = pd.DataFrame(columns = ['X&X', 'realy_delta_Y', 'my_delta_Y'])
    
    #сгенерируем датасет для проверки (функции те же), но датасет отличен от обучаемого
    X_val = []
    test_dataset = generate_simple_example(int(len(dataset)*0.4), 0.1)
    realy_delta_Y = []
    for i, x in test_dataset.iterrows():
        if x['T'] == 0:
            realy_delta_Y.append(mu1(x[0]) - x['Y'])
        if x['T'] == 1:
            realy_delta_Y.append(x['Y'] - mu0(x[0]))
    X_val = np.array(list([test_dataset[0].values]) + list([test_dataset[0].values])).T
    # print(X_val)
    my_delta_Y = hte.predict(X_val)
    # print(*X_val)
    # print(my_delta_Y)
    # print(realy_delta_Y)
    for i, x, ry, my in zip(range(len(my_delta_Y)), X_val, realy_delta_Y, my_delta_Y):
    #     print(i, x, ry, my)
        res_predict_data.loc[i] = [x, ry, my]
    return res_predict_data
res_predict_data = predict_simple_example(dataset)
```
