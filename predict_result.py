import numpy as np
import pandas as pd
import json

# 读取保存的特征名称
features_path = 'feature_names.json'
with open(features_path, 'r') as file:
    saved_feature_names = json.load(file)

def gen_new_col_name(df, indicator = 's__', notdo = True):
    new_column_names = {}
    if notdo:
        for col in df.columns:
            new_column_names[col] = col.replace('[' , '').replace(']' , '').replace('<' , '').replace('>' , '').replace(' ' , '_')
        return new_column_names
    for col in df.columns:
        # 根据 '|' 切分列标题
        if (indicator  not in col):
            new_column_names[col] = col.replace('[' , '').replace(']' , '').replace('<' , '').replace('>' , '').replace(' ' , '_')
            continue
        new_col = ''
        parts = col.split('|')
        for part in parts:
            if part.startswith(indicator):
                break
            new_col += part + '|'
        new_column_names[col] = new_col[:-1].replace('[' , '').replace(']' , '').replace('<' , '').replace('>' , '').replace(' ' , '_')
    return new_column_names
# 函数来读取并预处理新数据


def preprocess_new_data(file_path, saved_feature_names, indicator='s__'):
    # 读取数据
    df = pd.read_csv(file_path, sep='\t')

    
    # 应用预处理逻辑
    # 调用之前定义的 gen_new_col_name 函数来获取新列名
    new_column_names = gen_new_col_name(df, indicator, notdo=False)
    df = df.rename(columns=new_column_names)
    df = df.groupby(by=df.columns, axis=1).sum()


    # 创建一个新的DataFrame，初始化为0，确保所有保存的特征都包括在内
    aligned_df = pd.DataFrame(0, index=np.arange(len(df)), columns=saved_feature_names)

    # 填充新DataFrame中存在于原始数据中的特征值
    for feature in df.columns:
        if feature in aligned_df.columns:
            aligned_df[feature] = df[feature]
    # 计算每一行的总和
    row_sums = aligned_df.sum(axis=1)

    # 使用apply函数计算每个细菌的比例
    aligned_df.iloc[:, :-2] = aligned_df.apply(lambda x: x / row_sums)

    return aligned_df

# 假设你的新数据文件路径
new_data_path = 'test.txt'

# 处理新数据
processed_new_data = preprocess_new_data(new_data_path, saved_feature_names)

# load model
import joblib
model = joblib.load('xgboost_model.pkl')

# 预测新数据
predictions = model.predict(processed_new_data)
# save result to excel

result = pd.DataFrame(predictions, columns=['result'])
result.to_excel('result.xlsx', index=False)