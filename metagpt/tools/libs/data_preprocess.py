from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from metagpt.tools.tool_registry import register_tool

TAGS = ["data preprocessing", "machine learning"]


class MLProcess:
    def fit(self, df: pd.DataFrame):
        """
        训练模型，用于后续转换。

        参数:
            df (pd.DataFrame): 输入的 DataFrame。
        """
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用已训练的模型转换输入的 DataFrame。

        参数:
            df (pd.DataFrame): 输入的 DataFrame。

        返回:
            pd.DataFrame: 转换后的 DataFrame。
        """
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        训练并转换输入的 DataFrame。

        参数:
            df (pd.DataFrame): 输入的 DataFrame。

        返回:
            pd.DataFrame: 转换后的 DataFrame。
        """
        self.fit(df)
        return self.transform(df)


class DataPreprocessTool(MLProcess):
    """
    执行数据预处理操作。
    """

    def __init__(self, features: list):
        """
        初始化 self。

        参数:
            features (list): 需要处理的列名。
        """
        self.features = features
        self.model = None  # 由具体子类填充

    def fit(self, df: pd.DataFrame):
        if len(self.features) == 0:
            return
        self.model.fit(df[self.features])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.features) == 0:
            return df
        new_df = df.copy()
        new_df[self.features] = self.model.transform(new_df[self.features])
        return new_df


@register_tool(tags=TAGS)
class FillMissingValue(DataPreprocessTool):
    """
    使用简单策略填充缺失值。
    """

    def __init__(self, features: list, strategy: Literal["mean", "median", "most_frequent", "constant"] = "mean", fill_value=None):
        """
        初始化 self。

        参数:
            features (list): 需要处理的列名。
            strategy (Literal["mean", "median", "most_frequent", "constant"], optional): 填充策略，注意 'mean' 和 'median' 只能用于数值类型特征。默认 'mean'。
            fill_value (int, optional): 填充值，用于替换所有缺失值。默认 None。
        """
        self.features = features
        self.model = SimpleImputer(strategy=strategy, fill_value=fill_value)


@register_tool(tags=TAGS)
class MinMaxScale(DataPreprocessTool):
    """
    通过缩放每个特征到 (0, 1) 范围来转换特征。
    """

    def __init__(self, features: list):
        self.features = features
        self.model = MinMaxScaler()


@register_tool(tags=TAGS)
class StandardScale(DataPreprocessTool):
    """
    通过去除均值并缩放到单位方差来标准化特征。
    """

    def __init__(self, features: list):
        self.features = features
        self.model = StandardScaler()


@register_tool(tags=TAGS)
class MaxAbsScale(DataPreprocessTool):
    """
    通过最大绝对值缩放每个特征。
    """

    def __init__(self, features: list):
        self.features = features
        self.model = MaxAbsScaler()


@register_tool(tags=TAGS)
class RobustScale(DataPreprocessTool):
    """
    使用对异常值具有鲁棒性的统计数据来缩放特征。
    """

    def __init__(self, features: list):
        self.features = features
        self.model = RobustScaler()


@register_tool(tags=TAGS)
class OrdinalEncode(DataPreprocessTool):
    """
    将类别特征编码为有序整数。
    """

    def __init__(self, features: list):
        self.features = features
        self.model = OrdinalEncoder()


@register_tool(tags=TAGS)
class OneHotEncode(DataPreprocessTool):
    """
    对指定的类别列应用独热编码，原始列将被删除。
    """

    def __init__(self, features: list):
        self.features = features
        self.model = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ts_data = self.model.transform(df[self.features])
        new_columns = self.model.get_feature_names_out(self.features)
        ts_data = pd.DataFrame(ts_data, columns=new_columns, index=df.index)
        new_df = df.drop(self.features, axis=1)
        new_df = pd.concat([new_df, ts_data], axis=1)
        return new_df


@register_tool(tags=TAGS)
class LabelEncode(DataPreprocessTool):
    """
    对指定的类别列进行标签编码，直接在原始数据上修改。
    """

    def __init__(self, features: list):
        """
        初始化 self。

        参数:
            features (list): 需要标签编码的类别列。
        """
        self.features = features
        self.le_encoders = []

    def fit(self, df: pd.DataFrame):
        if len(self.features) == 0:
            return
        for col in self.features:
            le = LabelEncoder().fit(df[col].astype(str).unique().tolist() + ["unknown"])
            self.le_encoders.append(le)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.features) == 0:
            return df
        new_df = df.copy()
        for i in range(len(self.features)):
            data_list = df[self.features[i]].astype(str).tolist()
            for unique_item in np.unique(df[self.features[i]].astype(str)):
                if unique_item not in self.le_encoders[i].classes_:
                    data_list = ["unknown" if x == unique_item else x for x in data_list]
            new_df[self.features[i]] = self.le_encoders[i].transform(data_list)
        return new_df


def get_column_info(df: pd.DataFrame) -> dict:
    """
    分析 DataFrame 并根据数据类型将列分类。

    参数:
        df (pd.DataFrame): 需要分析的 DataFrame。

    返回:
        dict: 包含四个键（'Category'，'Numeric'，'Datetime'，'Others'）的字典。
              每个键对应一个包含该类别列名的列表。
    """
    column_info = {
        "Category": [],
        "Numeric": [],
        "Datetime": [],
        "Others": [],
    }
    for col in df.columns:
        data_type = str(df[col].dtype).replace("dtype('", "").replace("')", "")
        if data_type.startswith("object"):
            column_info["Category"].append(col)
        elif data_type.startswith("int") or data_type.startswith("float"):
            column_info["Numeric"].append(col)
        elif data_type.startswith("datetime"):
            column_info["Datetime"].append(col)
        else:
            column_info["Others"].append(col)

    if len(json.dumps(column_info)) > 2000:
        column_info["Numeric"] = column_info["Numeric"][0:5] + ["Too many cols, omission here..."]
    return column_info
