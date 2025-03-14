#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/17 10:33
# @Author  : lidanyang
# @File    : feature_engineering.py
# @Desc    : Feature Engineering Tools
from __future__ import annotations

import itertools

# import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.core.dtypes.common import is_object_dtype
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures

from metagpt.tools.libs.data_preprocess import MLProcess
from metagpt.tools.tool_registry import register_tool

TAGS = ["feature engineering", "machine learning"]


@register_tool(tags=TAGS)
class PolynomialExpansion(MLProcess):
    """
    从选定的数值列中为输入 DataFrame 添加多项式和交互特征。
    """

    def __init__(self, cols: list, label_col: str, degree: int = 2):
        """
        初始化

        参数:
            cols (list): 用于多项式扩展的列。
            label_col (str): 标签列名称。
            degree (int, 可选): 多项式特征的度数。默认为 2。
        """
        self.cols = cols
        self.degree = degree
        self.label_col = label_col
        if self.label_col in self.cols:
            self.cols.remove(self.label_col)
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)

    def fit(self, df: pd.DataFrame):
        if len(self.cols) == 0:
            return
        if len(self.cols) > 10:
            corr = df[self.cols + [self.label_col]].corr()
            corr = corr[self.label_col].abs().sort_values(ascending=False)
            self.cols = corr.index.tolist()[1:11]

        self.poly.fit(df[self.cols].fillna(0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(self.cols) == 0:
            return df
        ts_data = self.poly.transform(df[self.cols].fillna(0))
        column_name = self.poly.get_feature_names_out(self.cols)
        ts_data = pd.DataFrame(ts_data, index=df.index, columns=column_name)
        new_df = df.drop(self.cols, axis=1)
        new_df = pd.concat([new_df, ts_data], axis=1)
        return new_df


@register_tool(tags=TAGS)
class CatCount(MLProcess):
    """
    将分类列的值计数作为新特征添加。
    """

    def __init__(self, col: str):
        """
        初始化

        参数:
            col (str): 要进行值计数的列。
        """
        self.col = col
        self.encoder_dict = None

    def fit(self, df: pd.DataFrame):
        self.encoder_dict = df[self.col].value_counts().to_dict()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        new_df[f"{self.col}_cnt"] = new_df[self.col].map(self.encoder_dict)
        return new_df


@register_tool(tags=TAGS)
class TargetMeanEncoder(MLProcess):
    """
    通过标签列的均值对分类列进行编码，并将结果作为新特征添加。
    """

    def __init__(self, col: str, label: str):
        """
        初始化

        参数:
            col (str): 要进行均值编码的列。
            label (str): 预测标签列。
        """
        self.col = col
        self.label = label
        self.encoder_dict = None

    def fit(self, df: pd.DataFrame):
        self.encoder_dict = df.groupby(self.col)[self.label].mean().to_dict()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        new_df[f"{self.col}_target_mean"] = new_df[self.col].map(self.encoder_dict)
        return new_df


@register_tool(tags=TAGS)
class KFoldTargetMeanEncoder(MLProcess):
    """
    通过 K 折均值编码，将一个分类列的值与标签列进行编码，生成一个新特征。
    """

    def __init__(self, col: str, label: str, n_splits: int = 5, random_state: int = 2021):
        """
        初始化

        参数:
            col (str): 要进行 K 折均值编码的列。
            label (str): 预测标签列。
            n_splits (int, 可选): K 折的数量。默认为 5。
            random_state (int, 可选): 随机种子。默认为 2021。
        """
        self.col = col
        self.label = label
        self.n_splits = n_splits
        self.random_state = random_state
        self.encoder_dict = None

    def fit(self, df: pd.DataFrame):
        tmp = df.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        global_mean = tmp[self.label].mean()
        col_name = f"{self.col}_kf_target_mean"
        for trn_idx, val_idx in kf.split(tmp, tmp[self.label]):
            _trn, _val = tmp.iloc[trn_idx], tmp.iloc[val_idx]
            tmp.loc[tmp.index[val_idx], col_name] = _val[self.col].map(_trn.groupby(self.col)[self.label].mean())
        tmp[col_name].fillna(global_mean, inplace=True)
        self.encoder_dict = tmp.groupby(self.col)[col_name].mean().to_dict()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        new_df[f"{self.col}_kf_target_mean"] = new_df[self.col].map(self.encoder_dict)
        return new_df


@register_tool(tags=TAGS)
class CatCross(MLProcess):
    """
    添加两两交叉的特征，并将其转换为数值特征。
    """

    def __init__(self, cols: list, max_cat_num: int = 100):
        """
        初始化

        参数:
            cols (list): 要交叉的列，至少 2 列。
            max_cat_num (int, 可选): 每个交叉特征的最大类别数。默认为 100。
        """
        self.cols = cols
        self.max_cat_num = max_cat_num
        self.combs = []
        self.combs_map = {}

    @staticmethod
    def _cross_two(comb, df):
        """
        交叉两个列并将其转换为数值特征。

        参数:
            comb (tuple): 要交叉的列对。
            df (pd.DataFrame): 输入的 DataFrame。

        返回:
            tuple: 新列名和交叉特征映射。
        """
        new_col = f"{comb[0]}_{comb[1]}"
        new_col_combs = list(itertools.product(df[comb[0]].unique(), df[comb[1]].unique()))
        ll = list(range(len(new_col_combs)))
        comb_map = dict(zip(new_col_combs, ll))
        return new_col, comb_map

    def fit(self, df: pd.DataFrame):
        for col in self.cols:
            if df[col].nunique() > self.max_cat_num:
                self.cols.remove(col)
        self.combs = list(itertools.combinations(self.cols, 2))
        res = Parallel(n_jobs=4, require="sharedmem")(delayed(self._cross_two)(comb, df) for comb in self.combs)
        self.combs_map = dict(res)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        for comb in self.combs:
            new_col = f"{comb[0]}_{comb[1]}"
            _map = self.combs_map[new_col]
            new_df[new_col] = pd.Series(zip(new_df[comb[0]], new_df[comb[1]])).map(_map)
            # 将未知的值设置为新数字
            new_df[new_col].fillna(max(_map.values()) + 1, inplace=True)
            new_df[new_col] = new_df[new_col].astype(int)
        return new_df


@register_tool(tags=TAGS)
class GroupStat(MLProcess):
    """
    对DataFrame中的指定列进行聚合，按另一列分组，添加新的特征，命名为 '<agg_col>_<agg_func>_by_<group_col>'。
    """

    def __init__(self, group_col: str, agg_col: str, agg_funcs: list):
        """
        初始化函数。

        参数:
            group_col (str): 用于分组的列。
            agg_col (str): 执行聚合操作的列。
            agg_funcs (list): 要应用的聚合函数列表，如 ['mean', 'std']。每个函数必须是pandas支持的聚合函数。
        """
        self.group_col = group_col
        self.agg_col = agg_col
        self.agg_funcs = agg_funcs
        self.group_df = None

    def fit(self, df: pd.DataFrame):
        # 对数据按指定的列进行分组，并应用聚合函数
        group_df = df.groupby(self.group_col)[self.agg_col].agg(self.agg_funcs).reset_index()
        # 重命名列，形成新的列名
        group_df.columns = [self.group_col] + [
            f"{self.agg_col}_{agg_func}_by_{self.group_col}" for agg_func in self.agg_funcs
        ]
        self.group_df = group_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 将聚合结果与原始DataFrame合并
        new_df = df.merge(self.group_df, on=self.group_col, how="left")
        return new_df


@register_tool(tags=TAGS)
class SplitBins(MLProcess):
    """
    对连续数据进行分箱，直接返回整数编码的分箱标识符。
    """

    def __init__(self, cols: list, strategy: str = "quantile"):
        """
        初始化函数。

        参数:
            cols (list): 需要进行分箱的列。
            strategy (str, 可选): 用于定义分箱宽度的策略。可选策略包括['quantile', 'uniform', 'kmeans']，默认使用'quantile'。
        """
        self.cols = cols
        self.strategy = strategy
        self.encoder = None

    def fit(self, df: pd.DataFrame):
        # 使用指定的策略和分箱方式进行分箱
        self.encoder = KBinsDiscretizer(strategy=self.strategy, encode="ordinal")
        self.encoder.fit(df[self.cols].fillna(0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 对数据进行分箱转换
        new_df = df.copy()
        new_df[self.cols] = self.encoder.transform(new_df[self.cols].fillna(0))
        return new_df


# @register_tool(tags=TAGS)
class ExtractTimeComps(MLProcess):
    """
    从datetime列中提取时间组件，并将其作为新特征添加到DataFrame中。
    """

    def __init__(self, time_col: str, time_comps: list):
        """
        初始化函数。

        参数:
            time_col (str): 包含时间数据的列名。
            time_comps (list): 要提取的时间组件的列表。每个组件必须是 ['year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend'] 之一。
        """
        self.time_col = time_col
        self.time_comps = time_comps

    def fit(self, df: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        time_s = pd.to_datetime(df[self.time_col], errors="coerce")
        time_comps_df = pd.DataFrame()

        if "year" in self.time_comps:
            time_comps_df["year"] = time_s.dt.year
        if "month" in self.time_comps:
            time_comps_df["month"] = time_s.dt.month
        if "day" in self.time_comps:
            time_comps_df["day"] = time_s.dt.day
        if "hour" in self.time_comps:
            time_comps_df["hour"] = time_s.dt.hour
        if "dayofweek" in self.time_comps:
            time_comps_df["dayofweek"] = time_s.dt.dayofweek + 1
        if "is_weekend" in self.time_comps:
            time_comps_df["is_weekend"] = time_s.dt.dayofweek.isin([5, 6]).astype(int)
        new_df = pd.concat([df, time_comps_df], axis=1)
        return new_df


@register_tool(tags=TAGS)
class GeneralSelection(MLProcess):
    """
    删除所有包含NaN值的特征和只有一个唯一值的特征。
    """

    def __init__(self, label_col: str):
        self.label_col = label_col
        self.feats = []

    def fit(self, df: pd.DataFrame):
        feats = [f for f in df.columns if f != self.label_col]
        for col in df.columns:
            if df[col].isnull().sum() / df.shape[0] == 1:
                feats.remove(col)

            if df[col].nunique() == 1:
                feats.remove(col)

            if df.loc[df[col] == np.inf].shape[0] != 0 or df.loc[df[col] == np.inf].shape[0] != 0:
                feats.remove(col)

            if is_object_dtype(df[col]) and df[col].nunique() == df.shape[0]:
                feats.remove(col)

        self.feats = feats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 只返回选中的特征和标签列
        new_df = df[self.feats + [self.label_col]]
        return new_df


# skip for now because lgb is needed
# @register_tool(tags=TAGS)
class TreeBasedSelection(MLProcess):
    """
    基于树模型选择特征，并移除低重要性的特征。
    """

    def __init__(self, label_col: str, task_type: str):
        """
        初始化函数。

        参数:
            label_col (str): 标签列名。
            task_type (str): 任务类型，'cls'表示分类，'mcls'表示多分类，'reg'表示回归。
        """
        self.label_col = label_col
        self.task_type = task_type
        self.feats = None

    def fit(self, df: pd.DataFrame):
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 31,
        }

        if self.task_type == "cls":
            params["objective"] = "binary"
            params["metric"] = "auc"
        elif self.task_type == "mcls":
            params["objective"] = "multiclass"
            params["num_class"] = df[self.label_col].nunique()
            params["metric"] = "auc_mu"
        elif self.task_type == "reg":
            params["objective"] = "regression"
            params["metric"] = "rmse"

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cols = [f for f in num_cols if f not in [self.label_col]]

        dtrain = lgb.Dataset(df[cols], df[self.label_col])
        model = lgb.train(params, dtrain, num_boost_round=100)
        df_imp = pd.DataFrame({"feature_name": dtrain.feature_name, "importance": model.feature_importance("gain")})

        df_imp.sort_values("importance", ascending=False, inplace=True)
        df_imp = df_imp[df_imp["importance"] > 0]
        self.feats = df_imp["feature_name"].tolist()
        self.feats.append(self.label_col)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 只返回根据特征重要性选择的特征
        new_df = df[self.feats]
        return new_df


@register_tool(tags=TAGS)
class VarianceBasedSelection(MLProcess):
    """
    基于方差选择特征，移除低方差特征。
    """

    def __init__(self, label_col: str, threshold: float = 0):
        """
        初始化函数。

        参数：
            label_col (str): 标签列的名称。
            threshold (float, optional): 方差阈值。默认为0。
        """
        self.label_col = label_col  # 标签列的名称
        self.threshold = threshold  # 方差阈值
        self.feats = None  # 存储选择的特征列
        self.selector = VarianceThreshold(threshold=self.threshold)  # 方差选择器

    def fit(self, df: pd.DataFrame):
        """
        训练方差选择器。

        参数：
            df (pd.DataFrame): 输入的 DataFrame。
        """
        # 获取所有数值型特征列
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        # 排除标签列
        cols = [f for f in num_cols if f not in [self.label_col]]

        # 训练方差选择器
        self.selector.fit(df[cols])
        # 根据方差选择特征
        self.feats = df[cols].columns[self.selector.get_support(indices=True)].tolist()
        # 将标签列添加到特征列中
        self.feats.append(self.label_col)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对数据进行转换，返回选择的特征列。

        参数：
            df (pd.DataFrame): 输入的 DataFrame。

        返回：
            pd.DataFrame: 包含选择的特征的 DataFrame。
        """
        # 返回包含选择特征列的 DataFrame
        new_df = df[self.feats]
        return new_df
