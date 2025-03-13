#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : registry to store Dynamic Model from ActionNode.create_model_class to keep it as same Class
#           with same class name and mapping

from functools import wraps

action_outcls_registry = dict()


def register_action_outcls(func):
    """
    由于 `create_model` 返回的类即使有相同的类名和映射，也会返回不同的类。
    为了进行比较，使用 outcls_id 来标识具有相同类名和字段定义的相同类。
    """

    @wraps(func)
    def decorator(*args, **kwargs):
        """
        示例数组:
            [<class 'metagpt.actions.action_node.ActionNode'>, 'test', {'field': (str, Ellipsis)}]
        """
        arr = list(args) + list(kwargs.values())
        """
        outcls_id 示例:
            "<class 'metagpt.actions.action_node.ActionNode'>_test_{'field': (str, Ellipsis)}"
        """
        for idx, item in enumerate(arr):
            if isinstance(item, dict):
                arr[idx] = dict(sorted(item.items()))  # 排序字典，确保一致性
        outcls_id = "_".join([str(i) for i in arr])  # 拼接成一个唯一的标识符

        # 去除类型影响，确保一致性
        outcls_id = outcls_id.replace("typing.List", "list").replace("typing.Dict", "dict")

        if outcls_id in action_outcls_registry:
            return action_outcls_registry[outcls_id]  # 如果已有该标识符的类，直接返回缓存的类

        out_cls = func(*args, **kwargs)  # 调用原函数获取类
        action_outcls_registry[outcls_id] = out_cls  # 缓存类到注册表中
        return out_cls

    return decorator
