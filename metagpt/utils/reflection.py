"""class tools, including method inspection, class attributes, inheritance relationships, etc."""


def check_methods(C, *methods):
    """
    检查类是否具有指定的方法，借鉴自 _collections_abc 模块。

    在实现隐式接口时非常有用，例如定义抽象类，使用 isinstance 进行判断时不需要继承。

    参数:
        C (type): 要检查的类。
        methods (str): 要检查的方法名，多个方法可以作为参数传入。

    返回:
        bool: 如果类包含所有指定的方法，返回 True；否则返回 NotImplemented。
    """
    mro = C.__mro__  # 获取类的继承顺序链（Method Resolution Order）

    # 遍历所有给定的方法
    for method in methods:
        # 检查每个父类是否包含该方法
        for B in mro:
            if method in B.__dict__:  # 如果方法存在于当前类的字典中
                if B.__dict__[method] is None:  # 如果方法是 None，则返回 NotImplemented
                    return NotImplemented
                break  # 找到方法后跳出父类遍历
        else:
            # 如果在所有父类中都没有找到方法，返回 NotImplemented
            return NotImplemented

    return True  # 如果所有方法都被找到，返回 True
