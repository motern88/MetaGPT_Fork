from enum import Enum


class CompressType(Enum):
    """
    消息压缩类型。用于在超出令牌限制时压缩消息。
    - "": 无压缩。默认值。
    - "post_cut_by_msg": 保留尽可能多的最新消息。
    - "post_cut_by_token": 保留尽可能多的最新消息，并截断最早的适合消息。
    - "pre_cut_by_msg": 保留尽可能多的最早消息。
    - "pre_cut_by_token": 保留尽可能多的最早消息，并截断最新的适合消息。
    """

    NO_COMPRESS = ""            # 无压缩
    POST_CUT_BY_MSG = "post_cut_by_msg"  # 按消息数保留最新消息
    POST_CUT_BY_TOKEN = "post_cut_by_token"  # 按令牌数保留最新消息
    PRE_CUT_BY_MSG = "pre_cut_by_msg"  # 按消息数保留最早消息
    PRE_CUT_BY_TOKEN = "pre_cut_by_token"  # 按令牌数保留最早消息

    def __missing__(self, key):
        """如果传入的类型值不存在，返回默认值 NO_COMPRESS"""
        return self.NO_COMPRESS

    @classmethod
    def get_type(cls, type_name):
        """根据传入的类型名称返回相应的枚举成员，找不到时返回 NO_COMPRESS"""
        for member in cls:
            if member.value == type_name:
                return member
        return cls.NO_COMPRESS

    @classmethod
    def cut_types(cls):
        """返回包含 'cut' 字符串的所有枚举成员"""
        return [member for member in cls if "cut" in member.value]
