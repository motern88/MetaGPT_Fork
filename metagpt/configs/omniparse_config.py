from metagpt.utils.yaml_model import YamlModel


class OmniParseConfig(YamlModel):
    """
    OmniParse 配置类。

    属性：
        api_key (str): 用于 API 访问的密钥。
        base_url (str): API 的基本 URL。
        timeout (int): 请求的超时时间，单位为秒，默认值为 600 秒。
    """

    api_key: str = ""  # 用于 API 访问的密钥
    base_url: str = ""  # API 的基本 URL
    timeout: int = 600  # 请求的超时时间，单位为秒
