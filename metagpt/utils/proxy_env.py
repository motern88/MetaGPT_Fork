import os


def get_proxy_from_env():
    # 初始化代理配置字典
    proxy_config = {}
    server = None

    # 遍历常见的代理环境变量
    for i in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        # 如果找到代理服务器地址，设置server变量
        if os.environ.get(i):
            server = os.environ.get(i)

    # 如果找到代理服务器地址，添加到配置字典
    if server:
        proxy_config["server"] = server

    # 获取不需要使用代理的地址
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")

    # 如果设置了不使用代理的地址，添加到配置字典
    if no_proxy:
        proxy_config["bypass"] = no_proxy

    # 如果没有配置任何代理，设置为None
    if not proxy_config:
        proxy_config = None

    # 返回代理配置
    return proxy_config
