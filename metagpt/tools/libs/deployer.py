from metagpt.tools.tool_registry import register_tool


# 一个未实现的工具，预留用于将本地服务部署到公共环境
@register_tool(
    include_functions=[
        "deploy_to_public",
    ]
)
class Deployer:
    """将本地服务部署到公共环境。仅用于最终部署，开发和测试过程中不应使用此工具。"""

    async def static_server(self, src_path: str) -> str:
        """该函数将在远程服务中实现。"""
        return "http://127.0.0.1:8000/index.html"

    async def deploy_to_public(self, dist_dir: str):
        """
        将网页项目部署到公共环境。
        参数：
            dist_dir (str): 网页项目的 dist 目录，通常是运行 build 后生成的目录。
        示例：
            deployer = Deployer("2048_game/dist")
        """
        # 获取静态资源的 URL
        url = await self.static_server(dist_dir)
        # 返回部署成功的提示信息
        return "项目已部署到: " + url + "\n 部署成功！"
