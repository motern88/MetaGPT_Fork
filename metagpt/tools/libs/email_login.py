from imap_tools import MailBox

from metagpt.tools.tool_registry import register_tool

# 定义一个字典，将邮箱域名映射到其 IMAP 服务器地址
IMAP_SERVERS = {
    "outlook.com": "imap-mail.outlook.com",  # Outlook
    "163.com": "imap.163.com",  # 163 邮箱
    "qq.com": "imap.qq.com",  # QQ 邮箱
    "gmail.com": "imap.gmail.com",  # Gmail
    "yahoo.com": "imap.mail.yahoo.com",  # Yahoo 邮箱
    "icloud.com": "imap.mail.me.com",  # iCloud 邮箱
    "hotmail.com": "imap-mail.outlook.com",  # Hotmail (与 Outlook 相同)
    "live.com": "imap-mail.outlook.com",  # Live (与 Outlook 相同)
    "sina.com": "imap.sina.com",  # 新浪邮箱
    "sohu.com": "imap.sohu.com",  # 搜狐邮箱
    "yahoo.co.jp": "imap.mail.yahoo.co.jp",  # 日本 Yahoo 邮箱
    "yandex.com": "imap.yandex.com",  # Yandex 邮箱
    "mail.ru": "imap.mail.ru",  # Mail.ru 邮箱
    "aol.com": "imap.aol.com",  # AOL 邮箱
    "gmx.com": "imap.gmx.com",  # GMX 邮箱
    "zoho.com": "imap.zoho.com",  # Zoho 邮箱
}


@register_tool(tags=["email login"])
def email_login_imap(email_address, email_password):
    """
    使用 imap_tools 包登录支持 IMAP 协议的邮箱，验证并返回账户对象。

    参数：
        email_address (str): 需要登录并关联的邮箱地址。
        email_password (str): 需要登录并关联的邮箱密码。

    返回：
        object: 成功连接到邮箱后返回的 imap_tools 的 MailBox 对象，包含有关该账户的各种信息（如邮箱等），
                如果登录失败则返回 None。
    """

    # 从邮箱地址中提取域名
    domain = email_address.split("@")[-1]

    # 获取正确的 IMAP 服务器地址
    imap_server = IMAP_SERVERS.get(domain)

    # 如果没有找到对应的 IMAP 服务器，则抛出异常
    assert imap_server, f"未找到 {domain} 的 IMAP 服务器。"

    # 尝试使用 imap_tools 登录邮箱
    mailbox = MailBox(imap_server).login(email_address, email_password)
    return mailbox
