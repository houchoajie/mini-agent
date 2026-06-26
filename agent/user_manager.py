"""
============================================================
用户管理器 — 注册、登录、密码安全存储、数据目录管理
============================================================

职责：
1. 用户注册（密码加盐哈希存储，绝不存明文）
2. 用户登录（密码验证）
3. 用户目录创建（自动创建 session/log/task 子目录）
4. 提供用户目录路径给其他组件（实现数据隔离）

密码安全策略：
    使用 salt$sha256 格式存储密码。
    每次注册生成随机 8 字节 salt（128 位），
    存储的是 salt + sha256(salt+password) 的拼接。
    即使数据库泄露，攻击者也无法逆向得到密码。

数据隔离：
    所有用户数据存储在 .agent_data/<username>/ 目录下，
    包含 session、log、task 三个子目录。
    不同用户的会话、日志、任务完全隔离。

文件结构：
    .agent_data/
    ├── userInfo.json           ← 用户凭证（全局）
    ├── <username>/             ← 用户数据目录
    │   ├── session/            ← 会话文件
    │   ├── log/                ← Trace 日志
    │   └── task/               ← 任务数据
"""

import json
import os
import sys
import hashlib
import secrets
from pathlib import Path
from datetime import datetime


DATA_DIR = Path(__file__).parent.parent / ".agent_data"
USER_FILE = DATA_DIR / "userInfo.json"


class UserManager:
    """
    用户管理器。

    职责：
    1. 用户注册（密码加盐哈希存储）
    2. 用户登录（密码验证）
    3. 用户目录创建（自动创建 session/log/task 子目录）
    4. 提供用户目录路径给其他组件

    数据文件: .agent_data/userInfo.json
    格式:
        {
            "version": 1,
            "users": {
                "zhangsan": {
                    "password_hash": "a1b2c3d4$e99a18c428cb38d5f260853678922e03",
                    "created_at": "2026-06-24 10:00:00"
                }
            }
        }

    目录结构:
        .agent_data/<username>/{session, log, task}/
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._users = self._load()

    def _load(self) -> dict:
        """从 JSON 文件加载用户数据。文件不存在或损坏时返回空字典。"""
        if not USER_FILE.exists():
            return {}
        try:
            with open(USER_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("users", {})
        except (json.JSONDecodeError, IOError):
            return {}

    def _save(self):
        """保存用户数据到 JSON 文件。"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            # 先写临时文件再原子替换，防止写入中途崩溃导致文件损坏
            tmp_file = USER_FILE.with_suffix(".json.tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump({"users": self._users, "version": 1}, f,
                          ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp_file.replace(USER_FILE)  # 原子替换
        except (IOError, OSError, json.JSONEncodeError) as e:
            print(f"❌ [USER] 用户数据保存失败: {e}", file=sys.stderr)

    def _hash_password(self, password: str, salt: str | None = None) -> str:
        """
        加盐 SHA-256 哈希。

        格式：salt$sha256_hex
        其中 salt 是 8 字节随机 hex（128 位），
        sha256_hex 是 sha256(salt + password) 的 hexdigest。

        为什么用 salt + password 而非 password + salt：
        - 前者的彩虹表攻击成本更高（salt 固定长度，password 变长）
        - 但实践中两者差异不大，主要是约定俗成
        """
        if salt is None:
            salt = secrets.token_hex(8)
        h = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}${h}"

    def _verify_password(self, password: str, stored: str) -> bool:
        """
        验证密码。

        从 stored 中提取 salt（$ 之前的部分），
        用同样的 salt 对 password 重新哈希，比较结果。
        """
        if "$" not in stored:
            return False
        salt, _ = stored.split("$", 1)
        return self._hash_password(password, salt) == stored

    def register(self, username: str, password: str) -> tuple[bool, str]:
        """
        注册新用户。

        校验规则：
        - 用户名和密码不能为空
        - 密码长度至少 4 位
        - 用户名不能重复

        注册成功后自动创建用户目录及子目录。

        Returns:
            (是否成功, 消息)
        """
        if not username or not password:
            return False, "用户名和密码不能为空"
        if len(password) < 4:
            return False, "密码长度至少 4 位"
        if username in self._users:
            return False, "用户名已存在"

        self._users[username] = {
            "password_hash": self._hash_password(password),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save()

        # 创建用户目录结构（session/log/task 子目录）
        try:
            user_dir = self.get_user_dir(username)
            for sub in ["session", "log", "task"]:
                (user_dir / sub).mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # 目录创建失败不影响注册结果，但打印警告
            print(f"⚠️ [USER] 用户目录创建失败 ({username}): {e}", file=sys.stderr)

        return True, "注册成功"

    def login(self, username: str, password: str) -> tuple[bool, str]:
        """用户登录验证。"""
        if username not in self._users:
            return False, "用户名不存在"
        if self._verify_password(password, self._users[username]["password_hash"]):
            return True, "登录成功"
        return False, "密码错误"

    def get_user_dir(self, username: str) -> Path:
        """获取用户数据目录: .agent_data/<username>/。"""
        return DATA_DIR / username

    def has_users(self) -> bool:
        """是否有已注册用户。"""
        return len(self._users) > 0
