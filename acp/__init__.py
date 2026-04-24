"""ACP - Application Control Protocol"""

__version__ = "0.1.0"


def __getattr__(name: str):
    """延迟导入，避免循环导入和 -m 运行时的 RuntimeWarning。"""
    if name == "ACP":
        from acp.main import ACP  # noqa: PLC0415
        return ACP
    raise AttributeError(f"module 'acp' has no attribute {name!r}")


__all__ = ["ACP", "__version__"]
