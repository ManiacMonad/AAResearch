from typing import TypeVar, Generic, Union

T = TypeVar("T")


class Optional(Generic[T]):
    """
    回傳特殊值
    設計目標:製作簡單程式碼
    """

    available: bool
    value: T

    def __init__(self, value: T):
        self.available = value is not None
        self.value = value
