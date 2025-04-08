"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects
2. jaxtyping (https://github.com/google/jaxtyping/blob/main/API.md) for PyTorch tensors

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
2. Runtime type checking with typeguard (install with pip and triggered at runtime, mainly for tensor dtype and shape checking)
"""
"""
这个 `typing.py` 文件用于项目的类型注解。
它引入Python的类型提示及jaxtyping来标注PyTorch张量。
提及可进行静态(mypy)与运行时(typeguard)两种类型检查。
定义了基础类型、张量数据类型、配置类型、PyTorch张量类型,并提供运行时类型检查装饰器。
文件还自定义了 `FuncArgs` 类型，并给出校验方法 。 
"""
# Basic types
from typing import (
    Any, #表示任意类型
    Callable, #表示可调用对象（函数、方法）
    Dict,  #集合类型注解
    Iterable,
    List, #集合类型注解
    Literal,   #表示固定的几个可能值
    NamedTuple,
    NewType,
    Optional,  #表示可选类型（值或None
    Sized,
    Tuple,  #集合类型注解
    Type,
    TypedDict,  #带有特定字段类型的字典
    TypeVar,
    Union,  #表示多种可能类型之一
)

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
#张量数据类型注解
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt

# Config type
# 配置和张量类型
from omegaconf import DictConfig, ListConfig

# PyTorch Tensor type
from torch import Tensor

# Runtime type checking decorator
#运行时类型检查工具
from typeguard import typechecked as typechecker


# Custom types
class FuncArgs(TypedDict):
    """Type for instantiating a function with keyword arguments"""

    name: str
    kwargs: Dict[str, Any]

    @staticmethod  #静态方法装饰器
    #validate 是一个静态方法，用于验证输入的 variable 是否满足特定的格式要求。它会检查 variable 是否包含必要的键 "name" 和 "kwargs"，并且确保 "name" 的值是字符串类型，"kwargs" 的值是字典类型。如果验证通过，则返回原始的 variable；如果验证不通过，则会抛出相应的异常。
    def validate(variable):
        necessary_keys = ["name", "kwargs"]
        for key in necessary_keys:
            #使用 assert 语句检查该键是否存在于 variable 中。如果某个键不存在，assert 语句会抛出 AssertionError 异常，并附带相应的错误信息。    
            assert key in variable, f"Key {key} is missing in {variable}"
        if not isinstance(variable["name"], str):
            raise TypeError(
                f"Key 'name' should be a string, not {type(variable['name'])}"
            )
        if not isinstance(variable["kwargs"], dict):
            raise TypeError(
                f"Key 'kwargs' should be a dictionary, not {type(variable['kwargs'])}"
            )
        return variable
