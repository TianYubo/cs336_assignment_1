from typing import TypeVar
import numpy as np
import pytest
import os
from pathlib import Path
import torch
from torch import Tensor
import pickle

_A = TypeVar("_A", np.ndarray, Tensor)

def _canonicalize_array(arr: _A) -> np.ndarray:
    """将数组规范化为NumPy数组格式"""
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


class NumpySnapshot:
    """用于NumPy数组的快照测试工具，使用.npz格式存储。"""
    
    def __init__(
        self, 
        snapshot_dir: str = "tests/_snapshots",
    ):
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
    
    def _get_snapshot_path(self, test_name: str) -> Path:
        """获取快照文件的路径。"""
        return self.snapshot_dir / f"{test_name}.npz"
    
    def assert_match(
        self, 
        actual: _A | dict[str, _A], 
        test_name: str, 
        force_update: bool = False,
        rtol: float = 1e-4, 
        atol: float = 1e-2,
    ):
        """
        断言实际数组与快照匹配。
        
        参数:
            actual: 单个NumPy数组或命名字典的数组
            test_name: 测试名称（用于快照文件）
            force_update: 如果为True，则更新快照而不是比较
        """
        snapshot_path = self._get_snapshot_path(test_name)
        
        # 将单个数组转换为字典以便一致处理
        arrays_dict = actual if isinstance(actual, dict) else {"array": actual}
        arrays_dict = {
            k: _canonicalize_array(v)
            for k, v in arrays_dict.items()
        }
        
        # 加载快照
        expected_arrays = dict(np.load(snapshot_path))
        
        # 验证所有预期数组都存在
        missing_keys = set(arrays_dict.keys()) - set(expected_arrays.keys())
        if missing_keys:
            raise AssertionError(f"快照中未找到键 {missing_keys}，测试名称：{test_name}")
        
        # 验证所有实际数组都是预期的
        extra_keys = set(expected_arrays.keys()) - set(arrays_dict.keys())
        if extra_keys:
            raise AssertionError(f"快照包含额外键 {extra_keys}，测试名称：{test_name}")
        
        # 比较所有数组
        for key in arrays_dict:
            np.testing.assert_allclose(
                _canonicalize_array(arrays_dict[key]),
                expected_arrays[key], 
                rtol=rtol, 
                atol=atol,
                err_msg=f"数组 '{key}' 与快照不匹配，测试名称：{test_name}"
            )


class Snapshot:
    def __init__(self, snapshot_dir: str = "tests/_snapshots"):
        """
        用于任意数据类型的快照，保存为pickle文件。
        """
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
    
    def _get_snapshot_path(self, test_name: str) -> Path:
        return self.snapshot_dir / f"{test_name}.pkl"
    
    def assert_match(
        self,
        actual: _A | dict[str, _A],
        test_name: str,
        force_update: bool = False,
    ):
        """
        断言实际数据与快照匹配。
        参数:
            actual: 单个对象或命名字典的对象
            test_name: 测试名称（用于快照文件）
            force_update: 如果为True，则更新快照而不是比较
        """
    
        snapshot_path = self._get_snapshot_path(test_name)


        # 加载快照
        with open(snapshot_path, "rb") as f:
            expected_data = pickle.load(f)
        
        if isinstance(actual, dict):
            for key in actual: 
                if key not in expected_data:
                    raise AssertionError(f"快照中未找到键 '{key}'，测试名称：{test_name}")
                assert actual[key] == expected_data[key], f"键 '{key}' 的数据与快照不匹配，测试名称：{test_name}"
        else:
            assert actual == expected_data, f"数据与快照不匹配，测试名称：{test_name}"
        

@pytest.fixture
def snapshot(request):
    """
    提供快照测试功能的fixture。
    
    用法:
        def test_my_function(snapshot):
            result = my_function()
            snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    # 使用默认设置创建快照处理器
    snapshot_handler = Snapshot()
    
    # 修补assert_match方法以默认包含更新标志
    original_assert_match = snapshot_handler.assert_match
    
    def patched_assert_match(actual, test_name=None, force_update=force_update):
        # 如果未提供测试名称，使用测试函数名称
        if test_name is None:
            test_name = request.node.name
        return original_assert_match(actual, test_name=test_name, force_update=force_update)
    
    snapshot_handler.assert_match = patched_assert_match
    
    return snapshot_handler



# 可以在所有测试中使用的fixture
@pytest.fixture
def numpy_snapshot(request):
    """
    提供NumPy快照测试功能的fixture。
    
    用法:
        def test_my_function(numpy_snapshot):
            result = my_function()
            numpy_snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    match_exact = request.config.getoption("--snapshot-exact", default=False)
    
    # 使用默认设置创建快照处理器
    snapshot = NumpySnapshot()
    
    # 修补assert_match方法以默认包含更新标志
    original_assert_match = snapshot.assert_match
    
    def patched_assert_match(actual, test_name=None, force_update=force_update, rtol=1e-4, atol=1e-2):
        # 如果未提供测试名称，使用测试函数名称
        if test_name is None:
            test_name = request.node.name
        if match_exact:
            rtol = atol = 0
        return original_assert_match(actual, test_name=test_name, force_update=force_update, rtol=rtol, atol=atol)
    
    snapshot.assert_match = patched_assert_match
    
    return snapshot


@pytest.fixture
def ts_state_dict(request):
    from .common import FIXTURES_PATH
    import json
    state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    config = json.load(open(FIXTURES_PATH / "ts_tests" / "model_config.json"))
    state_dict = {
        k.replace('_orig_mod.', ''): v for k, v in state_dict.items()
    }
    return state_dict, config



# 用于模型fixture的模型参数

@pytest.fixture
def n_layers():
    return 3


@pytest.fixture
def vocab_size():
    return 10_000


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def n_queries():
    return 12


@pytest.fixture
def n_keys():
    return 16


@pytest.fixture
def n_heads():
    return 4

@pytest.fixture
def d_head():
    return 16

@pytest.fixture
def d_model(n_heads, d_head):
    return n_heads * d_head

@pytest.fixture
def d_ff():
    return 128

@pytest.fixture
def q(batch_size, n_queries, d_model):
    torch.manual_seed(1)
    return torch.randn(batch_size, n_queries, d_model)

@pytest.fixture
def k(batch_size, n_keys, d_model):
    torch.manual_seed(2)
    return torch.randn(batch_size, n_keys, d_model)

@pytest.fixture
def v(batch_size, n_keys, d_model):
    torch.manual_seed(3)
    return torch.randn(batch_size, n_keys, d_model)

@pytest.fixture
def in_embeddings(batch_size, n_queries, d_model):
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)

@pytest.fixture
def mask(batch_size, n_queries, n_keys):
    torch.manual_seed(5)
    return torch.randn(batch_size, n_queries, n_keys) > 0.5

@pytest.fixture
def in_indices(batch_size, n_queries):
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (batch_size, n_queries))

@pytest.fixture
def theta():
    return 10000.0

@pytest.fixture
def pos_ids(n_queries):
    return torch.arange(0, n_queries)


# # 使用示例:
# def test_single_array(numpy_snapshot):
#     # 产生NumPy数组的示例函数
#     def my_function():
#         return np.array([[1.0, 2.0], [3.0, 4.0001]])
    
#     result = my_function()
    
#     # 只需提供结果 - 测试名称将被推断
#     numpy_snapshot.assert_match(result)


# def test_multiple_arrays(numpy_snapshot):
#     # 产生多个数组的函数
#     def my_function():
#         return {
#             "weights": np.array([0.1, 0.2, 0.3]),
#             "biases": np.array([0.01, 0.02]),
#             "gradients": np.array([[0.001, 0.002], [0.003, 0.004]])
#         }
    
#     results = my_function()
    
#     # 使用显式名称和自定义容差进行测试
#     # custom_snapshot = NumpySnapshot()
#     numpy_snapshot.assert_match(
#         results, 
#         "my_special_test",
#         rtol=1e-4,
#         atol=1e-6,
#     )

# def test_state_dict(ts_state_dict):
#     print(ts_state_dict)