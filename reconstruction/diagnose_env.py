"""
环境与项目自检脚本：结合 Cursor workspaceStorage 中的 workspace.json，
输出 Python / pip / conda、网络连通性、以及 Mito3D_Reconstruction_Thesis 所需依赖与数据路径。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlparse


def check_python_info():
    """检查 Python 版本、架构（32/64 位）"""
    print("===== 1. Python 基础信息 =====")
    print(f"Python 版本: {sys.version}")
    print(f"Python 架构: {'64位' if sys.maxsize > 2**32 else '32位'}")
    print(f"Python 可执行文件路径: {sys.executable}")
    print(f"系统架构: {platform.architecture()[0]}")
    print(f"操作系统: {platform.system()} {platform.release()}")


def check_pip_info():
    """检查 pip 路径、版本、配置"""
    print("\n===== 2. Pip 信息 =====")
    try:
        which_cmd = ["where", "pip"] if platform.system() == "Windows" else ["which", "pip"]
        pip_path = subprocess.check_output(which_cmd, text=True).strip()
        print(f"Pip 可执行文件路径:\n{pip_path}")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"无法定位 pip: {e}")

    try:
        pip_version = subprocess.check_output(
            [sys.executable, "-m", "pip", "--version"], text=True
        ).strip()
        print(f"Pip 版本: {pip_version}")
    except subprocess.CalledProcessError as e:
        print(f"Pip 版本: 读取失败 ({e})")

    try:
        pip_config = subprocess.check_output(
            [sys.executable, "-m", "pip", "config", "list"], text=True
        ).strip()
        print(f"Pip 配置: {pip_config if pip_config else '无自定义配置'}")
    except subprocess.CalledProcessError:
        print("Pip 配置: 无（或读取失败）")


def check_conda_env():
    """检查 Conda 虚拟环境信息"""
    print("\n===== 3. Conda 环境信息 =====")
    if "CONDA_DEFAULT_ENV" in os.environ:
        print(f"当前激活的 Conda 环境: {os.environ['CONDA_DEFAULT_ENV']}")
        print(f"Conda 环境路径: {os.environ.get('CONDA_PREFIX', '未知')}")
    else:
        print("未检测到已激活的 Conda 环境（CONDA_DEFAULT_ENV 未设置）。")


def file_uri_to_local_path(folder_uri: str) -> Optional[str]:
    """将 Cursor workspace.json 中的 file:/// URI 转为本地路径。"""
    if not folder_uri or not isinstance(folder_uri, str):
        return None
    parsed = urlparse(folder_uri)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path or "")
    # Windows: file:///d%3A/foo -> path 常为 /d:/foo
    if path.startswith("/") and len(path) >= 3 and path[2] == ":":
        path = path[1:]
    return os.path.normpath(path) if path else None


def _read_workspace_json(workspace_json: Path) -> Optional[Tuple[str, str]]:
    """读取单个 workspace.json，返回 (父目录名或 workspace 文件夹名, 本地 folder 路径)。"""
    try:
        data = json.loads(workspace_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    folder = data.get("folder")
    local = file_uri_to_local_path(folder) if folder else None
    if not local:
        return None
    wid = workspace_json.parent.name
    return (wid, local)


def find_cursor_workspace_roots(storage_root: Path) -> List[Tuple[str, str]]:
    """
    扫描 Cursor workspaceStorage 下各子目录的 workspace.json，返回 (子目录名, 本地 folder 路径)。
    也支持直接传入某个 UUID 子目录（该目录下即有 workspace.json）。
    """
    roots: List[Tuple[str, str]] = []
    if not storage_root.is_dir():
        return roots
    direct = storage_root / "workspace.json"
    if direct.is_file():
        pair = _read_workspace_json(direct)
        if pair:
            roots.append(pair)
        return roots
    for child in storage_root.iterdir():
        if not child.is_dir():
            continue
        wj = child / "workspace.json"
        if not wj.is_file():
            continue
        pair = _read_workspace_json(wj)
        if pair:
            roots.append(pair)
    return roots


def check_cursor_workspace_binding(storage_root: Optional[Path], cwd: Path):
    """结合 Cursor workspaceStorage：列出与本机工程可能相关的 workspace 根目录。"""
    print("\n===== 4. Cursor 工作区绑定（workspace.json）=====")
    default_roots = []
    env_storage = os.environ.get("CURSOR_WORKSPACE_STORAGE")
    candidates: List[Path] = []
    if storage_root is not None:
        candidates.append(storage_root)
    if env_storage:
        candidates.append(Path(env_storage))
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "Cursor" / "User" / "workspaceStorage")

    seen = set()
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen:
            continue
        seen.add(key)
        if not c.is_dir():
            print(f"路径不存在或不是目录，跳过: {c}")
            continue
        pairs = find_cursor_workspace_roots(c)
        if not pairs:
            print(f"在 {c} 下未发现有效的 workspace.json")
            continue
        print(f"扫描目录: {c}")
        cwd_norm = str(cwd.resolve())
        for wid, local in pairs:
            match = "  <-- 与当前工作目录一致" if os.path.normpath(local) == cwd_norm else ""
            if "Mito3D" in local or os.path.normpath(local) == cwd_norm:
                default_roots.append((wid, local))
            print(f"  [{wid}] -> {local}{match}")
        if not default_roots:
            print("  （未标记与 Mito3D 或当前目录匹配的项；仍可使用上面列表手动核对）")
    return default_roots


def check_pytorch_index_access():
    """检查能否访问 PyTorch cu128 nightly 索引（可选）"""
    print("\n===== 5. PyTorch 索引访问测试（cu128 nightly）=====")
    test_url = "https://download.pytorch.org/whl/nightly/cu128/torch/"
    try:
        import requests

        response = requests.head(test_url, timeout=10, allow_redirects=True)
        print(f"索引地址: {test_url}")
        print(
            f"HTTP 状态码: {response.status_code} "
            f"(2xx=可访问；其余可能是网络或地址变更)"
        )
    except ImportError:
        print("未安装 requests，跳过（可执行: pip install requests）")
    except Exception as e:  # noqa: BLE001
        print(f"索引访问失败: {e}（可能是网络/代理问题）")


def check_package_compatibility():
    """检查当前解释器是否满足常见 PyTorch 安装前提（Python 位数与版本范围）"""
    print("\n===== 6. 解释器基础兼容性 =====")
    py_version = tuple(map(int, sys.version.split()[0].split(".")))
    is_py_compatible = (3, 9) <= py_version <= (3, 12)
    print(f"Python 版本兼容 (3.9-3.12，便于搭配多数 PyTorch 轮子): {is_py_compatible}")
    is_64bit = sys.maxsize > 2**32
    print(f"64 位解释器: {is_64bit}")
    ok = is_py_compatible and is_64bit
    print(f"\n✅ 解释器基础项通过: {ok}")
    if not is_py_compatible:
        print("❌ Python 版本不在 3.9-3.12 范围内时，部分官方轮子可能不可用")
    if not is_64bit:
        print("❌ 非 64 位解释器通常无法使用官方 CUDA 版 PyTorch")


def check_mito3d_project(project_root: Path):
    """本项目关键依赖与数据路径。"""
    print("\n===== 7. Mito3D 项目：依赖与数据 =====")
    print(f"项目根目录（假定）: {project_root.resolve()}")

    def _try_import(name: str, attr: Optional[str] = None):
        try:
            mod = __import__(name, fromlist=["_"])
            if attr:
                return True, getattr(mod, attr, None)
            return True, mod
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    packages = [
        ("torch", "__version__"),
        ("numpy", "__version__"),
        ("zarr", "__version__"),
        ("skimage", None),
        ("trimesh", None),
        ("matplotlib", None),
    ]
    for name, attr in packages:
        ok, info = _try_import(name, attr)
        if ok and attr:
            print(f"  {name}: {info}")
        elif ok:
            print(f"  {name}: 已安装")
        else:
            print(f"  {name}: 缺失或导入失败 -> {info}")

    # torch + numpy 组合：老版本 torch 与 numpy 2.x 易导致 tensor.numpy() 异常
    try:
        import numpy as np
        import torch

        nv = tuple(int(x) for x in np.__version__.split(".")[:2])
        tv = getattr(torch, "__version__", "")
        if nv >= (2, 0) and tv and not tv.startswith("2.1") and "dev" not in tv:
            print(
                "  ⚠️ 提示: numpy≥2 与部分 torch 2.0.x 组合可能导致 .numpy() 报错；"
                "本项目 post_process 需要 tensor->numpy，建议 numpy 1.26.x"
            )
    except Exception:  # noqa: BLE001
        pass

    try:
        import torch

        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  当前 GPU: {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            arch_list = torch.cuda.get_arch_list()
            print(f"  device_capability: sm_{cap[0]}{cap[1]}")
            print(f"  torch.cuda.get_arch_list(): {arch_list}")
    except Exception as e:  # noqa: BLE001
        print(f"  CUDA 信息读取失败: {e}")

    ok_rt, msg = _try_import("rtx50_compat", None)
    print(f"  rtx50_compat（train.py 使用）: {'可用' if ok_rt else msg}")

    zarr_path = project_root / "data" / "sample_hela2" / "sample.zarr"
    if zarr_path.is_dir():
        print(f"  数据: 找到 {zarr_path}（可运行 train.py）")
    else:
        print(
            f"  数据: 未找到 {zarr_path}，可先运行: python prepare_data.py"
        )

    ckpt = project_root / "checkpoints" / "model_final.pth"
    if ckpt.is_file():
        print(f"  权重: 找到 {ckpt}（可运行 generate.py）")
    else:
        print(f"  权重: 未找到 {ckpt}，需先训练生成 checkpoints")


def main():
    parser = argparse.ArgumentParser(description="Mito3D / Cursor 环境诊断")
    parser.add_argument(
        "--workspace-storage",
        type=Path,
        default=None,
        help="Cursor workspaceStorage 根目录 "
        "(默认: %%APPDATA%%\\Cursor\\User\\workspaceStorage；也可用环境变量 CURSOR_WORKSPACE_STORAGE)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="项目根目录（默认: 本脚本所在目录）",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve() if args.project_root else Path(__file__).resolve().parent
    cwd = Path.cwd()

    check_python_info()
    check_pip_info()
    check_conda_env()
    check_cursor_workspace_binding(args.workspace_storage, cwd)
    check_pytorch_index_access()
    check_package_compatibility()
    check_mito3d_project(project_root)

    print("\n===== 完成 =====")


if __name__ == "__main__":
    main()
