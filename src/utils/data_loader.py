from __future__ import annotations
from typing import List
import os
import math

class DataLoader:
    __slots__ = ("path", "n", "degree", "adj")

    def __init__(self, path: str):
        self.path: str = path
        self.n: int = 0
        self.degree: List[int] = []
        self.adj: List[List[int]] = []
        self._load()

    def _load(self) -> None:
        path = self.path

        # —— 流式读取（readline），避免与 tell() 冲突 —— #
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
            if not first:
                raise ValueError("图文件为空或缺少顶点数 n。")
            n = int(first.strip())
            self.n = n

            # 初始化
            degree: List[int] = [0] * n
            adj: List[List[int]] = [[] for _ in range(n)]

            # —— 进度统计（基于字节，单次扫描，不预读）—— #
            # 使用 fstat 更快更直接
            total_size = os.fstat(f.fileno()).st_size
            start_pos = f.tell()  # 跳过首行后的文件位置
            denom = max(total_size - start_pos, 1)  # 防止除零
            next_tick = 5  # 下一次需要打印的百分比阈值
            # print("加载进度：0%", flush=True)

            # —— 单次扫描：就地写入邻接表 + 统计度 —— #
            # 为了速度做局部绑定
            _adj = adj
            _degree = degree
            append_u = None  # 占位，减少属性创建（无实际使用，仅保持风格）
            CHECK_INTERVAL_MASK = 0x1FFF  # 每 8192 行检查一次以降低 tell() 开销
            line_count = 0

            # 用 while+readline，避免 for line in f 与 tell 冲突
            readline = f.readline
            tell = f.tell

            while True:
                line = readline()
                if not line:
                    break
                line_count += 1

                # 快速跳过空白行
                # 绝大多数图文件行都是 "u v\n"，先检查首字符，避免不必要 split
                if line == "\n":
                    # 周期性检查进度
                    if (line_count & CHECK_INTERVAL_MASK) == 0 and next_tick < 100:
                        cur = tell()
                        progress = int((cur - start_pos) * 100 / denom)
                        while progress >= next_tick and next_tick < 100:
                            #print(f"加载进度：{next_tick}%", flush=True)
                            next_tick += 5
                    continue

                parts = line.split()
                if not parts:
                    if (line_count & CHECK_INTERVAL_MASK) == 0 and next_tick < 100:
                        cur = tell()
                        progress = int((cur - start_pos) * 100 / denom)
                        while progress >= next_tick and next_tick < 100:
                            #print(f"加载进度：{next_tick}%", flush=True)
                            next_tick += 5
                    continue

                # 解析边（假设 0 <= u, v < n）
                u = int(parts[0]); v = int(parts[1])

                # 跳过自环
                if u == v:
                    if (line_count & CHECK_INTERVAL_MASK) == 0 and next_tick < 100:
                        cur = tell()
                        progress = int((cur - start_pos) * 100 / denom)
                        while progress >= next_tick and next_tick < 100:
                            #print(f"加载进度：{next_tick}%", flush=True)
                            next_tick += 5
                    continue

                _adj[u].append(v)
                _adj[v].append(u)
                _degree[u] += 1
                _degree[v] += 1

                # 周期性检查进度（避免每行 tell）
                if (line_count & CHECK_INTERVAL_MASK) == 0 and next_tick < 100:
                    cur = tell()
                    progress = int((cur - start_pos) * 100 / denom)
                    while progress >= next_tick and next_tick < 100:
                        #print(f"加载进度：{next_tick}%", flush=True)
                        next_tick += 5

        # 完成后补齐剩余的 5% 刻度并打印 100%
        while next_tick < 100:
            #print(f"加载进度：{next_tick}%", flush=True)
            next_tick += 5
        print("加载进度：100%", flush=True)

        # —— 局部排序：每个节点的邻居按 (deg[v], v) 升序 —— #
        # 提速点：
        # 1) 预计算单调可比较的 rank_key[v]，避免在排序时反复创建 (deg[v], v) 元组
        # 2) 使用 rank_key.__getitem__ 作为 key，避免 lambda 带来的开销
        deg = degree
        # 计算能容纳 v 的位宽（ceil(log2(n)))，与度拼成一个整数，保持词典序
        # 注意：若 n==1，shift 设为 1 以避免位宽为 0
        shift = max(1, (n - 1).bit_length())
        rank_key = [(d << shift) | v for v, d in enumerate(deg)]
        rk_get = rank_key.__getitem__

        for u in range(n):
            # 就地排序：按 rank_key[v] 升序
            adj[u].sort(key=rk_get)

        # 写回属性
        self.degree = degree
        self.adj = adj

    def get_graph(self):
        """返回 n, degree, adj（邻居已按度升序、同度按编号升序）"""
        return self.n, self.degree, self.adj
