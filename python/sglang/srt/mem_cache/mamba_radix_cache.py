from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
"""

import heapq
import queue
import threading
from collections import OrderedDict, defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
)
from sglang.srt.mem_cache.rsvd_eigh import randomized_svd_eigh
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

import logging

logger = logging.getLogger(__name__)


class TreeNode:

    counter = 0
    last_access_time_counter_float = float64(1.0)

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        self.mamba_host_value: Optional[torch.Tensor] = None
        self.mamba_compressed: bool = False
        self.compressed_slot: Optional[int] = None
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        self.last_access_time = get_last_access_time()

        self.hit_count = 0
        self.host_ref_counter = 0
        self.host_mamba_ref_counter = 0
        # store the host indices of KV cache
        self.host_value = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None
        self.host_mamba_prev = None
        self.host_mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def mamba_evicted(self):
        return self.mamba_value is None and not self.mamba_compressed

    @property
    def has_mamba_state(self) -> bool:
        return self.mamba_value is not None or self.mamba_compressed

    @property
    def backuped(self):
        return self.host_value is not None

    @property
    def mamba_backuped(self):
        return self.mamba_host_value is not None

    def protect_host(self):
        """Protect the host KV value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host KV value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def protect_host_mamba(self):
        """Protect the host mamba value from eviction."""
        self.host_mamba_ref_counter += 1

    def release_host_mamba(self):
        """Release the host mamba value, allowing it to be evicted."""
        if self.host_mamba_ref_counter > 0:
            self.host_mamba_ref_counter -= 1
        else:
            raise RuntimeError("Host mamba reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: "TreeNode") -> List[str]:
        if node is None or node.hash_value is None:
            return []
        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if self.mamba:
            self.prv = "mamba_prev"
            self.nxt = "mamba_next"
            self.lock_ref = "mamba_lock_ref"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}"
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            if not self.mamba or node.mamba_value is not None:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    def pretty_print(self, tree_cache: Optional["MambaRadixCache"] = None):
        """
        Pretty print the lru list
        """
        msg = f"{self.mamba=} LRU list: "
        x_lru = self._get_lru()
        while x_lru is not None and x_lru.id in self.cache:
            msg += f"[{x_lru.id}] {x_lru.last_access_time:f} -> "
            x_lru = getattr(x_lru, self.prv)
        print(msg)

        if not tree_cache:
            return
        msg = f"{self.mamba=} Nodes (sorted by last_access_time): "
        if self.mamba:
            nodes = tree_cache._collect_nontombstone_nodes()
        else:
            nodes = tree_cache._collect_all_nodes()
        heapq.heapify(nodes)
        while len(nodes):
            x = heapq.heappop(nodes)
            msg += f"[{x.id}] {x.last_access_time:f} -> "
        print(msg)

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += (
                len(node.value) if not self.mamba else len(node.mamba_value)
            )
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: "MambaRadixCache"):
        """
        Check if the lru list is valid by rebuilding the lru list from the tree, heapifying it, and
        checking if the lru list is valid.
        """
        try:
            if self.mamba:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru = len(self.cache)
            # heapify based on last_access_time
            heapq.heapify(nodes)
            # the root node is not in the lru list
            assert len(nodes) == (
                total_lru + (0 if self.mamba else 1)
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru()
            while len(nodes):
                x = heapq.heappop(nodes)
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x_lru is not None and x_lru.id in self.cache
                ), f"Incorrect LRU list, x_lru is None or not in cache: {x_lru=}, {x.id=}"

                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}, {x.last_access_time=}, {x_lru.last_access_time=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}"
                assert (
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.mamba:
                evictable_size = tree_cache.mamba_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            if get_tensor_model_parallel_rank() == 0:
                msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}"
                logger.error(msg)
                tree_cache.pretty_print()
                tree_cache.full_lru_list.pretty_print(tree_cache)
                tree_cache.mamba_lru_list.pretty_print(tree_cache)
                raise Exception(msg)


class MambaRadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        assert isinstance(
            params.token_to_kv_pool_allocator, TokenToKVPoolAllocator
        ) or isinstance(params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator)
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator

        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer

        if not self.enable_mamba_extra_buffer:
            assert (
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}"

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        # SVD compression config
        server_args = get_global_server_args()
        self.enable_svd_compression = getattr(
            server_args, "mamba_svd_compression", False
        )
        self.svd_rank = getattr(server_args, "mamba_svd_rank", 16)
        self.svd_niter = getattr(server_args, "mamba_svd_niter", 1)
        self.svd_oversample = getattr(server_args, "mamba_svd_oversample", 4)
        # Greedy worker batch cap: how many pending snapshots to fold into one
        # batched SVD call. Larger amortizes kernel launches; bounded so a burst
        # of completions can't build an unboundedly large transient tensor.
        self.svd_worker_batch = getattr(server_args, "mamba_svd_worker_batch", 8)
        self._compression_thread: Optional[threading.Thread] = None
        self._compression_stop_event: Optional[threading.Event] = None
        if self.enable_svd_compression and self.req_to_token_pool.mamba_pool is not None:
            self._init_compression_state()

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)
        self.reset()

    ##### Public API #####

    def supports_mamba(self) -> bool:
        return True

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = RadixKey([], None)
        self.root_node.value = []
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key and optional Mamba-specific parameters.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        mamba_value = params.mamba_value
        prev_prefix_len = params.prev_prefix_len

        if value is None:
            value = torch.tensor([x for x in key.token_ids], dtype=torch.int64)
        prefix_len, mamba_exist = self._insert_helper(
            self.root_node, key, value, mamba_value, params.chunked, prev_prefix_len
        )
        return InsertResult(prefix_len=prefix_len, mamba_exist=mamba_exist)

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        """Cache request when it finishes."""
        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_mamba_cache(req)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_committed_len
        ]

        if is_insert:
            cache_len = (
                req.mamba_last_track_seqlen
                if self.enable_mamba_extra_buffer
                else len(token_ids)
            )
            if cache_len is None:
                cache_len = 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free(kv_indices[cache_end_idx:])
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
            else:
                page_aligned_len = len(kv_indices)
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

            assert (
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_committed_len=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this"

            # Radix Cache takes one ref in memory pool
            # insert the token_ids and kv_indices into the radix tree
            if self.enable_mamba_extra_buffer:
                mamba_ping_pong_track_buffer_to_keep = (
                    self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
                mamba_value = (
                    req.mamba_ping_pong_track_buffer[
                        mamba_ping_pong_track_buffer_to_keep
                    ]
                    .unsqueeze(-1)
                    .clone()
                )
            else:
                mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
                mamba_ping_pong_track_buffer_to_keep = None

            result = self.insert(
                InsertParams(
                    key=RadixKey(token_ids[:page_aligned_len], req.extra_key),
                    value=page_aligned_kv_indices,
                    mamba_value=mamba_value,
                    prev_prefix_len=req.cache_protected_len,
                )
            )
            mamba_exist = result.mamba_exist
        else:
            self.token_to_kv_pool_allocator.free(kv_indices[req.cache_protected_len :])
            mamba_exist = True

        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None

        free_mamba_cache = True if self.enable_mamba_extra_buffer else mamba_exist

        if free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )

        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""

        def _skip_cache_unfinished_req(req: Req) -> None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        token_ids = req.fill_ids
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None:
            return _skip_cache_unfinished_req(req)

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        # kv_indices is the kv indices to be cached
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        assert page_aligned_len == len(
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {FLA_CHUNK_SIZE=}"

        page_aligned_token_ids = token_ids[:page_aligned_len]

        if self.enable_mamba_extra_buffer:
            # copy from the ping pong track buffer
            mamba_ping_pong_track_buffer_to_keep = (
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
            mamba_value = (
                req.mamba_ping_pong_track_buffer[mamba_ping_pong_track_buffer_to_keep]
                .unsqueeze(-1)
                .clone()
            )
        else:
            mamba_value = self.req_to_token_pool.get_mamba_indices(
                req.req_pool_idx
            ).unsqueeze(-1)
        # radix tree mamba value is forked from req space
        mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(mamba_value)

        # if alloc mamba cache failed, do evict and alloc again
        if mamba_value_forked is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            mamba_value_forked = self.req_to_token_pool.mamba_pool.fork_from(
                mamba_value
            )
            assert mamba_value_forked is not None, "Can not alloc mamba cache"
        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_forked,
                prev_prefix_len=req.cache_protected_len,
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
        # there is a mamba cache in radix cache, release it
        if mamba_exist:
            self.req_to_token_pool.mamba_pool.free(mamba_value_forked)

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_forked)

        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # NOTE: this is needed for both page_size == 1 and page_size > 1
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_mamba_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #mamba_num: {total_mamba_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> Tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

        assert (
            x.mamba_value is not None or x.mamba_compressed
        ), f"leaf node mamba state missing, {x.id=}"
        was_compressed = x.mamba_compressed
        # 1. a leaf node, free full tokens
        self.token_to_kv_pool_allocator.free(x.value)
        full_num_evicted = len(x.value)
        # Compressed leaves do not recover full-pool mamba slots (already freed
        # at drain time), so they report 0 here.
        mamba_num_evicted = 0 if was_compressed else len(x.mamba_value)

        # 2. get the next node, update the lru lists BEFORE clearing mamba state
        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        # compressed nodes were removed from mamba_lru_list at compression commit
        if not was_compressed:
            self.mamba_lru_list.remove_node(x)

        # 3. delete the leaf node (accounting uses mamba_value for non-compressed path)
        self._delete_leaf(x)

        # 4. now release the actual underlying mamba state (full slot or compressed slot)
        self._free_mamba_state(x)

        # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_evicted = 0
        mamba_num_evicted = 0

        if params.num_tokens > 0:
            full_num_evicted = self.evict_full(params.num_tokens)
        if params.mamba_num > 0:
            mamba_num_evicted = self.evict_mamba(params.mamba_num)

        return EvictResult(
            num_tokens_evicted=full_num_evicted, mamba_num_evicted=mamba_num_evicted
        )

    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states. Returns the number of mamba states evicted."""
        if self.disable or mamba_num <= 0:
            return 0
        # get the least recently used node that is not locked, doesn't have to be a leaf
        x = self.mamba_lru_list.get_lru_no_lock()
        mamba_num_evicted = 0
        # evict lru leaf nodes until mamba_num_tokens is reached
        while mamba_num_evicted < mamba_num and (self.mamba_lru_list.in_list(x)):
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            if len(x.children) > 0:
                # 1. an internal node, free mamba tokens.
                self.req_to_token_pool.mamba_pool.free(x.mamba_value)
                mamba_num_evicted += len(x.mamba_value)

                # 2. get the next node, update the lru lists
                x_next = self.mamba_lru_list.get_prev_no_lock(x)
                self.mamba_lru_list.remove_node(x)

                # 3. tombstone the node
                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, x_next = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

            x = x_next

        return mamba_num_evicted

    def evict_full(self, full_num_tokens: int) -> int:
        """Evict full KV cache. Returns the number of tokens evicted."""
        if self.disable or full_num_tokens <= 0:
            return 0

        full_num_evicted = 0
        # get the least recently used leaf node that is not locked
        x = self.full_lru_list.get_leaf_lru_no_lock()

        while full_num_evicted < full_num_tokens and self.full_lru_list.in_list(x):
            assert (
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}"
            full_num_evicted_delta, _, x, x_next = self._evict_leaf_node(x, False)
            full_num_evicted += full_num_evicted_delta

            # if parent has no more children, it is a leaf. It is possible that this node is lru, so
            # we need to get the first leaf node in the lru list
            if len(x.parent.children) == 0:
                x_next = self.full_lru_list.get_leaf_lru_no_lock()

            x = x_next

        return full_num_evicted

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return IncLockRefResult()

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1
        elif node.mamba_compressed:
            # Compressed nodes don't contribute to mamba_evictable_size_ /
            # mamba_protected_size_ (those count the full mamba pool only),
            # but we still track mamba_lock_ref so _evict_compressed_lru skips
            # in-use entries.
            node.mamba_lock_ref += 1

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1
            node = node.parent
        return IncLockRefResult()

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return DecLockRefResult()

        if node.mamba_value is not None:
            assert (
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}"
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1
        elif node.mamba_compressed:
            # Mirror of inc_lock_ref's compressed branch. Counter adjustments
            # already happened at drain commit (mamba_protected_size_ -= 1 for
            # locked-at-commit, mamba_evictable_size_ -= 1 for unlocked); here
            # we just decrement the lock ref. Tolerate mamba_lock_ref == 0 to
            # match the existing code style for skipped branches.
            if node.mamba_lock_ref > 0:
                node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1
            node = node.parent

        return DecLockRefResult()

    def sanity_check(self):
        if self.disable:
            return
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def all_mamba_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            if node.mamba_value is not None:
                values.append(node.mamba_value)
            for _, child in node.children.items():
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = self.full_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _init_compression_state(self) -> None:
        """Allocate compressed pool + start background SVD worker."""
        pool = self.req_to_token_pool.mamba_pool
        temporal_shape = pool.mamba_cache.temporal.shape  # [L, N, H, D, S]
        L, _, H, D, S = temporal_shape
        temporal_dtype = pool.mamba_cache.temporal.dtype
        r = self.svd_rank
        packed_per_head = r * (D + 1 + S)
        full_per_head = D * S
        assert packed_per_head <= full_per_head, (
            f"SVD rank {r} too large: packed U,S,V ({packed_per_head} elements) "
            f"exceeds full-rank per-head size ({full_per_head}) for "
            f"head_dim={D}, state_size={S}. Max rank: {full_per_head // (D + 1 + S)}"
        )

        num_compressed_slots = max(1, pool.size // 2)
        self.compressed_temporal = torch.empty(
            (num_compressed_slots, L, H, packed_per_head),
            dtype=temporal_dtype,
            device=self.device,
        )
        conv_tensors = pool.mamba_cache.conv
        self.compressed_conv = [
            torch.empty(
                (num_compressed_slots, L, *c.shape[2:]),
                dtype=c.dtype,
                device=self.device,
            )
            for c in conv_tensors
        ]
        self._compressed_free_slots: List[int] = list(range(num_compressed_slots))
        self._compressed_lru: "OrderedDict[int, TreeNode]" = OrderedDict()
        self._pending_compression: Dict[int, TreeNode] = {}
        self._compression_queue: queue.Queue = queue.Queue()
        self._compression_done_queue: queue.Queue = queue.Queue()
        self._compression_stop_event = threading.Event()
        self._max_inflight_compression = 8
        # Dedicated CUDA stream so SVD kernels overlap with inference instead
        # of serializing on the default stream.
        self._svd_device = pool.mamba_cache.temporal.device
        if self._svd_device.type == "cuda":
            self._svd_stream = torch.cuda.Stream(device=self._svd_device)
        else:
            self._svd_stream = None
        # # Diagnostic counters — cheap, always on. Inspect via tree._compression_stats().
        # self._svd_enqueued = 0
        # self._svd_enqueue_skipped_inflight = 0
        # self._svd_enqueue_skipped_no_slot = 0
        # self._svd_worker_processed = 0
        # self._svd_worker_failed = 0
        # self._svd_committed = 0
        # self._svd_dropped_evicted = 0
        # self._svd_dropped_locked = 0
        # self._svd_dropped_pool_full = 0
        # self._svd_decompress_hits = 0

        compressed_bytes = (
            self.compressed_temporal.element_size() * self.compressed_temporal.numel()
            + sum(c.element_size() * c.numel() for c in self.compressed_conv)
        )
        logger.info(
            "Async SVD compression enabled: rank=%d, head_dim=%d, state_size=%d, "
            "packed_per_head=%d, compressed_slots=%d, compressed_bytes=%.2fMB",
            r,
            D,
            S,
            packed_per_head,
            num_compressed_slots,
            compressed_bytes / (1024 * 1024),
        )

        self._compression_thread = threading.Thread(
            target=self._compression_worker,
            name="mamba-svd-worker",
            daemon=True,
        )
        self._compression_thread.start()
        logger.info(
            "mamba-svd-worker thread started (alive=%s, daemon=%s)",
            self._compression_thread.is_alive(),
            self._compression_thread.daemon,
        )

    # def _compression_stats(self) -> dict:
    #     """Diagnostic snapshot of compression pipeline state."""
    #     if not self.enable_svd_compression:
    #         return {"enable_svd_compression": False}
    #     return {
    #         "enable_svd_compression": True,
    #         "worker_alive": bool(
    #             self._compression_thread and self._compression_thread.is_alive()
    #         ),
    #         "work_queue_depth": self._compression_queue.qsize(),
    #         "done_queue_depth": self._compression_done_queue.qsize(),
    #         "pending": len(self._pending_compression),
    #         "compressed_lru_size": len(self._compressed_lru),
    #         "compressed_free_slots": len(self._compressed_free_slots),
    #         "enqueued_total": self._svd_enqueued,
    #         "enqueue_skipped_inflight": self._svd_enqueue_skipped_inflight,
    #         "enqueue_skipped_no_slot": self._svd_enqueue_skipped_no_slot,
    #         "worker_processed_total": self._svd_worker_processed,
    #         "worker_failed_total": self._svd_worker_failed,
    #         "committed_total": self._svd_committed,
    #         "dropped_evicted": self._svd_dropped_evicted,
    #         "dropped_locked": self._svd_dropped_locked,
    #         "dropped_pool_full": self._svd_dropped_pool_full,
    #         "decompress_hits": self._svd_decompress_hits,
    #     }

    def _enqueue_compression(self, node: TreeNode) -> None:
        """Snapshot the node's full-rank state and post to the SVD worker.

        The snapshot clone is issued on the *SVD stream*, not the default
        (inference) stream, so the ~50-100MB copy does not contend with decode.
        An event recorded on the current stream makes the SVD stream wait for
        the prefill write that produced this slot before it reads. The host
        cost of this method is just an event record plus a kernel launch on the
        side stream, so the scheduler thread is not stalled by the copy."""
        if not self.enable_svd_compression:
            return
        if node.mamba_value is None or node.mamba_compressed:
            return
        if len(self._pending_compression) >= self._max_inflight_compression:
            return
        if not self._compressed_free_slots and not self._compressed_lru:
            return
        if node.id in self._pending_compression:
            return

        pool = self.req_to_token_pool.mamba_pool
        # pool.mamba_cache.temporal shape is [L, N, H, D, S]; mamba_value is a [1] slot tensor
        try:
            src = pool.mamba_cache.temporal[:, node.mamba_value].squeeze(1).detach()
            if self._svd_stream is not None:
                event = torch.cuda.Event()
                event.record()  # default stream: orders after the prefill write
                with torch.cuda.stream(self._svd_stream):
                    self._svd_stream.wait_event(event)
                    gpu_snapshot = src.clone()  # clone runs on the SVD stream
            else:
                gpu_snapshot = src.clone()
        except Exception as e:
            logger.warning("Failed to snapshot mamba state for node %s: %s", node.id, e)
            return

        self._pending_compression[node.id] = node
        self._compression_queue.put((node.id, gpu_snapshot))

    def _compression_worker(self) -> None:
        """Background daemon thread: pops GPU snapshots, runs a single *batched*
        randomized SVD on a side CUDA stream over all snapshots immediately
        available (up to svd_worker_batch), blocks the worker (not the
        scheduler) until the GPU work completes, then publishes packed tensors.

        Batching matters: every snapshot is the same shape [L, H, D, S], so
        stacking K of them into [K, L, H, D, S] lets one rsvd_eigh call cover
        the whole group. That amortizes kernel-launch overhead and keeps the
        SVD kernels large enough to run efficiently, which shortens the total
        time the side stream competes with inference for SMs."""
        stop_event = self._compression_stop_event
        if self._svd_device.type == "cuda":
            torch.cuda.set_device(self._svd_device)
        while not stop_event.is_set():
            try:
                first = self._compression_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                continue
            if first is None:
                continue
            # Greedily gather any other snapshots already waiting, so a backlog
            # collapses into one batched SVD instead of N serial ones.
            batch = [first]
            while len(batch) < self.svd_worker_batch:
                try:
                    nxt = self._compression_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is not None:
                    batch.append(nxt)
            try:
                self._process_compression_batch(batch)
            except Exception as e:
                node_ids = [b[0] for b in batch]
                logger.warning("Async SVD batch failed for nodes %s: %s", node_ids, e)

    def _process_compression_batch(self, batch: List[Tuple[int, torch.Tensor]]) -> None:
        """Run one batched SVD over `batch` snapshots and post packed results."""
        rank = self.svd_rank
        node_ids = [b[0] for b in batch]
        snapshots = [b[1] for b in batch]
        K = len(snapshots)
        L, H, D, S = snapshots[0].shape
        effective_rank = min(rank, D, S)

        def _run():
            stacked = torch.stack(snapshots, dim=0)  # [K, L, H, D, S]
            u, s, vh = randomized_svd_eigh(
                stacked,
                rank=effective_rank,
                n_iter=self.svd_niter,
                oversample=self.svd_oversample,
            )
            u_flat = u.reshape(K, L, H, D * effective_rank)
            v_flat = vh.transpose(-2, -1).reshape(K, L, H, S * effective_rank)
            packed = torch.cat([u_flat, s, v_flat], dim=-1)  # [K, L, H, (D+1+S)*r]
            return packed

        if self._svd_stream is not None:
            with torch.cuda.stream(self._svd_stream):
                packed = _run()
            self._svd_stream.synchronize()
        else:
            packed = _run()

        for i, node_id in enumerate(node_ids):
            self._compression_done_queue.put((node_id, packed[i]))

    def drain_compression_completions(self, max_per_call: int = 8) -> None:
        """Main-thread drain: commit completed SVD packs into the compressed pool.

        Capped low (8) so a burst of completions does not fire many GPU->GPU
        commit copies on the inference (default) stream in a single scheduler
        step, which would spike decode latency. Remaining completions are
        committed on subsequent steps."""
        if not self.enable_svd_compression:
            return
        pool = self.req_to_token_pool.mamba_pool
        rank = self.svd_rank
        for _ in range(max_per_call):
            try:
                node_id, packed = self._compression_done_queue.get_nowait()
            except queue.Empty:
                return
            node = self._pending_compression.pop(node_id, None)
            if node is None:
                continue
            if node.mamba_value is None or node.mamba_compressed:
                continue  # evicted or already committed — discard
            # NOTE: committing while locked (mamba_lock_ref > 0) is safe. The
            # tree's node.mamba_value is a fork separate from req.mamba_pool_idx,
            # so freeing it does not disturb the running request's working slot.
            # The scheduler is single-threaded, so no concurrent read race.
            if not self._compressed_free_slots:
                if not self._evict_compressed_lru():
                    continue  # every compressed entry is locked — drop
            c_slot = self._compressed_free_slots.pop()
            full_slot = node.mamba_value
            # 1. Write compressed temporal + conv to the new compressed slot.
            # `packed` already lives on the same CUDA device as compressed_temporal
            # (produced by the GPU SVD worker); .to() is a dtype cast only.
            self.compressed_temporal[c_slot].copy_(
                packed.to(dtype=self.compressed_temporal.dtype)
            )
            for i, conv_t in enumerate(pool.mamba_cache.conv):
                self.compressed_conv[i][c_slot].copy_(
                    conv_t[:, full_slot].squeeze(1)
                )
            # 2. Option A: remove from mamba_lru_list BEFORE clearing mamba_value
            # (the list's assertions require mamba_value to still be non-None).
            if self.mamba_lru_list.in_list(node):
                self.mamba_lru_list.remove_node(node)
            # 3. Hand off the node's contribution from the full-pool counters.
            # A locked full-rank node lives in mamba_protected_size_; an
            # unlocked one in mamba_evictable_size_. Dispatch based on the
            # current lock state. After commit, neither counter tracks this
            # node (compressed pool has its own bookkeeping).
            if node.mamba_lock_ref > 0:
                self.mamba_protected_size_ -= len(full_slot)
            else:
                self.mamba_evictable_size_ -= len(full_slot)
            # 4. Free the full-rank slot.
            pool.free(full_slot)
            # 5. Atomic state swap — single-writer (scheduler) thread, no lock.
            node.mamba_value = None
            node.compressed_slot = c_slot
            node.mamba_compressed = True
            # 6. Insert into compressed LRU. Locked compressed nodes stay in
            # the list; _evict_compressed_lru skips them.
            self._compressed_lru[node.id] = node

    def _evict_compressed_lru(self) -> bool:
        """Reclaim one compressed slot. Internal victims become mamba tombstones;
        leaf victims are fully evicted (attention KV + radix leaf). Returns True on success."""
        if not self._compressed_lru:
            return False
        scanned: set = set()
        while self._compressed_lru:
            node_id, victim = next(iter(self._compressed_lru.items()))
            if node_id in scanned:
                return False  # every entry is locked — give up
            # Internal victims only need mamba_lock_ref==0; leaf victims also need
            # full_lock_ref==0 because we'll free their attention KV.
            is_leaf = len(victim.children) == 0
            if victim.mamba_lock_ref > 0 or (is_leaf and victim.full_lock_ref > 0):
                scanned.add(node_id)
                self._compressed_lru.move_to_end(node_id)
                continue

            del self._compressed_lru[node_id]
            self._compressed_free_slots.append(victim.compressed_slot)
            victim.compressed_slot = None
            victim.mamba_compressed = False
            # compressed entries do not contribute to mamba_evictable_size_
            # (that counter tracks the full mamba pool), so no decrement here.

            if is_leaf:
                # Free attention KV and unlink from radix tree.
                self.token_to_kv_pool_allocator.free(victim.value)
                self.full_evictable_size_ -= len(victim.key)
                self.full_lru_list.remove_node(victim)
                key = self.get_child_key_fn(victim.key)
                victim.parent.children.pop(key, None)
                # Propagate tombstone cleanup up the tree if ancestors are now orphaned.
                self._iteratively_delete_tombstone_leaf(victim)
            # For internal victims, the node becomes a mamba tombstone but keeps its
            # attention KV — matches the existing evict_mamba tombstone semantics.
            return True
        return False

    def _free_mamba_state(self, node: TreeNode) -> None:
        """Dispatcher: release a node's mamba state (compressed or full-rank)."""
        if node.mamba_compressed:
            if node.compressed_slot is not None:
                self._compressed_free_slots.append(node.compressed_slot)
                node.compressed_slot = None
            self._compressed_lru.pop(node.id, None)
            node.mamba_compressed = False
        elif node.mamba_value is not None:
            self.req_to_token_pool.mamba_pool.free(node.mamba_value)
            node.mamba_value = None

    def _decompress_from_pool(
        self, src_node: TreeNode, dst_index: torch.Tensor
    ) -> None:
        """Reconstruct full mamba state at dst_index from either compressed or full-rank source."""
        pool = self.req_to_token_pool.mamba_pool
        if not src_node.mamba_compressed:
            logger.info(f"Full-rank mamba state from {src_node.id} to {dst_index}")
            pool.copy_from(src_node.mamba_value, dst_index)
            return

        logger.info(f"Decompressing compressed mamba state from {src_node.id} to {dst_index}")
        rank = self.svd_rank
        c_slot = src_node.compressed_slot
        packed = self.compressed_temporal[c_slot]  # [L, H, (D+1+S)*r]
        L, H, _ = packed.shape
        D = pool.mamba_cache.temporal.shape[-2]
        S = pool.mamba_cache.temporal.shape[-1]
        r = rank
        u = packed[:, :, : D * r].reshape(L, H, D, r)
        s = packed[:, :, D * r : D * r + r]
        v = packed[:, :, D * r + r : D * r + r + S * r].reshape(L, H, S, r)
        full_state = (u * s.unsqueeze(-2)) @ v.transpose(-2, -1)
        pool.mamba_cache.temporal[:, dst_index] = full_state.unsqueeze(1).to(
            dtype=pool.mamba_cache.temporal.dtype
        )
        for i, conv_t in enumerate(pool.mamba_cache.conv):
            conv_t[:, dst_index] = self.compressed_conv[i][c_slot].unsqueeze(1).to(
                dtype=conv_t.dtype
            )

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = self.get_child_key_fn(key)

        value: List[torch.Tensor] = []
        best_value_len = 0
        best_last_node = node
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # update best_value_len and best_last_node if needed
            if node.has_mamba_state:
                best_value_len = len(value)
                best_last_node = node

            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if node.has_mamba_state:
            best_value_len = len(value)
            best_last_node = node

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key

        if self.disable or len(key) == 0:
            return None

        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        cow_mamba = params.cow_mamba
        req = params.req

        # update time for matched nodes, and make nodes closer to root to be least recently used
        # this allows mamba to evict nodes closer to root first
        node_update = last_node
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        self.mamba_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        # Bump compressed ancestors in the compressed LRU (they're excluded from
        # mamba_lru_list under Option A, so we walk separately).
        if self.enable_svd_compression and self._compressed_lru:
            ancestor = node_update
            while ancestor is not None and ancestor != self.root_node:
                if ancestor.mamba_compressed and ancestor.id in self._compressed_lru:
                    self._compressed_lru.move_to_end(ancestor.id)
                ancestor = ancestor.parent

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        # Calculate the branching point. It is defined as the last aligned position that
        # does not have a mamba value.
        if len(value) > best_value_len:
            mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
            mamba_cache_chunk_aligned_seqlen = (
                sum(len(v) for v in value) // mamba_cache_chunk_size
            ) * mamba_cache_chunk_size
            mamba_branching_seqlen = (
                mamba_cache_chunk_aligned_seqlen
                if mamba_cache_chunk_aligned_seqlen > 0
                else None
            )
        else:
            mamba_branching_seqlen = None

        # Copy mamba state to req local space if cow is true
        if cow_mamba and last_node.has_mamba_state:
            # for reqs without mamba cache
            if req.mamba_pool_idx is None:
                dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                # try to alloc again, protect last_node from eviction
                if dst_index is None:
                    self.inc_lock_ref(last_node)
                    self.evict(EvictParams(num_tokens=0, mamba_num=1))
                    dst_index = self.req_to_token_pool.mamba_pool.alloc(1)
                    self.dec_lock_ref(last_node)
                    assert dst_index is not None, "Can not alloc mamba cache"
                self._decompress_from_pool(last_node, dst_index)
                req.mamba_pool_idx = dst_index[0]
            else:
                dst_index = req.mamba_pool_idx.unsqueeze(0)
                self._decompress_from_pool(last_node, dst_index)

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.mamba_value = None  # mamba cache can not be split
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()

        # child time should be later than parent's time for mamba tombstone
        child.last_access_time = get_last_access_time()

        self.full_lru_list.remove_node(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # insert the new node and child into the lru lists, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        if child.mamba_value is not None:
            self.mamba_lru_list.insert_mru(child)
        return new_node

    def _bump_mamba_mru(self, node: TreeNode) -> None:
        """Dispatch MRU bump for a node that has any form of mamba state."""
        if node.mamba_compressed:
            if node.id in self._compressed_lru:
                self._compressed_lru.move_to_end(node.id)
        elif node.mamba_value is not None:
            self.mamba_lru_list.reset_node_mru(node)

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        chunked: bool = False,
        prev_prefix_len: int = 0,
    ) -> Tuple[int, bool]:
        # Update the last access time from root to leaf, so that
        # mamba will tombstone the node closer to root first
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
            self._bump_mamba_mru(node)
        if len(key) == 0:
            return 0, True

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            self._bump_mamba_mru(node)
            prefix_len = self.key_match_fn(node.key, key)

            if prev_prefix_len < total_prefix_length + prefix_len:
                start = max(0, prev_prefix_len - total_prefix_length)
                self.token_to_kv_pool_allocator.free(value[start:prefix_len])

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        mamba_value_exist = False
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.mamba_value = mamba_value
            self.full_lru_list.insert_mru(new_node)
            self.mamba_lru_list.insert_mru(new_node)
            node.children[child_key] = new_node
            self.full_evictable_size_ += len(value)
            self.mamba_evictable_size_ += len(mamba_value)
            self._enqueue_compression(new_node)
        elif node.mamba_value is None and not node.mamba_compressed:  # tombstone upgrade
            node.mamba_value = mamba_value
            self.full_lru_list.reset_node_mru(node)
            self.mamba_lru_list.insert_mru(node)
            self.mamba_evictable_size_ += len(mamba_value)
            node.last_access_time = get_last_access_time()
            self._enqueue_compression(node)
        else:  # mamba state already exists (full slot or compressed)
            mamba_value_exist = True
            self.full_lru_list.reset_node_mru(node)
            self._bump_mamba_mru(node)
            node.last_access_time = get_last_access_time()

        return total_prefix_length, mamba_value_exist

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.mamba_value is None and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.mamba_lock_ref == 0
            ), f"tombstone mamba_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.mamba_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            self.token_to_kv_pool_allocator.free(node.parent.value)
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is not None or node.mamba_compressed
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)
        # Compressed leaves no longer contribute to mamba_evictable_size_ (that
        # counter was decremented at drain time), so only decrement for the
        # full-pool case.
        if not node.mamba_compressed:
            self.mamba_evictable_size_ -= len(node.mamba_value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None
        node.mamba_compressed = False

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            # Only nodes carrying full-rank mamba state belong in mamba_lru_list;
            # compressed nodes live in _compressed_lru exclusively (Option A).
            if cur_node.mamba_value is not None:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                f"[{current_node.id}]",
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"mr={current_node.mamba_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"mll={self.mamba_lru_list.in_list(current_node)}",
                f"mv={current_node.mamba_value}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_mamba_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if current_node.mamba_value is not None:
                total_mamba_size += len(current_node.mamba_value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_mamba_size
