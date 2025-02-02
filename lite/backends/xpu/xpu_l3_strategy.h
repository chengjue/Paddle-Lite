// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

#include "lite/backends/xpu/xpu_l3_cache_block.h"
#include "lite/utils/macros.h"

namespace paddle {
namespace lite {

class XPUL3Planner {
 public:
  void set_current_query_shape(
      const std::vector<std::vector<int64_t>>& query_shape, size_t l3_size) {
    query_shape_.clear();
    if (l3_size <= 0) return;
    for (size_t node_idx = 0; node_idx < query_shape.size(); node_idx++) {
      for (size_t shape_idx = 0; shape_idx < query_shape[node_idx].size();
           shape_idx++) {
        query_shape_.push_back(query_shape[node_idx][shape_idx]);
      }
    }
    query_shape_.push_back(l3_size);
  }

  std::vector<size_t>* get_current_plan() {
    if (plans_.size() <= 0 || query_shape_.empty()) {
      return nullptr;
    } else {
      auto it = plans_.lower_bound(query_shape_);
      if (it == plans_.end()) {
        LOG(INFO) << "new query_shape, use the first L3 cache plan";
        return &(plans_.begin()->second);
      } else {
        return &(it->second);
      }
    }
  }
  bool if_find_plan_query_shape() {
    return (!query_shape_.empty() && plans_.find(query_shape_) != plans_.end());
  }
  // greedy strategy
  void run_autotune_greedy(const std::vector<XPUL3CacheBlock*>& l3_block_dict,
                           size_t l3_size) {
    if (l3_block_dict.size() == 0 || l3_size <= 0 || query_shape_.size() == 0 ||
        plans_.find(query_shape_) != plans_.end()) {
      return;
    }
    VLOG(3) << "AutoTune(greedy) XPU L3 Cache Block Start.";
    struct node {
      size_t weights = 0;
      size_t scores = 0;
      float ratio = 0.f;  // score/weights
    };
    std::vector<std::vector<node>> records;
    std::vector<size_t> record_map;
    size_t total_scores = 0;
    for (size_t block_idx = 0; block_idx < l3_block_dict.size(); block_idx++) {
      XPUL3CacheBlock* cur_block = l3_block_dict[block_idx];
      std::vector<size_t>& history = cur_block->history_;
      auto history_size = history.size();
      size_t score = 0;
      VLOG(3) << "Block Idx is " << block_idx;
      if (history_size > l3_tune_level_) {
        std::vector<node> block_nodes{node()};
        std::sort(history.begin(), history.end());
        for (size_t i = 0; i < history_size; i++) {
          VLOG(3) << "Size History : " << i << " is " << history[i];
          if (history[i] > l3_size) {
            break;
          }
          if (history[i] <= 0) continue;
          score += history[i];
          if (i == history_size - 1 || history[i + 1] != history[i]) {
            node cur_node;
            cur_node.weights = history[i];
            cur_node.scores = score;
            cur_node.ratio = score * 1.0 / cur_node.weights;
            if (block_nodes.back().ratio < cur_node.ratio) {
              if (block_nodes.size() < 2) {
                block_nodes.push_back(cur_node);
              } else {
                block_nodes.back().weights = cur_node.weights;
                block_nodes.back().scores = cur_node.scores;
                block_nodes.back().ratio = cur_node.ratio;
              }
              VLOG(3) << "History : " << i
                      << ", Node Weights is:" << cur_node.weights
                      << ", Node Scores is: " << score
                      << ", profit: " << cur_node.ratio;
            }
          }
        }
        total_scores += score;
        records.push_back(block_nodes);
        record_map.push_back(block_idx);
      }
    }
    if (records.size() <= 0) {
      return;
    }
    {  // greedy search
      std::vector<int> ret_index(records.size());
      std::iota(ret_index.begin(), ret_index.end(), 0);
      auto customGreater = [&records](int a, int b) {
        if (records[a].back().ratio > records[b].back().ratio) {
          return true;
        } else if (records[a].back().ratio == records[b].back().ratio) {
          return records[a].back().weights > records[b].back().weights;
        } else {
          return false;
        }
      };
      std::stable_sort(ret_index.begin(), ret_index.end(), customGreater);
      int total_l3_size = 0;
      std::vector<size_t> final_res(l3_block_dict.size() + 1, 0);
      for (size_t i = 0; i < ret_index.size(); i++) {
        int block_idx = record_map[ret_index[i]];
        const node& select_node = records[ret_index[i]].back();
        if (select_node.weights > 0 &&
            total_l3_size + select_node.weights <= l3_size) {
          final_res[block_idx] = select_node.weights;
          total_l3_size += select_node.weights;
          VLOG(3) << "BLOCK IDX is " << block_idx << ", Acquired L3 Size is "
                  << select_node.weights << ", profit" << select_node.ratio;
        }
      }
      int xdnn_ctx_l3_size = (l3_size - total_l3_size) / 64 * 64;
      CHECK_GE(xdnn_ctx_l3_size, 0) << "invalid remaining xdnn L3 size: "
                                    << xdnn_ctx_l3_size;
      LOG(INFO) << "greedy search L3 tune strategy, lite use L3: "
                << total_l3_size << ", xdnn left l3 size: " << xdnn_ctx_l3_size;

      double l3_global_ratio =
          static_cast<double>(total_l3_size) / total_scores;
      VLOG(3) << "Tensor Space in L3 / Tensor Space in Global :"
              << l3_global_ratio * 100 << " %";
      final_res[l3_block_dict.size()] = xdnn_ctx_l3_size;
      plans_.insert({query_shape_, final_res});
      VLOG(3) << "AutoTune(greedy) XPU L3 Cache Block End.";
      return;
    }
  }
  void run_autotune(const std::vector<XPUL3CacheBlock*>& l3_block_dict,
                    size_t l3_size) {
    if (l3_block_dict.size() == 0 || l3_size <= 0 || query_shape_.size() == 0 ||
        plans_.find(query_shape_) != plans_.end()) {
      return;
    }
    // greedy search
    if (l3_tune_level_ <= 1) {
      return run_autotune_greedy(l3_block_dict, l3_size);
    }
    VLOG(3) << "AutoTune XPU L3 Cache Block Start";
    struct node {
      size_t weights = 0;
      size_t scores = 0;
      std::vector<size_t> choices{0};
    };
    std::vector<std::vector<node>> records;
    std::vector<size_t> record_map;
    size_t total_scores = 0;
    for (size_t block_idx = 0; block_idx < l3_block_dict.size(); block_idx++) {
      XPUL3CacheBlock* cur_block = l3_block_dict[block_idx];
      std::vector<size_t>& history = cur_block->history_;
      auto history_size = history.size();
      size_t score = 0;
      VLOG(3) << "Block Idx is " << block_idx;
      if (history_size > l3_tune_level_) {
        std::vector<node> block_nodes{node()};
        std::sort(history.begin(), history.end());
        for (size_t i = 0; i < history_size; i++) {
          VLOG(3) << "Size History : " << i << " is " << history[i];
          if (history[i] > l3_size) {
            break;
          }
          score += history[i];
          if (i == history_size - 1 || history[i + 1] != history[i]) {
            node cur_node;
            cur_node.weights = history[i];
            cur_node.choices = {history[i]};
            cur_node.scores = score;
            block_nodes.push_back(cur_node);
            VLOG(3) << "Node Weights is:" << cur_node.weights
                    << ", Node Scores is: " << score;
          }
        }
        total_scores += score;
        records.push_back(block_nodes);
        record_map.push_back(block_idx);
      }
    }
    if (records.size() <= 0) {
      return;
    }
    std::vector<node> res(records[0]);
    for (size_t block_idx = 1; block_idx < records.size(); block_idx++) {
      std::vector<node> new_nodes;
      for (size_t node_idx = 0; node_idx < records[block_idx].size();
           node_idx++) {
        for (size_t res_idx = 0; res_idx < res.size(); res_idx++) {
          node cur_node;
          size_t cur_weights =
              records[block_idx][node_idx].weights + res[res_idx].weights;
          if (cur_weights > l3_size) {
            break;
          }
          cur_node.scores =
              records[block_idx][node_idx].scores + res[res_idx].scores;
          cur_node.weights = cur_weights;
          cur_node.choices = res[res_idx].choices;
          cur_node.choices.push_back(records[block_idx][node_idx].choices[0]);
          new_nodes.push_back(cur_node);
        }
      }
      struct {
        bool operator()(node a, node b) const {
          if (a.weights < b.weights) {
            return true;
          } else if (a.weights == b.weights) {
            return a.scores > b.scores;
          } else {
            return false;
          }
        }
      } customLess;

      std::sort(new_nodes.begin(), new_nodes.end(), customLess);
      std::vector<bool> stay(new_nodes.size(), true);
      for (int i = new_nodes.size() - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
          if (new_nodes[j].scores >= new_nodes[i].scores) {
            stay[i] = false;
            break;
          }
        }
      }
      res.clear();
      for (size_t i = 0; i < new_nodes.size(); i++) {
        if (stay[i] == true) {
          res.push_back(new_nodes[i]);
        }
      }
      VLOG(3) << "XPU L3 Block IDX is " << block_idx
              << ", Choices before filter are " << new_nodes.size()
              << ", Choices after filter are " << res.size();
    }
    // final result: res.back().choices
    //               std::vector<size_t> record_map;
    for (size_t i = 0; i < res.back().choices.size(); i++) {
      VLOG(3) << "BLOCK IDX is " << i << ", Acquired L3 Size is "
              << res.back().choices[i];
    }
    double l3_global_ratio = static_cast<double>(res.back().scores) /
                             static_cast<double>(total_scores);
    VLOG(3) << "Tensor Space in L3 / Tensor Space in Global :"
            << l3_global_ratio * 100 << " %";

    size_t block_l3_size = std::accumulate(
        res.back().choices.begin(), res.back().choices.end(), 0);
    size_t xdnn_ctx_l3_size = (l3_size - block_l3_size) / 64 * 64;

    VLOG(3) << "Block L3 Size : " << block_l3_size
            << ", XDNN Ctx L3 Size : " << xdnn_ctx_l3_size;

    std::vector<size_t> final_res(l3_block_dict.size() + 1, 0);
    for (size_t i = 0; i < res.back().choices.size(); i++) {
      final_res[record_map[i]] = res.back().choices[i];
    }
    final_res[l3_block_dict.size()] = xdnn_ctx_l3_size;
    plans_.insert({query_shape_, final_res});
    VLOG(3) << "AutoTune XPU L3 Cache Block End.";
  }

  void set_l3_tune_level(int v) {
    if (v < 0) {
      LOG(FATAL) << "invalid l3_tune_leve value: " << v;
    }
    l3_tune_level_ = v;
    LOG(INFO) << "set_l3_tune_level: " << l3_tune_level_;
  }

 private:
  // plans_ format: [query_shape_] : [block0 block1 ... blockn xdnn_ctx_l3_size]
  std::map<std::vector<int64_t>, std::vector<size_t>> plans_;
  // query_shape format: [shape0 shape1 ... shapen l3_size]
  std::vector<int64_t> query_shape_;
  int l3_tune_level_{1};  // tensor reuse threshold in graph
};

}  // namespace lite
}  // namespace paddle
