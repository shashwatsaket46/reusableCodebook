#pragma once
#include <array>
#include <cstddef>

#define HIGH_ACC_FAST_SCAN

static constexpr std::array<size_t, 6> EXP_EVAL_KS = {1, 2, 4, 8, 16, 32};
static constexpr size_t EXP_EVAL_KMAX = 32;
static constexpr size_t EXP_TOPK = 32;
static constexpr size_t EXP_ROUND = 3;
