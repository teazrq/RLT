#!/usr/bin/env Rscript

# RLT Regression QC Demo
# 演示如何使用QC系统

cat("=== RLT Regression QC Demo ===\n\n")

# 加载QC函数
source("R/quick_regression_qc.r")

# 1. 运行快速QC测试
cat("1. Running Quick QC Test:\n")
quick_qc_result <- quick_regression_qc()
cat("Quick QC result:", ifelse(quick_qc_result, "PASSED", "FAILED"), "\n\n")

# 2. 测试输入验证
cat("2. Testing Input Validation:\n")

# 正常数据
x_good <- matrix(rnorm(1000), 1000, 10)
y_good <- rnorm(1000)
param_good <- list(ntrees = 100, mtry = 5, nmin = 10)

cat("Testing good data:\n")
validate_regression_inputs(x_good, y_good, param_good)

# 错误数据 - mtry太大
param_bad_mtry <- list(ntrees = 100, mtry = 15, nmin = 10)
cat("\nTesting bad mtry:\n")
validate_regression_inputs(x_good, y_good, param_bad_mtry)

# 错误数据 - 维度不匹配
y_bad_length <- y_good[1:500]
cat("\nTesting dimension mismatch:\n")
validate_regression_inputs(x_good, y_bad_length, param_good)

# 错误数据 - NA值
x_with_na <- x_good
x_with_na[1, 1] <- NA
cat("\nTesting NA values:\n")
validate_regression_inputs(x_with_na, y_good, param_good)

cat("\n=== Demo Completed ===\n") 