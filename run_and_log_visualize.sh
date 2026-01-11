#!/bin/bash

# 检查是否提供了脚本路径参数
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_script_to_execute>"
  echo "Example: $0 ./scripts/supervised_learning/RmGPT_pump_supervised.sh"
  exit 1
fi

# 1. 获取命令行输入的脚本路径
SCRIPT_TO_EXECUTE="$1"

# 2. 从输入路径中提取脚本的基本名称（不含扩展名）
#     basename ${SCRIPT_TO_EXECUTE} .sh
#    这个命令会提取文件名，例如 "RmGPT_pump_supervised"
SCRIPT_BASENAME=$(basename "${SCRIPT_TO_EXECUTE}" .sh) 


# 3. 从脚本所在的目录中提取目录名称
#    例如，如果 SCRIPT_TO_EXECUTE 是 /path/to/your/script.sh
#    SCRIPT_DIR 会是 /path/to/your
#    然后 basename "${SCRIPT_DIR}" 会提取出 "your"
SCRIPT_DIR=$(dirname "${SCRIPT_TO_EXECUTE}")
PARENT_DIR_NAME=$(basename "${SCRIPT_DIR}")


# 2. 生成当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 3. 定义日志文件存放的目录
LOG_DIR="./logs/visualize/${PARENT_DIR_NAME}/" # 也可以是其他你喜欢的位置

# 检查目录是否存在，如果不存在则创建
if [ ! -d "$LOG_DIR" ]; then
  echo "Directory '$LOG_DIR' not found. Creating it..."
  mkdir -p "$LOG_DIR"
fi

# 4. 构建带时间戳和文件种类的日志文件名
OUTPUT_LOG="${LOG_DIR}${SCRIPT_BASENAME}_${TIMESTAMP}_output.log"
ERROR_LOG="${LOG_DIR}${SCRIPT_BASENAME}_${TIMESTAMP}_error.log"

# 5. 执行脚本，并将输出和异常分别重定向到带时间戳的文件
echo "Starting script: ${SCRIPT_TO_EXECUTE}"
echo "Output log file: ${OUTPUT_LOG}"
echo "Error log file:  ${ERROR_LOG}"
echo "Timestamp:       ${TIMESTAMP}"

bash "${SCRIPT_TO_EXECUTE}" "${PARENT_DIR_NAME}"> "${OUTPUT_LOG}" 2> "${ERROR_LOG}"

# 6. (可选) 检查退出状态，判断是否成功
if [ $? -eq 0 ]; then
    echo "Script finished successfully. Output logged to ${OUTPUT_LOG}, errors to ${ERROR_LOG}."
else
    echo "Script finished with errors. Check ${ERROR_LOG} for details."
fi