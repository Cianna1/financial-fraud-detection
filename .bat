@echo off
REM ===== 修改你的路径信息 =====
SET SPARK_HOME="C:\spark\spark-3.3.1-bin-hadoop3"
SET APP_PATH="C:\Users\陈一心\PycharmProjects\financial fraud\streaming_job.py"

REM ===== 设置 Kafka Connector 依赖版本 =====
SET KAFKA_PACKAGE=org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.1

REM ===== 执行 Spark Streaming 程序 =====
%SPARK_HOME%\bin\spark-submit ^
  --packages %KAFKA_PACKAGE% ^
  %APP_PATH%

pause
