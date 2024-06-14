#!/bin/bash

# vitbit 폴더 내의 실행 파일 경로
VITBIT_EXECUTABLE="../TC_ViT/TC_ViT"

# profiling 폴더 내의 프로파일링 결과 파일 경로
PROFILING_RESULT="./profile_TC_ViT"

# nsight compute를 사용하여 실행 파일 프로파일링
ncu -o $PROFILING_RESULT --set full $VITBIT_EXECUTABLE