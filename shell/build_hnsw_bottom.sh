# MEX=48
# M=16
# efC=500
# ./test/build_hnsw_bottom --base_data_path /SSD/Text-to-Image/base.10M.fbin \
# --metric ip_float \
# --result_hnsw_index_path /SSD/models/NGFix/t2i10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --M ${M} --MEX ${MEX} --efC ${efC}

# MEX=48
# M=16
# efC=500
# ./test/build_hnsw_bottom --base_data_path /SSD/LAION/LAION_base_imgemb_10M.fbin \
# --metric ip_float \
# --result_hnsw_index_path /SSD/models/NGFix/laion10M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --M ${M} --MEX ${MEX} --efC ${efC}


# MEX=48
# M=16
# efC=500
# ./test/build_hnsw_bottom --base_data_path /SSD/MainSearch/base.fbin \
# --metric ip_float \
# --result_hnsw_index_path /SSD/models/NGFix/mainse_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --M ${M} --MEX ${MEX} --efC ${efC}

# MEX=48
# M=16
# efC=500
# ./test/build_hnsw_bottom --base_data_path /SSD/SIFT1M/sift_base.fbin \
# --metric l2_float \
# --result_hnsw_index_path /SSD/models/NGFix/sift1M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}.index \
# --M ${M} --MEX ${MEX} --efC ${efC}


# ============================================================
# WebVid2.5M HNSW Bottom Graph
# NGFix paper setting
#
# Dataset:
#   base = 2,505,000 x 512
#
# Distance:
#   cosine -> ip_float
#
# HNSW parameters:
#   M    = 16
#   MEX  = 48
#   efC  = 2000
# ============================================================

ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
DATA=/data/RoarGraph-main/data/clip-webvid-2.5M
MODEL=${ROOT}/models

M=16
MEX=48
efC=2000

mkdir -p ${MODEL}

${ROOT}/build/test/build_hnsw_bottom \
    --base_data_path ${DATA}/base.2.5M.fbin \
    --metric ip_float \
    --result_hnsw_index_path ${MODEL}/webvid2.5M_HNSWBottom_M${M}_efC${efC}_MEX${MEX}_ip.index \
    --M ${M} \
    --MEX ${MEX} \
    --efC ${efC}