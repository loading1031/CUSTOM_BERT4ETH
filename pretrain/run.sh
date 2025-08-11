# "훈련"

# 1. 데이터 통합(정상/피싱 거래 통합 및 RR#1 전략)
# 커맨드 경로: pretrain/

python gen_seq.py --bizdate=0000
# 본 스크립트는 normal, phsihing 데이터에만 국한
# 다른 anony, tornado 데이터를 쓴다면 --bizate=all 로 구분 필요

# 2. mlm 학습
# 커맨드 경로: pretrain/

python -m mlm.run_pretrain --bizdate=0000

# 3. cluster 학습
# 커맨드 경로: pretrain/
python -m cluster.run_pretrain --bizdate=0000

# 4. mlm/cluster 임베딩 최종
# 커맨드 경로: pretrain/
python -m run_embed --bizdate=0000 --init_checkpoint=mlm/cpkt_local/epoch_50.pth
python -m run_embed --bizdate=0000 --init_checkpoint=cluster/cpkt_local/epoch_50.pth

# 5. mlm/cluster 최종 임베딩 피싱 시각화 결과 확인
# 커맨드 경로: eval/
 python phish_visualize.py --data_dir=mlm_epoch_50
 python phish_visualize.py --data_dir=cluster_epoch_50

 # 6. 피싱 분류 성능 측정
 # 커맨드 경로: eval/
 python phish_detection_mlp.py --input_dir=../outputs/mlm_epoch_50_0000
 python phish_detection_rf.py --input_dir=../outputs/mlm_epoch_50_0000

 python phish_detection_mlp.py --input_dir=../outputs/cluster_epoch_5_0000
 python phish_detection_rf.py --input_dir=../outputs/cluster_epoch_5_0000
