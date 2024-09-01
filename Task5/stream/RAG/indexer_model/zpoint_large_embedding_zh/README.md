---
tags:
- mteb
model-index:
- name: zpoint_large_embedding_zh
  results:
  - task:
      type: STS
    dataset:
      type: C-MTEB/AFQMC
      name: MTEB AFQMC
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 56.52479321107392
    - type: cos_sim_spearman
      value: 60.72175935031135
    - type: euclidean_pearson
      value: 59.40990657564856
    - type: euclidean_spearman
      value: 60.72175934804556
    - type: manhattan_pearson
      value: 59.4134322847349
    - type: manhattan_spearman
      value: 60.724413114688225
  - task:
      type: STS
    dataset:
      type: C-MTEB/ATEC
      name: MTEB ATEC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 56.492631347325464
    - type: cos_sim_spearman
      value: 58.765171687177656
    - type: euclidean_pearson
      value: 63.236364373113844
    - type: euclidean_spearman
      value: 58.765171686714865
    - type: manhattan_pearson
      value: 63.22241814845751
    - type: manhattan_spearman
      value: 58.762780342648234
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (zh)
      config: zh
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 49.72
    - type: f1
      value: 46.588683657317084
  - task:
      type: STS
    dataset:
      type: C-MTEB/BQ
      name: MTEB BQ
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 73.07779128771674
    - type: cos_sim_spearman
      value: 75.03682691328844
    - type: euclidean_pearson
      value: 73.68098259699073
    - type: euclidean_spearman
      value: 75.03683037648963
    - type: manhattan_pearson
      value: 73.66963332679124
    - type: manhattan_spearman
      value: 75.02269337817758
  - task:
      type: Clustering
    dataset:
      type: C-MTEB/CLSClusteringP2P
      name: MTEB CLSClusteringP2P
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 58.2897067752906
  - task:
      type: Clustering
    dataset:
      type: C-MTEB/CLSClusteringS2S
      name: MTEB CLSClusteringS2S
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 48.79170511177673
  - task:
      type: Reranking
    dataset:
      type: C-MTEB/CMedQAv1
      name: MTEB CMedQAv1
      config: default
      split: test
      revision: None
    metrics:
    - type: map
      value: 91.10738371185181
    - type: mrr
      value: 92.82496031746031
  - task:
      type: Reranking
    dataset:
      type: C-MTEB/CMedQAv2
      name: MTEB CMedQAv2
      config: default
      split: test
      revision: None
    metrics:
    - type: map
      value: 90.06959035874831
    - type: mrr
      value: 92.00789682539683
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/CmedqaRetrieval
      name: MTEB CmedqaRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 27.132
    - type: map_at_10
      value: 40.400999999999996
    - type: map_at_100
      value: 42.246
    - type: map_at_1000
      value: 42.351
    - type: map_at_3
      value: 35.94
    - type: map_at_5
      value: 38.527
    - type: mrr_at_1
      value: 41.285
    - type: mrr_at_10
      value: 49.474000000000004
    - type: mrr_at_100
      value: 50.4
    - type: mrr_at_1000
      value: 50.438
    - type: mrr_at_3
      value: 46.891
    - type: mrr_at_5
      value: 48.353
    - type: ndcg_at_1
      value: 41.285
    - type: ndcg_at_10
      value: 47.159
    - type: ndcg_at_100
      value: 54.163
    - type: ndcg_at_1000
      value: 55.921
    - type: ndcg_at_3
      value: 41.678
    - type: ndcg_at_5
      value: 44.069
    - type: precision_at_1
      value: 41.285
    - type: precision_at_10
      value: 10.468
    - type: precision_at_100
      value: 1.611
    - type: precision_at_1000
      value: 0.183
    - type: precision_at_3
      value: 23.648
    - type: precision_at_5
      value: 17.229
    - type: recall_at_1
      value: 27.132
    - type: recall_at_10
      value: 57.977999999999994
    - type: recall_at_100
      value: 86.88
    - type: recall_at_1000
      value: 98.586
    - type: recall_at_3
      value: 41.487
    - type: recall_at_5
      value: 48.79
  - task:
      type: PairClassification
    dataset:
      type: C-MTEB/CMNLI
      name: MTEB Cmnli
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_accuracy
      value: 86.06133493686109
    - type: cos_sim_ap
      value: 92.54288511740305
    - type: cos_sim_f1
      value: 86.85572811163628
    - type: cos_sim_precision
      value: 83.72748969407681
    - type: cos_sim_recall
      value: 90.22679448211363
    - type: dot_accuracy
      value: 86.06133493686109
    - type: dot_ap
      value: 92.53922591080917
    - type: dot_f1
      value: 86.85572811163628
    - type: dot_precision
      value: 83.72748969407681
    - type: dot_recall
      value: 90.22679448211363
    - type: euclidean_accuracy
      value: 86.06133493686109
    - type: euclidean_ap
      value: 92.54287994398305
    - type: euclidean_f1
      value: 86.85572811163628
    - type: euclidean_precision
      value: 83.72748969407681
    - type: euclidean_recall
      value: 90.22679448211363
    - type: manhattan_accuracy
      value: 86.01322910402887
    - type: manhattan_ap
      value: 92.53060255301997
    - type: manhattan_f1
      value: 86.81441683456458
    - type: manhattan_precision
      value: 83.27249302125833
    - type: manhattan_recall
      value: 90.67103109656301
    - type: max_accuracy
      value: 86.06133493686109
    - type: max_ap
      value: 92.54288511740305
    - type: max_f1
      value: 86.85572811163628
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/CovidRetrieval
      name: MTEB CovidRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 78.899
    - type: map_at_10
      value: 86.232
    - type: map_at_100
      value: 86.331
    - type: map_at_1000
      value: 86.332
    - type: map_at_3
      value: 85.256
    - type: map_at_5
      value: 85.883
    - type: mrr_at_1
      value: 79.347
    - type: mrr_at_10
      value: 86.252
    - type: mrr_at_100
      value: 86.342
    - type: mrr_at_1000
      value: 86.343
    - type: mrr_at_3
      value: 85.283
    - type: mrr_at_5
      value: 85.91
    - type: ndcg_at_1
      value: 79.347
    - type: ndcg_at_10
      value: 89.143
    - type: ndcg_at_100
      value: 89.541
    - type: ndcg_at_1000
      value: 89.58
    - type: ndcg_at_3
      value: 87.227
    - type: ndcg_at_5
      value: 88.31400000000001
    - type: precision_at_1
      value: 79.347
    - type: precision_at_10
      value: 9.905
    - type: precision_at_100
      value: 1.0070000000000001
    - type: precision_at_1000
      value: 0.101
    - type: precision_at_3
      value: 31.261
    - type: precision_at_5
      value: 19.305
    - type: recall_at_1
      value: 78.899
    - type: recall_at_10
      value: 97.99799999999999
    - type: recall_at_100
      value: 99.684
    - type: recall_at_1000
      value: 100
    - type: recall_at_3
      value: 92.808
    - type: recall_at_5
      value: 95.46900000000001
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/DuRetrieval
      name: MTEB DuRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 27.107999999999997
    - type: map_at_10
      value: 82.525
    - type: map_at_100
      value: 85.168
    - type: map_at_1000
      value: 85.194
    - type: map_at_3
      value: 57.74399999999999
    - type: map_at_5
      value: 72.53699999999999
    - type: mrr_at_1
      value: 92.30000000000001
    - type: mrr_at_10
      value: 94.705
    - type: mrr_at_100
      value: 94.76599999999999
    - type: mrr_at_1000
      value: 94.76599999999999
    - type: mrr_at_3
      value: 94.55
    - type: mrr_at_5
      value: 94.64
    - type: ndcg_at_1
      value: 92.30000000000001
    - type: ndcg_at_10
      value: 89.23100000000001
    - type: ndcg_at_100
      value: 91.556
    - type: ndcg_at_1000
      value: 91.81700000000001
    - type: ndcg_at_3
      value: 88.558
    - type: ndcg_at_5
      value: 87.316
    - type: precision_at_1
      value: 92.30000000000001
    - type: precision_at_10
      value: 42.38
    - type: precision_at_100
      value: 4.818
    - type: precision_at_1000
      value: 0.488
    - type: precision_at_3
      value: 79.14999999999999
    - type: precision_at_5
      value: 66.63
    - type: recall_at_1
      value: 27.107999999999997
    - type: recall_at_10
      value: 89.914
    - type: recall_at_100
      value: 97.658
    - type: recall_at_1000
      value: 99.00099999999999
    - type: recall_at_3
      value: 59.673
    - type: recall_at_5
      value: 76.437
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/EcomRetrieval
      name: MTEB EcomRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 55.00000000000001
    - type: map_at_10
      value: 65.57600000000001
    - type: map_at_100
      value: 66.096
    - type: map_at_1000
      value: 66.103
    - type: map_at_3
      value: 63.217
    - type: map_at_5
      value: 64.562
    - type: mrr_at_1
      value: 55.00000000000001
    - type: mrr_at_10
      value: 65.57600000000001
    - type: mrr_at_100
      value: 66.096
    - type: mrr_at_1000
      value: 66.103
    - type: mrr_at_3
      value: 63.217
    - type: mrr_at_5
      value: 64.562
    - type: ndcg_at_1
      value: 55.00000000000001
    - type: ndcg_at_10
      value: 70.74000000000001
    - type: ndcg_at_100
      value: 73.001
    - type: ndcg_at_1000
      value: 73.223
    - type: ndcg_at_3
      value: 65.837
    - type: ndcg_at_5
      value: 68.264
    - type: precision_at_1
      value: 55.00000000000001
    - type: precision_at_10
      value: 8.7
    - type: precision_at_100
      value: 0.97
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 24.467
    - type: precision_at_5
      value: 15.86
    - type: recall_at_1
      value: 55.00000000000001
    - type: recall_at_10
      value: 87
    - type: recall_at_100
      value: 97
    - type: recall_at_1000
      value: 98.8
    - type: recall_at_3
      value: 73.4
    - type: recall_at_5
      value: 79.3
  - task:
      type: Classification
    dataset:
      type: C-MTEB/IFlyTek-classification
      name: MTEB IFlyTek
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 51.696806464024625
    - type: f1
      value: 40.02655259854763
  - task:
      type: Classification
    dataset:
      type: C-MTEB/JDReview-classification
      name: MTEB JDReview
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 88.87429643527206
    - type: ap
      value: 59.89821610336161
    - type: f1
      value: 83.98100504939507
  - task:
      type: STS
    dataset:
      type: C-MTEB/LCQMC
      name: MTEB LCQMC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 72.59510783330644
    - type: cos_sim_spearman
      value: 79.75022839599451
    - type: euclidean_pearson
      value: 79.54475341768782
    - type: euclidean_spearman
      value: 79.75021730266204
    - type: manhattan_pearson
      value: 79.53741020350834
    - type: manhattan_spearman
      value: 79.74152434784455
  - task:
      type: Reranking
    dataset:
      type: C-MTEB/Mmarco-reranking
      name: MTEB MMarcoReranking
      config: default
      split: dev
      revision: None
    metrics:
    - type: map
      value: 38.86925357762224
    - type: mrr
      value: 38.17460317460318
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/MMarcoRetrieval
      name: MTEB MMarcoRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 68.731
    - type: map_at_10
      value: 78.52
    - type: map_at_100
      value: 78.792
    - type: map_at_1000
      value: 78.797
    - type: map_at_3
      value: 76.586
    - type: map_at_5
      value: 77.876
    - type: mrr_at_1
      value: 71.003
    - type: mrr_at_10
      value: 79.03
    - type: mrr_at_100
      value: 79.27
    - type: mrr_at_1000
      value: 79.274
    - type: mrr_at_3
      value: 77.373
    - type: mrr_at_5
      value: 78.46600000000001
    - type: ndcg_at_1
      value: 71.003
    - type: ndcg_at_10
      value: 82.381
    - type: ndcg_at_100
      value: 83.504
    - type: ndcg_at_1000
      value: 83.627
    - type: ndcg_at_3
      value: 78.78699999999999
    - type: ndcg_at_5
      value: 80.94
    - type: precision_at_1
      value: 71.003
    - type: precision_at_10
      value: 9.961
    - type: precision_at_100
      value: 1.05
    - type: precision_at_1000
      value: 0.106
    - type: precision_at_3
      value: 29.694
    - type: precision_at_5
      value: 18.963
    - type: recall_at_1
      value: 68.731
    - type: recall_at_10
      value: 93.697
    - type: recall_at_100
      value: 98.546
    - type: recall_at_1000
      value: 99.515
    - type: recall_at_3
      value: 84.328
    - type: recall_at_5
      value: 89.42
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (zh-CN)
      config: zh-CN
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 76.79219905850707
    - type: f1
      value: 73.15228001501512
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (zh-CN)
      config: zh-CN
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 84.9562878278413
    - type: f1
      value: 84.0910677219451
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/MedicalRetrieval
      name: MTEB MedicalRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 57.8
    - type: map_at_10
      value: 64.732
    - type: map_at_100
      value: 65.315
    - type: map_at_1000
      value: 65.347
    - type: map_at_3
      value: 63.14999999999999
    - type: map_at_5
      value: 63.934999999999995
    - type: mrr_at_1
      value: 57.99999999999999
    - type: mrr_at_10
      value: 64.852
    - type: mrr_at_100
      value: 65.435
    - type: mrr_at_1000
      value: 65.467
    - type: mrr_at_3
      value: 63.266999999999996
    - type: mrr_at_5
      value: 64.072
    - type: ndcg_at_1
      value: 57.8
    - type: ndcg_at_10
      value: 68.14
    - type: ndcg_at_100
      value: 71.04899999999999
    - type: ndcg_at_1000
      value: 71.856
    - type: ndcg_at_3
      value: 64.813
    - type: ndcg_at_5
      value: 66.241
    - type: precision_at_1
      value: 57.8
    - type: precision_at_10
      value: 7.89
    - type: precision_at_100
      value: 0.927
    - type: precision_at_1000
      value: 0.099
    - type: precision_at_3
      value: 23.200000000000003
    - type: precision_at_5
      value: 14.62
    - type: recall_at_1
      value: 57.8
    - type: recall_at_10
      value: 78.9
    - type: recall_at_100
      value: 92.7
    - type: recall_at_1000
      value: 99
    - type: recall_at_3
      value: 69.6
    - type: recall_at_5
      value: 73.1
  - task:
      type: Classification
    dataset:
      type: C-MTEB/MultilingualSentiment-classification
      name: MTEB MultilingualSentiment
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 79.22333333333333
    - type: f1
      value: 79.01276765455862
  - task:
      type: PairClassification
    dataset:
      type: C-MTEB/OCNLI
      name: MTEB Ocnli
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_accuracy
      value: 85.32755820249052
    - type: cos_sim_ap
      value: 90.56118966152913
    - type: cos_sim_f1
      value: 86.28428927680798
    - type: cos_sim_precision
      value: 81.75803402646503
    - type: cos_sim_recall
      value: 91.34107708553326
    - type: dot_accuracy
      value: 85.32755820249052
    - type: dot_ap
      value: 90.56120405888693
    - type: dot_f1
      value: 86.28428927680798
    - type: dot_precision
      value: 81.75803402646503
    - type: dot_recall
      value: 91.34107708553326
    - type: euclidean_accuracy
      value: 85.32755820249052
    - type: euclidean_ap
      value: 90.56118966152913
    - type: euclidean_f1
      value: 86.28428927680798
    - type: euclidean_precision
      value: 81.75803402646503
    - type: euclidean_recall
      value: 91.34107708553326
    - type: manhattan_accuracy
      value: 85.43584190579317
    - type: manhattan_ap
      value: 90.52296007826511
    - type: manhattan_f1
      value: 86.42099949520444
    - type: manhattan_precision
      value: 82.7852998065764
    - type: manhattan_recall
      value: 90.3907074973601
    - type: max_accuracy
      value: 85.43584190579317
    - type: max_ap
      value: 90.56120405888693
    - type: max_f1
      value: 86.42099949520444
  - task:
      type: Classification
    dataset:
      type: C-MTEB/OnlineShopping-classification
      name: MTEB OnlineShopping
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 94.87999999999998
    - type: ap
      value: 93.12892276945414
    - type: f1
      value: 94.86921245385685
  - task:
      type: STS
    dataset:
      type: C-MTEB/PAWSX
      name: MTEB PAWSX
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 38.4367277229591
    - type: cos_sim_spearman
      value: 45.942712312151656
    - type: euclidean_pearson
      value: 44.96055989566686
    - type: euclidean_spearman
      value: 45.94279939044163
    - type: manhattan_pearson
      value: 44.979762134562925
    - type: manhattan_spearman
      value: 45.96004430328375
  - task:
      type: STS
    dataset:
      type: C-MTEB/QBQTC
      name: MTEB QBQTC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 41.45428416733968
    - type: cos_sim_spearman
      value: 43.462057455255845
    - type: euclidean_pearson
      value: 38.20089604291246
    - type: euclidean_spearman
      value: 43.46288438624811
    - type: manhattan_pearson
      value: 38.175045608320694
    - type: manhattan_spearman
      value: 43.468885824666344
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (zh)
      config: zh
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 65.61911213187778
    - type: cos_sim_spearman
      value: 66.70525921118497
    - type: euclidean_pearson
      value: 65.35554462551515
    - type: euclidean_spearman
      value: 66.70525921118497
    - type: manhattan_pearson
      value: 65.25174169329627
    - type: manhattan_spearman
      value: 66.6550752269368
  - task:
      type: STS
    dataset:
      type: C-MTEB/STSB
      name: MTEB STSB
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 81.27160581568329
    - type: cos_sim_spearman
      value: 83.34482829304406
    - type: euclidean_pearson
      value: 82.98079434913451
    - type: euclidean_spearman
      value: 83.34503180775212
    - type: manhattan_pearson
      value: 82.95256917013506
    - type: manhattan_spearman
      value: 83.31034894907503
  - task:
      type: Reranking
    dataset:
      type: C-MTEB/T2Reranking
      name: MTEB T2Reranking
      config: default
      split: dev
      revision: None
    metrics:
    - type: map
      value: 69.29054152015013
    - type: mrr
      value: 79.73472208788729
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/T2Retrieval
      name: MTEB T2Retrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 27
    - type: map_at_10
      value: 75.871
    - type: map_at_100
      value: 79.664
    - type: map_at_1000
      value: 79.725
    - type: map_at_3
      value: 53.14
    - type: map_at_5
      value: 65.365
    - type: mrr_at_1
      value: 88.642
    - type: mrr_at_10
      value: 91.732
    - type: mrr_at_100
      value: 91.818
    - type: mrr_at_1000
      value: 91.821
    - type: mrr_at_3
      value: 91.217
    - type: mrr_at_5
      value: 91.561
    - type: ndcg_at_1
      value: 88.642
    - type: ndcg_at_10
      value: 83.815
    - type: ndcg_at_100
      value: 87.689
    - type: ndcg_at_1000
      value: 88.266
    - type: ndcg_at_3
      value: 84.807
    - type: ndcg_at_5
      value: 83.53699999999999
    - type: precision_at_1
      value: 88.642
    - type: precision_at_10
      value: 41.725
    - type: precision_at_100
      value: 5.024
    - type: precision_at_1000
      value: 0.516
    - type: precision_at_3
      value: 74.10600000000001
    - type: precision_at_5
      value: 62.192
    - type: recall_at_1
      value: 27
    - type: recall_at_10
      value: 83.292
    - type: recall_at_100
      value: 95.66799999999999
    - type: recall_at_1000
      value: 98.56
    - type: recall_at_3
      value: 55.111
    - type: recall_at_5
      value: 69.327
  - task:
      type: Classification
    dataset:
      type: C-MTEB/TNews-classification
      name: MTEB TNews
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 54.346
    - type: f1
      value: 52.302508458396055
  - task:
      type: Clustering
    dataset:
      type: C-MTEB/ThuNewsClusteringP2P
      name: MTEB ThuNewsClusteringP2P
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 72.47709523787981
  - task:
      type: Clustering
    dataset:
      type: C-MTEB/ThuNewsClusteringS2S
      name: MTEB ThuNewsClusteringS2S
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 69.35293863978707
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/VideoRetrieval
      name: MTEB VideoRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 64.60000000000001
    - type: map_at_10
      value: 75.683
    - type: map_at_100
      value: 75.961
    - type: map_at_1000
      value: 75.96199999999999
    - type: map_at_3
      value: 74.083
    - type: map_at_5
      value: 75.03800000000001
    - type: mrr_at_1
      value: 64.60000000000001
    - type: mrr_at_10
      value: 75.683
    - type: mrr_at_100
      value: 75.961
    - type: mrr_at_1000
      value: 75.96199999999999
    - type: mrr_at_3
      value: 74.083
    - type: mrr_at_5
      value: 75.03800000000001
    - type: ndcg_at_1
      value: 64.60000000000001
    - type: ndcg_at_10
      value: 80.26299999999999
    - type: ndcg_at_100
      value: 81.487
    - type: ndcg_at_1000
      value: 81.5
    - type: ndcg_at_3
      value: 77.003
    - type: ndcg_at_5
      value: 78.708
    - type: precision_at_1
      value: 64.60000000000001
    - type: precision_at_10
      value: 9.43
    - type: precision_at_100
      value: 0.997
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 28.467
    - type: precision_at_5
      value: 17.9
    - type: recall_at_1
      value: 64.60000000000001
    - type: recall_at_10
      value: 94.3
    - type: recall_at_100
      value: 99.7
    - type: recall_at_1000
      value: 99.8
    - type: recall_at_3
      value: 85.39999999999999
    - type: recall_at_5
      value: 89.5
  - task:
      type: Classification
    dataset:
      type: C-MTEB/waimai-classification
      name: MTEB Waimai
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 89.36
    - type: ap
      value: 75.26507519569006
    - type: f1
      value: 87.89845508858562
language:
- zh
license: mit
library_name: sentence-transformers
---
<h2 align="left">ZPoint Large Embedding for Chinese</h2>

- **[2024-06-04]** Release zpoint_large_embedding_zh, and upload model weight to huggingface
- **[2024-06-05]** Add training details

### Training Details

**Base Model**
1) We chose [Stella](https://huggingface.co/infgrad/stella-mrl-large-zh-v3.5-1792d) as our base model.

**Training Data**
1) **Hard negative samping**
- For retrieval task, We sampled 10 hard negative passages/answers from top50-top200 related passages/answers for each query.
- For classification/clustering tasks, we sampled 5 hard negative samples from other classes/cluster for each sample.
- For classification/clustering tasks, we also used the category names of each class and cluster as positive and negative samples.

2) **Data synthesis by LLM (ZPoint-72B)**
- For retrieval tasks, we used LLM to rewrite each query, generating five different rewritten results. 
- For retrieval tasks, we also generated five new queries for some documents by LLM.
- For non-retrieval tasks, we used LLM to rewrite the queries, generating five rewritten results for each query.
- Finally, total amount of synthesized data is about 30 million.

3) **Collect more data for retrieval-type tasks**
- [miracl/miracl](https://huggingface.co/datasets/miracl/miracl)
- [FreedomIntelligence/Huatuo26M-Lite](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite)
- [PaddlePaddle/dureader_robust](https://huggingface.co/datasets/PaddlePaddle/dureader_robust) **C-MTEB test filtered**
- [THUIR/T2Ranking](https://huggingface.co/datasets/THUIR/T2Ranking) **C-MTEB test filtered**
- [Shitao/bge-reranker-data](https://huggingface.co/datasets/Shitao/bge-reranker-data)
- [Shitao/MLDR](https://huggingface.co/datasets/Shitao/MLDR)
- ...

***We constructed a dataset of approximately 100 million training samples through collection, machine translation, and LLM synthesis. This dataset includes data from various fields such as healthcare, law, electricity, automotive, and 3C (Consumer Electronics).***


**Training loss**
1) Multi-Task loss like [Piccolo](https://huggingface.co/sensenova/piccolo-large-zh-v2)
2) Matryoshka Representation Learning


### Example

```python
from sentence_transformers import SentenceTransformer
sentences1 = ["这个产品真垃圾"]
sentences2 = ["我太喜欢这个产品了"]
model = SentenceTransformer('iampanda/zpoint_large_embedding_zh')
embeddings_1 = model.encode(sentences1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```