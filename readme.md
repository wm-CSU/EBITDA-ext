# EBITDA 抽取程序

先运行S1，将目标txt文件移入txt集合中；
再运行S3，实现txt到sentence-label pair的处理；
再运行main

基于Bert


file
```
project
│  file-tree.txt
│  main.py
│  readme.md
│  S1_preprocess.py
│  S2_EBITDA_locate.py
│  S3_sentence_division.py
│  S4_dataset.py
│  S5_model.py
│  S6_train.py
│  S7_evaluate.py
│  S8_predict.py
│  utils.py
│          
├─config
│      bert_config.json
│      sentence_embedding_config.json
│      
├─data
│  │  batch_one.xlsx
│  │  batch_two.xlsx
│  │  batch_two_for_test.xlsx
│  │  data_for_test.zip
│  │  label_map.json
│  │  matchtxt.xlsx
│  │  merge_data.xlsx
│  │  test.xlsx
│  │  
│  ├─adjust_txt
│  │      1031316_117152011000151_2.txt
│  │      1031316_117152012000827_2.txt
│  │      1031316_117152013000569_4.txt
│  │      ...
│  │      
│  ├─sent_label
│  │      1001606_110465910013133_2.txt
│  │      1001606_119312511215924_2.txt
│  │      1011174_119312512003719_2.txt
│  │      ...
│  │      
│  ├─sent_multi_label
│  │      1001606_110465910013133_2.txt
│  │      1001606_119312511215924_2.txt
│  │      1011174_119312512003719_2.txt
│  │      ...
│  │      
│  ├─test_adjust_txt
│  │      1035688_95012311055986_2.txt
│  │      105319_119312513210678_2.txt
│  │      1062613_110465911002707_4.txt
│  │      ...
│  │      
│  ├─test_txt_set
│  │      1035688_95012311055986_2.txt
│  │      105319_119312513210678_2.txt
│  │      1062613_110465911002707_4.txt
│  │      ...
│  │      
│  ├─txt_set
│  │      1001606_110465910013133_2.txt
│  │      1001606_119312511215924_2.txt
│  │      1011174_119312512003719_2.txt
│  │      ...
│          
├─log
├─model
│  └─bert-base-uncased
│      │  config.json
│      │  pytorch_model.bin
│      │  tokenizer.json
│      │  tokenizer_config.json
│      │  vocab.txt
│      │  
│      ├─BERT-data1-multi_class
│      │      bert-best_model.bin
│              
│
├─tools
│  │  all_utils.py
│  │  pytorchtools.py
│  │  
│  └─__pycache__
│          all_utils.cpython-37.pyc
│          pytorchtools.cpython-37.pyc
```


