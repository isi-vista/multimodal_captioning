#!/bin/bash

train=1
if [ $train -eq 1 ]; then
    # Download and prepare data visual_genome_1.2 as described in README.md
    # The prepared data is stored in scps/densecap-tensorflow/data
    # Download pretrained model from https://jbox.sjtu.edu.cn/l/j5EeUN and store it in ./output/coco_2014_train+coco_2014_valminusminival/. In our experiments, we are using res101 based network.

    bash scripts/dense_cap_train.sh visual_genome_1.2 \
                                    res101 \
                                    output/coco_2014_train+coco_2014_valminusminival/checkpoint.ckpt \
                                    data \
                                    1 
fi

# performance evaluation
evaluate=0
if [ $evaluate -eq 1 ]; then
    # the input is the trained models in ./output/ckpt/
    bash scripts/dense_cap_test.sh 0 \
                                   output/ckpt/res50_densecap_iter_500000.ckpt \
                                   vg_1.2_test
fi

# extract dense caption features 
test=0
if [ $test -eq 1 ]; then
    # prepare train, val, test image list and store the list file in ./inputs/

    for subset in train val test; do
        bash scripts/dense_cap_test_batch.sh output/dc_tune_conv/vg_1.2_train/ \
                                             data/1.2/vocabulary.txt \
                                             inputs/coco_$subset.txt \
                                             inputs/coco_$subset.pkl

        # tokenization and applying BPE
        python tools/load_data_with_bpe.py --tokenizer ../augment_ms-coco-data/expts/data/ms-coco-bpe/tokenizer.pkl \
                                           --codes ../augment_ms-coco-data/expts/data/ms-coco-bpe/bpe.codes.pkl \
                                           --inputs_trans inputs/coco_$subset.pkl \
                                           --output_prefix $subset
    done
fi
