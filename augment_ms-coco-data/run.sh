#!/bin/bash -l

source ~/.bashrc

data_root=/nas/multimodal/data/ms-coco

# Translate MS-COCO data to German 
load_coco_bpe=0
if [ $load_coco_bpe -eq 1 ]; then
    model_dir=`pwd`/scps/fairseq/models
    annotation_file=$data_root/annotations/dataset_coco.json
    output_dir=`pwd`/expts/data/ms-coco-bpe

    mkdir -p $output_dir

    python -u scps/translate.py --input-annotation-file $annotation_file --input-image-dir $data_root --fairseq-model-dir $model_dir --output-dir $output_dir
fi

# Load augmented MS-COCO with confidence
load_coco_bpe_with_confidence=1
if [ $load_coco_bpe_with_confidence -eq 1 ]; then
    output_dir=`pwd`/expts/data/ms-coco-bpe-confidence
    mkdir -p $output_dir

    for subset in train val test; do
        in_trans_file=`pwd`/expts/data/ms-coco-bpe/${subset}.trans.pkl
        dict_file=$output_dir/${subset}.dict.pkl
        out_trans_file=$output_dir/$subset}.trans.pkl

        python -u scps/add_conf.py --input-trans-file $in_trans_files --output-file $dict_file
        python -u scps/update_trans.py --input-trans-file $in_trans_file --conf-dict-file $dict_file --output-file $out_trans_file
    done

fi


data_root=/nas/multimodal/data

# Load Multi30K dataset with BPE
load_multi30k_bpe=0
if [ $load_multi30k_bpe -eq 1 ]; then
    multi30k_root=$data_root/multi30k/data/task1
    image_root=$data_root/Flickr30k/flickr30k_images
    image_list=$multi30k_root/image_splits/train_images.txt

    output_dir=`pwd`/expts/data/multi30k-bpe
    mkdir -p $output_dir

    for prefix in train val test_2016_flickr; do
        python -u scps/load_multi30k_data_with_bpe.py --image_list $multi30k_root/image_splits/${prefix}.txt \
                                             --tokenizer `pwd`/expts/data/ms-coco-bpe/tokenizer.pkl \
                                             --codes `pwd`/expts/data/ms-coco-bpe/bpe.codes.pkl \
                                             --en_trans $multi30k_root/tok/${prefix}.lc.norm.tok.en \
                                             --de_trans $multi30k_root/tok/${prefix}.lc.norm.tok.de \
                                             --image_dir $image_root --output_prefix $output_dir/${prefix}
    done

fi

