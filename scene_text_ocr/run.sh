#!/bin/bash

source ~/.bashrc

# The users need to download and unzip the datasets Incidental-Scene-Text-2015 from https://rrc.cvc.uab.es/?ch=4&com=downloads into the $data_root=/nas/multimudal/data/Incidental-Scene-Text-2015
# Modify scps/PytorchOCR/config/det_train_db_resnet34_config.py
#   set config.dataset['train']['dataset']['file'] = r'/nas/multimudal/data/Incidental-Scene-Text-2015/detection/train.json'
#   set config.dataset['eval']['dataset']['file'] = r'/nas/multimudal/data/Incidental-Scene-Text-2015/detection/test.json'

train=0
if [ $train -eq 1 ]; then
    python3 -u scps/PytorchOCR/tools/det_train.py --config scps/PytorchOCR/config/det_train_db_resnet34_config.py

    python3 -u scps/PytorchOCR/tools/rec_train.py --config scps/PytorchOCR/config/rec_train_config.py
fi


process_ms_coco=0
if [ $process_ms_coco -eq 1 ]; then
    image_root=/nas/multimudal/data/ms-coco

    for subset in train val; do
        output_dir=`pwd`/expts/det_results/ms-coco.${subset}/imgs
        mkdir -p $output_dir

        # scene text detection
        find $image_root/${subset}2014/ -iname "*.jpg" > $output_dir/../ms-coco.${subset}.list

        python3 -u scps/PytorchOCR/tools/det_infer.py --model_path scps/PytorchOCR/output/DBNet_ResNet34/checkpoint/best.pth \
                                                      --file_path $output_dir/../ms-coco.${subset}.list \
                                                      --output_path $output_dir

        # scene text recognition
        find $output_dir/ -iname "*.png" > $output_dir/../ms-coco.${subset}.det.list
        python3 -u scps/PytorchOCR/tools/rec_infer.py --model_path scps/PytorchOCR/output/CRNN/checkpoint/best.pth \
                                                      --img_path $output_dir/../ms-coco.${subset}.det.list \
                                                      --output_file $output_dir/../ms-coco.${subset}.ocr.results

        python3 -u scps/PytorchOCR/tools/filter_ocr_results.py --input_file $output_dir/../ms-coco.${subset}.ocr.results \
                                                               --tokenizer `pwd`/../prepare_ms-coco-data/expts/data/ms-coco/tokenizer.pkl \
                                                               --output_file $output_dir/../ms-coco.${subset}.ocr.results.fil.pkl

        # re-select objects' features, image classification features, and German text
        python3 -u scps/collect_coco_data.py --input_key_file $output_dir/../ms-coco.${subset}.ocr.results.fil.pkl \
                                        --input_visual_feat_file `pwd`/../augment_ms-coco-data/expts/feats/ms-coco-faster-rcnn/${subset}_cascade_fastrcnn_featex.pkl \
                                        --input_image_class_file `pwd`/../augment_ms-coco-data/expts/classes/ms-coco-inceptionv3-classes/${subset}_inceptionv3_classes.pkl \
                                        --input_text_file `pwd`/../augment_ms-coco-data/expts/data/ms-coco-bpe/${subset}.trans.pkl \
                                        --output_visual_feat_file $output_dir/../ms-coco.${subset}_cascade_fastrcnn_featex.pkl \
                                        --output_image_class_file $output_dir/../ms-coco.${subset}_inceptionv3_classes.pkl \
                                        --output_text_file $output_dir/../ms-coco.${subset}_trans.pkl
    done
fi

process_multi30k=1
if [ $process_multi30k -eq 1 ]; then
    image_root=/nas/multimudal/data/Flickr30k/flickr30k_images

    output_dir=`pwd`/expts/det_results/multi30k/imgs
    mkdir -p $output_dir

    # scene text detection
    find $image_root/ -iname "*.jpg" > $output_dir/../multi30k.image.list

    python3 -u scps/PytorchOCR/tools/det_infer.py --model_path scps/PytorchOCR/output/DBNet_ResNet34/checkpoint/best.pth \
                                                  --file_path $output_dir/../multi30k.image.list \
                                                  --output_path $output_dir

    for subset in train val; do
        # scene text recognition
        cat /nas/multimudal/data/multi30k-dataset/data/task1/image_splits/${subset}.txt | perl -pe 's/\.jpg/_/; s/^/\//' > .list
        find $output_dir -iname "*.png" | grep -F -f .list > $output_dir/../multi30k.${subset}.det.list

        python3 -u scps/PytorchOCR/tools/rec_infer.py --model_path scps/PytorchOCR/output/CRNN/checkpoint/best.pth \
                                                      --img_path $output_dir/../multi30k.${subset}.det.list \
                                                      --output_file $output_dir/../multi30k.${subset}.ocr.results

        python3 -u scps/PytorchOCR/tools/filter_ocr_results.py --input_file $output_dir/../multi30k.${subset}.ocr.results \
                                                               --tokenizer `pwd`/../run.19/expts/data/ms-coco/tokenizer.pkl \
                                                               --output_file $output_dir/../multi30k.${subset}.ocr.results.fil.pkl

        # re-select objects' features, image classification features, and German text
        python3 -u scps/collect_multi30k_data.py --input_key_file $output_dir/multi30k.${subset}.ocr.results.fil.pkl \
                                        --input_visual_feat_file `pwd`/../augment_ms-coco-data/expts/feats/multi30k-faster-rcnn/${subset}_cascade_fastrcnn_featex.pkl \
                                        --input_image_class_file `pwd`/../augment_ms-coco-data/expts/classes/multi30k-inceptionv3-classes/${subset}_inceptionv3_classes.pkl \
                                        --input_text_file `pwd`/../augment_ms-coco-data/expts/data/multi30k-bpe/${subset}.trans.pkl \
                                        --output_visual_feat_file $output_dir/../multi30k.${subset}_cascade_fastrcnn_featex.pkl \
                                        --output_image_class_file $output_dir/../multi30k.${subset}_inceptionv3_classes.pkl \
                                        --output_text_file $output_dir/../multi30k.${subset}_trans.pkl
    done

fi
