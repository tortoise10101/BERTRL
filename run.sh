export DATASET="fb237"
export HOPS="3"
export DATAF="full" # "1000" # "2000"
export SUFFIX="_hop${HOPS}_${DATAF}_neg10_path3_max_inductive"
# test
export MODEL_SUFFIX="_hop${HOPS}_${DATAF}_neg10_path3_max_inductive"
export DATA_SUFFIX="_hop${HOPS}_${DATAF}_neg10_path3_max_inductive"
export OUTPUT_SUFFIX="_hop${HOPS}_${DATAF}_neg10_path3_max_inductive_new"

function train() {
    CUDA_VISIBLE_DEVICES=0 python run_bertrl.py \
    --model_name_or_path bert-base-cased \
    --task_name MRPC \
    --do_train \
    --data_dir ./bertrl_data/${DATASET}${SUFFIX} \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 2.0 \
    --output_dir output_${DATASET}${SUFFIX} \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --do_predict \
    --do_eval \
    --eval_steps 1000 \
    --save_steps 1000 \
    --per_device_eval_batch_size 750 \
    --overwrite_cache
    
    #   --evaluate_during_training \
    #   --warmup_steps 20000 \
    #   --save_steps 20000
}

function test() {
    CUDA_VISIBLE_DEVICES=0  python run_bertrl.py \
    --model_name_or_path ./output_${DATASET}${MODEL_SUFFIX}/ \
    --task_name MRPC \
    --do_predict \
    --data_dir ./bertrl_data/${DATASET}${DATA_SUFFIX}\
    --max_seq_length 128 \
    --output_dir output_${DATASET}${OUTPUT_SUFFIX}/ \
    --per_device_eval_batch_size 1000 \
    --overwrite_output_dir \
    --overwrite_cache
}

function eval() {
    python eval_bertrl.py -d ${DATASET}${DATA_SUFFIX}
}

function preprocess() {
    python load_data.py -d $DATASET -st train --part $DATAF --hop $HOPS --ind_suffix "_ind" --suffix ${SUFFIX:10}
    python load_data.py -d $DATASET -st test --part $DATAF --hop $HOPS --ind_suffix "_ind" --suffix ${SUFFIX:10}
    python load_data.py -d $DATASET -st dev --part $DATAF --hop $HOPS --ind_suffix "_ind" --suffix ${SUFFIX:10}
}

if [ "$1" == "train" ]; then
    train
fi
if [ "$1" == "test" ]; then
    test
fi
if [ "$1" == "pre" ]; then
    preprocess
fi
if [ "$1" == "eval" ]; then
    eval
fi
