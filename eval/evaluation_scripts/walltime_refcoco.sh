# Download VLM-R1 rec_jsons_processed, for the final results using VLM-R1 validation set.
cd ../../dataset/RefCOCO
if [ ! -d rec_jsons_processed ]; then
    wget https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip
    unzip rec_jsons_processed.zip
    rm rec_jsons_processed.zip
fi;
cd ../../eval/evaluation_scripts

# Inference Start
export CHECKPOINT='PaDT-MLLM/PaDT_Pro_3B'
export LOG_SUFFIX='padt_pro_3b_walltime'

SPLIT=refcoco_val
# Multi-GPU run inference and save to log files: (pred_comp file records response sentences, pred_results file records structured outputs, i.g. bbox, mask, ...)
CUDA_VISIBLE_DEVICES=0 torchrun --master_port="12369" --nproc_per_node=1 inference_refcoco.py $CHECKPOINT $SPLIT $LOG_SUFFIX



