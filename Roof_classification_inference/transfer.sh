CKPT_PATH="/fast/zcb/code/cbzhao/bridge2025/Roof_classification_inference/bg_extract/ckpt/mm_road/road.pth"

python bg_extract/deploy.py \
    "/fast/zcb/code/cbzhao/bridge2025/Roof_classification_inference/bg_extract/deploy/configs/mmseg/segmentation_tensorrt_static-512x512.py" \
    "/fast/zcb/code/cbzhao/bridge2025/Roof_classification_inference/bg_extract/ckpt/mm_road/road_config.py" \
    "${CKPT_PATH}" \
    "/fast/zcb/code/cbzhao/bridge2025/Roof_classification_inference/bg_extract/deploy/demo1.png" \
    --work-dir "/fast/zcb/code/cbzhao/bridge2025/Roof_classification_inference/bg_extract/tensorrt_road" \
    --device cuda \
    --dump-info