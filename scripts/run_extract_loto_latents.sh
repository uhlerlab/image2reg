cd /home/paysan_d/PycharmProjects/image2reg/
for file in config/image_embedding/specific_targets/loto/analyze_loto_exp/extract_loto_latents/complete/*; do python run.py --config $file; done