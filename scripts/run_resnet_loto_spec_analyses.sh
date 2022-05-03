cd /home/paysan_d/PycharmProjects/image2reg/
for file in config/image_embedding/specific_targets/loto/analyze_loto_exp/resnet_analyses/*; do python run.py --config $file; done