cd /home/paysan_d/PycharmProjects/image2reg/
for file in config/image_embedding/specific_targets/cv/*; do python run.py --config $file; done