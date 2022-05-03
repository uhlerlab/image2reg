cd /home/paysan_d/PycharmProjects/image2reg/
for file in config/image_embedding/specific_targets/loto/train_resnet/stratified/*; do python run.py --config $file; done