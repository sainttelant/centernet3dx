about convert to 3d onnx model

1\ conda activate CenterTrack
2\ cd src
3\ python export_onnx_ddd.py


python main.py --task ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 16 --master_batch 7 --num_epochs 70 --lr_step 45,60 --gpus 0

python demo.py --task ddd --demo ../../../CenterTrack/videos/ceshi.mp4 --load_model ../models/ddd_3dop.pth