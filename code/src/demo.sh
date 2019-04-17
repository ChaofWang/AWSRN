

#===================================================================== AWSRN =======================================================================
#-------------AWSRNx2 train test
python main.py --model AWSRN --n_resblocks 4 --scale 2  --save AWSRNx2  --epochs 1000 --reset --patch_size 96

python main.py --model AWSRN  --n_resblocks 4 --scale 2  --pre_train ../experiment/AWSRNx2/model/model_latest.pt --save AWSRNx2 --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRNx3 train test
python main.py --model AWSRN --n_resblocks 4 --scale 3 --save AWSRNx3  --epochs 1000 --reset --patch_size 144

python main.py --model AWSRN  --n_resblocks 4 --scale 3  --pre_train ../experiment/AWSRNx3/model/model_latest.pt --save AWSRNx3 --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRNx4 train test
python main.py --model AWSRN --n_resblocks 4 --scale 4  --save AWSRNx4  --epochs 1000 --reset --patch_size 192

python main.py --model AWSRN  --n_resblocks 4 --scale 4  --pre_train ../experiment/AWSRNx4/model/model_latest.pt --save AWSRNx4 --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRNx8 train test
python main.py --model AWSRN --n_resblocks 4 --scale 8  --save AWSRNx8  --epochs 1000 --reset --patch_size 384

python main.py --model AWSRN  --n_resblocks 4 --scale 8  --pre_train ../experiment/AWSRNx8/model/model_latest.pt --save AWSRNx8 --test_only  --dir_data ../../DATA/ --data_test Set5


#===================================================================== AWSRN-M =======================================================================
#-------------AWSRN_Mx2 train test
python main.py --model AWSRN --n_resblocks 3 --scale 2  --save AWSRN_Mx2  --epochs 1000 --reset --patch_size 96

python main.py --model AWSRN  --n_resblocks 3 --scale 2  --pre_train ../experiment/AWSRN_Mx2/model/model_latest.pt --save AWSRN_Mx2 --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRN_Mx3 train test
python main.py --model AWSRN --n_resblocks 3 --scale 3  --save AWSRN_Mx3  --epochs 1000 --reset --patch_size 144

python main.py --model AWSRN  --n_resblocks 3 --scale 3  --pre_train ../experiment/AWSRN_Mx3/model/model_latest.pt --save AWSRN_Mx3 --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRN_Mx4 train test
python main.py --model AWSRN --n_resblocks 3 --scale 4  --save AWSRN_Mx4  --epochs 1000 --reset --patch_size 192

python main.py --model AWSRN  --n_resblocks 3 --scale 4  --pre_train ../experiment/AWSRN_Mx4/model/model_latest.pt --save AWSRN_Mx4 --test_only  --dir_data ../../DATA/ --data_test Set5


#===================================================================== AWSRN-S =======================================================================
#-------------AWSRN_Sx2 train test
python main.py --model AWSRN --n_resblocks 1 --scale 2  --save AWSRN_Sx2  --epochs 1000 --reset --patch_size 96

python main.py --model AWSRN  --n_resblocks 1 --scale 2  --pre_train ../experiment/AWSRN_Sx2/model/model_latest.pt --save AWSRN_Sx2 --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRN_Sx3 train test
python main.py --model AWSRN --n_resblocks 1 --scale 3 --save AWSRN_Sx3_retrain  --epochs 1000  --reset --patch_size 144

python main.py --model AWSRN --n_resblocks 1 --scale 3  --pre_train ../experiment/AWSRN_Sx3/model/model_latest.pt --save AWSRN_Sx3_retrain --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRN_Sx4 train test
python main.py --model AWSRN --n_resblocks 1 --scale 4 --save AWSRN_Sx4  --epochs 1000  --reset --patch_size 192

python main.py --model AWSRN --n_resblocks 1 --scale 4  --pre_train ../experiment/AWSRN_Sx4/model/model_latest.pt --save AWSRN_Sx4 --test_only  --dir_data ../../DATA/ --data_test Set5


#===================================================================== AWSRN-SD =======================================================================
#-------------AWSRN_SDx2 train test
python main.py --model AWSRND --n_resblocks 1 --n_feats 16 --block_feats 128 --scale 2  --save AWSRN_SDx2  --epochs 1000 --reset --patch_size 96 

python main.py --model AWSRND  --n_resblocks 1 --n_feats 16 --block_feats 128 --scale 2  --pre_train ../experiment/AWSRN_SDx2/model/model_latest.pt --save AWSRN_SDx2 --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRN_SDx3 train test
python main.py --model AWSRND --n_resblocks 1 --n_feats 16 --block_feats 128 --scale 3  --save AWSRN_SDx3  --epochs 1000 --reset --patch_size 144 

python main.py --model AWSRND  --n_resblocks 1 --n_feats 16 --block_feats 128 --scale 3  --pre_train ../experiment/AWSRN_SDx3/model/model_latest.pt --save AWSRN_SDx3 --test_only  --dir_data ../../DATA/ --data_test Set5


#-------------AWSRN_SDx4 train test
python main.py --model AWSRND --n_resblocks 1 --n_feats 16 --block_feats 128 --scale 4  --save AWSRN_SDx4  --epochs 1000 --reset --patch_size 192 

python main.py --model AWSRND  --n_resblocks 1 --n_feats 16 --block_feats 128 --scale 4  --pre_train ../experiment/AWSRN_SDx4/model/model_latest.pt --save AWSRN_SDx4 --test_only  --dir_data ../../DATA/ --data_test Set5
