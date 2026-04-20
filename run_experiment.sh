
# copy from lambda machines 
experiment=$1 # experiment numer
root_folder=$2 # root folder from remote server
ip=$3

# load weights and config from remote server + save to correct directory
mkdir test_weights/$experiment
mkdir test_weights/$experiment/staging
scp -i ~/.ssh/tallambda.pem ubuntu@$ip:$root_folder/hparams.json test_weights/$experiment
scp -i ~/.ssh/tallambda.pem ubuntu@$ip:$root_folder/saves/rlbench_gddlp_*.pth test_weights/$experiment/staging
mv test_weights/$experiment/staging/*_best.pth test_weights/$experiment/best.pth
mv test_weights/$experiment/staging/*.pth test_weights/$experiment/recent.pth
rmdir test_weights/$experiment/staging



