TRAINER_PACKAGE_PATH="/Users/berna/Dropbox/wenance/tf/example1/trainer"
MAIN_TRAINER_MODULE="trainer.task"
PACKAGE_STAGING_PATH="gs://liquid-layout-213214-mlengine/"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="mnist_$now"
JOB_DIR="gs://liquid-layout-213214-mlengine/example1/out"
REGION="us-east1"


gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --job-dir $JOB_DIR  \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    -- \
    --bucket /tmp
