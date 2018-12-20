PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-mlengine
echo $BUCKET_NAME
REGION=us-east1
gsutil mb -l $REGION gs://$BUCKET_NAME
gsutil ls
