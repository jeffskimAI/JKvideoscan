#!/usr/bin/env bash
# Setup Eventarc trigger to route GCS upload events to Cloud Run
set -euo pipefail

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project)}"
REGION="${GCP_REGION:-us-central1}"
BUCKET_NAME="${GCS_BUCKET:-jeffsvideoscan-ingest}"
SERVICE_NAME="jeffsvideoscan"
TRIGGER_NAME="gcs-video-upload-trigger"

echo "=== Configuring Eventarc Trigger ==="
echo "Bucket:  gs://${BUCKET_NAME}"
echo "Service: ${SERVICE_NAME}"
echo "Trigger: ${TRIGGER_NAME}"

# Ensure bucket exists
if ! gsutil ls -b "gs://${BUCKET_NAME}" >/dev/null 2>&1; then
  echo "Creating bucket gs://${BUCKET_NAME} in ${REGION}..."
  gcloud storage buckets create "gs://${BUCKET_NAME}" --location="${REGION}"
fi

# Enable required APIs
gcloud services enable \
  eventarc.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  aiplatform.googleapis.com \
  firestore.googleapis.com

# Create Eventarc trigger for GCS object finalization
gcloud eventarc triggers create "${TRIGGER_NAME}" \
  --location="${REGION}" \
  --destination-run-service="${SERVICE_NAME}" \
  --destination-run-region="${REGION}" \
  --event-filters="type=google.cloud.storage.object.v1.finalized" \
  --event-filters="bucket=${BUCKET_NAME}" \
  --service-account="jeffsvideoscan-sa@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== Eventarc trigger ${TRIGGER_NAME} successfully created! ==="
