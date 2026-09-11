#!/usr/bin/env bash
# Deploy Jeff's VideoScan service to Google Cloud Run
set -euo pipefail

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="jeffsvideoscan"
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/jeffsvideoscan-repo/${SERVICE_NAME}:latest"

echo "=== Deploying ${SERVICE_NAME} ==="
echo "Project: ${PROJECT_ID}"
echo "Region:  ${REGION}"
echo "Image:   ${IMAGE_TAG}"

# 1. Build and push container using Cloud Build
echo "Building container image with Google Cloud Build..."
gcloud builds submit --tag "${IMAGE_TAG}" .

# 2. Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_TAG}" \
  --platform managed \
  --region "${REGION}" \
  --service-account "jeffsvideoscan-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --memory 4Gi \
  --cpu 2 \
  --timeout 900 \
  --concurrency 4 \
  --set-env-vars "GCP_PROJECT=${PROJECT_ID},GCP_REGION=${REGION},GEMINI_MODEL=gemini-2.5-flash,GCS_BUCKET=jeffsvideoscan-ingest" \
  --allow-unauthenticated

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format 'value(status.url)')
echo "=== Successfully deployed ${SERVICE_NAME} to ${SERVICE_URL} ==="
