SET IMAGE_NAME="harmonizer-app-deployment"
SET GCP_PROJECT="ai5-c3-group7"
SET GCP_ZONE="us-west1-b"
SET GOOGLE_APPLICATION_CREDENTIALS=/secrets/deployment.json
SET USE_GKE_GCLOUD_AUTH_PLUGIN=True
docker build -t %IMAGE_NAME% -f Dockerfile .
cd ..
docker run --rm --name %IMAGE_NAME% -ti ^
           -v /var/run/docker.sock:/var/run/docker.sock ^
           --mount type=bind,source="%cd%\deployment",target=/app ^
           --mount type=bind,source="%cd%\secrets",target=/secrets ^
           --mount type=bind,source="C:\Users\<UserName>\.ssh",target=/home/app/.ssh ^
           --mount type=bind,source="%cd%\api-service",target=/api-service ^
           --mount type=bind,source="%cd%\frontend-react",target=/frontend-react ^
           -e GOOGLE_APPLICATION_CREDENTIALS="%GOOGLE_APPLICATION_CREDENTIALS%" ^
           -e GCP_PROJECT="%GCP_PROJECT%" ^
           -e USE_GKE_GCLOUD_AUTH_PLUGIN="%USE_GKE_GCLOUD_AUTH_PLUGIN%" ^
           -e GCP_ZONE="%GCP_ZONE%" %IMAGE_NAME%
