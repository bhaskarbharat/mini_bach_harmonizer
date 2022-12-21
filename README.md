# Mini Bach Harmonizer

## Problem Statement
The Mini Bach Harmonizer is a tool that takes 8 note melody as input and generates 3 accompanying harmonies infused with the creative genius of the famous 19th century Baroque composer Johann Sebastian Bach.

## Project Presentation

### Presentation
<iframe src="https://prezi.com/p/embed/XfH2B5x19bg28RRCiHuH/" id="iframe_container" frameborder="0" webkitallowfullscreen="" mozallowfullscreen="" allowfullscreen="" allow="autoplay; fullscreen" height="315" width="560"></iframe>

### Video Presentation
[![video presentation](https://github.com/bhaskarbharat/mini_bach_harmonizer/raw/main/images/Presentation_Wallpaper.png)](https://www.youtube.com/watch?v=G5-u3YE2TSQ&t=229s)


### Here are the steps to deploy the __harmonizer-app__:
 - Download [harmonizer-app](https://github.com/bhaskarbharat/mini_bach_harmonizer/tree/main/harmonizer-app) folder.
 - `mkdir persistent-folder` inside the harmonizer-app folder.
 - `mkdir secrets` inside the harmonizer-app folder.
### API's to enable in GCP before you begin
Search for each of these in the GCP search bar and click enable to enable these API's on the Project ID on which you want to deploy the app.
- Compute Engine API
- Service Usage API
- Cloud Resource Manager API
- Google Container Registry API
- Kubernetes Engine API
### Create a service account for deployment
Do this on the Project ID on which you want to deploy the app.
- Go to GCP Console, search for "Service accounts" from the top search box. or go to: "IAM & Admins" > "Service accounts" from the top-left menu and create a new service account called "deployment"
- Give the following roles:
- For `deployment`:
    - Compute Admin
    - Compute OS Login
    - Container Registry Service Agent
    - Kubernetes Engine Admin
    - Service Account User
    - Storage Admin
- Then click done.
- This will create a service account
- On the right "Actions" column click the vertical ... and select "Create key". A prompt for Create private key for "deployment" will appear select "JSON" and click create. This will download a Private key json file to your computer. Copy this json file into the secrets folder.
- Rename the json key file to `deployment.json`
- Follow the same process Create another service account called gcp-service (to access and manage Google Container Registry)
- For `gcp-service` give the following roles:
    - Storage Object Viewer
- Then click done.
- This will create a service account
- On the right "Actions" column click the vertical ... and select "Create key". A prompt for Create private key for "gcp-service" will appear select "JSON" and click create. This will download a Private key json file to your computer. Copy this json file into the secrets folder.
- Rename the json key file to `gcp-service.json`
### Setup Docker Container (Ansible, Docker, Kubernetes)
Rather than each of installing different tools for deployment we will use Docker to build and run a standard container will all required software.
- cd into deployment
- Go into docker-shell.sh or docker-shell.bat and change GCP_PROJECT to your project id
- Run sh docker-shell.sh or docker-shell.bat for windows
- Check versions of tools:
 - `gcloud --version`
 - `ansible --version`
 - `kubectl version --client`
- Check to make sure you are authenticated to GCP
 - Run `gcloud auth list`

Now you have a Docker container that connects to your GCP and call create VMs, deploy containers all from the command line

### SSH Setup
#### Configuring OS Login for service account
`gcloud compute project-info add-metadata --project <YOUR GCP_PROJECT> --metadata enable-oslogin=TRUE`
 
#### Create SSH key for service account
- `cd /secrets`
- `ssh-keygen -f ssh-key-deployment`
- `cd /app`
 
#### Providing public SSH keys to instances
`gcloud compute os-login ssh-keys add --key-file=/secrets/ssh-key-deployment.pub`
 
From the output of the above command keep note of the username. Here is a snippet of the output:
```
- accountId: ai5-project
    gid: '3906553998'
    homeDirectory: /home/sa_100110341521630214262
    name: users/deployment@ai5-project.iam.gserviceaccount.com/projects/ai5-project
    operatingSystemType: LINUX
    primary: true
    uid: '3906553998'
    username: sa_100110341521630214262
```
 
The username is sa_100110341521630214262

### Deployment Setup
- Add ansible user details in `inventory.yml` file
- GCP project details in `inventory.yml` file
- GCP Compute instance details in `inventory.yml` file

### Deployment
- Build and Push Docker Containers to GCR (Google Container Registry)
`ansible-playbook deploy-docker-images.yml -i inventory.yml`
 
### Deploy to Kubernetes Cluster
We will use ansible to create and deploy the mushroom app into a Kubernetes Cluster
#### Create & Deploy Cluster
`ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present`
#### View the App
- Copy the `nginx_ingress_ip` from the terminal from the create cluster command
- Go to `http://<YOUR INGRESS IP>.sslip.io`
#### Delete Cluster
`ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=absent`


 

## Team Members:
1. [Ajinkya Bhanudas Dessai](https://www.linkedin.com/in/ajinkyabhanudas/), Senior Ecosystems Engineer, Analog Devices
2. [Bhaskar Bharat](https://www.linkedin.com/in/bhaskarbharat/), Academic Program Manager, Univ.AI
3. [Shibani Budhraja](https://www.linkedin.com/in/shibanibudhraja/), Workshop Coordinator Univ.AI 
4. [Srish Kulkarni](https://www.linkedin.com/in/srish18/), B.E. CS GNDEC Bidar