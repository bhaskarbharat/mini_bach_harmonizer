In terminal type:
cd /harmonizer-app/api-service

Then:
sh docker-shell.sh
(It will run the Dockerfile. Build the image and run the container)

Once the container starts running, ssh into the container. In the terminal you should see a prompt ending with 'app'. Type in the prompt:
uvicorn_server
(It will run the fastapi service)

Once the API service starts running, go to browser and type:
http://localhost:9000/docs

There you can see a tile '/predict'. Click on 'Try it out' to check if the API is working correctly.