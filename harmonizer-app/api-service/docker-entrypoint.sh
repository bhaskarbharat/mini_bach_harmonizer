#!/bin/bash

echo "Container is running!!!"

# this will run the api/service.py file with the instantiated app FastAPI
uvicorn_server() {
    uvicorn api.service:app --host 0.0.0.0 --port 9000 --log-level debug --reload --reload-dir api/ "$@"
}

uvicorn_server_production() {
    pipenv run uvicorn api.service:app --host 0.0.0.0 --port 9000 --lifespan on
}

export -f uvicorn_server
export -f uvicorn_server_production

echo -en "\033[92m
The following commands are available:
    uvicorn_server
        Run the Uvicorn Server
\033[0m
"
# Here I can download model weights
mkdir /persistent/transformer_weights
mkdir /persistent/coconet_weights
mkdir /persistent/output_json
pipenv run gdown https://drive.google.com/uc?id=1Pe7E-7WCCGs9mv7biqxre_lc8WUgMs4T -O /persistent/transformer_weights/keras_transformer_model_weights.ckpt.data-00000-of-00001
pipenv run gdown https://drive.google.com/uc?id=1N3BeASouU0Lu68QBmCOnb7n55BPTxYQQ -O /persistent/transformer_weights/keras_transformer_model_weights.ckpt.index
pipenv run gdown https://drive.google.com/uc?id=1mIbDPn9PsASBHDwM96JMafcOO4CaVljX -O /persistent/coconet_weights/my_model.data-00000-of-00001
pipenv run gdown https://drive.google.com/uc?id=1_DLbqvCj8nsXY65tVavhAx9ZWYbeGeAe -O /persistent/coconet_weights/my_model.index
pipenv run gdown https://drive.google.com/uc?id=1fGLDatyEHq12jfdCTHnpWrldgDXaF4J9 -O /persistent/output_json/output.json

if [ "${DEV}" = 1 ]; then
  pipenv shell
else
  uvicorn_server_production
fi
