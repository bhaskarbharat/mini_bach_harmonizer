import os
import asyncio
import json
import gdown
import subprocess


transformer_weights_path = "/persistent/transformer_weights"
coconet_weights_path = "/persistent/coconet_weights"
output_json_path = "/persistent/output_json"

# Setup transformer_weights folder
if not os.path.exists(transformer_weights_path):
    os.mkdir(transformer_weights_path)

# Setup coconet_weights folder
if not os.path.exists(coconet_weights_path):
    os.mkdir(coconet_weights_path)

# Setup output_json folder
if not os.path.exists(output_json_path):
    os.mkdir(output_json_path)

def download_model_weights():
    urls = {
    '/persistent/transformer_weights/keras_transformer_model_weights.ckpt.data-00000-of-00001':'https://drive.google.com/uc?id=1Pe7E-7WCCGs9mv7biqxre_lc8WUgMs4T',
    '/persistent/transformer_weights/keras_transformer_model_weights.ckpt.index':'https://drive.google.com/uc?id=1N3BeASouU0Lu68QBmCOnb7n55BPTxYQQ',
    '/persistent/coconet_weights/my_model.data-00000-of-00001':'https://drive.google.com/uc?id=1mIbDPn9PsASBHDwM96JMafcOO4CaVljX',
    '/persistent/coconet_weights/my_model.index':'https://drive.google.com/uc?id=1_DLbqvCj8nsXY65tVavhAx9ZWYbeGeAe',
    '/persistent/output_json/output.json':'https://drive.google.com/uc?id=1fGLDatyEHq12jfdCTHnpWrldgDXaF4J9',
        }

    # for output, url in urls.items():
    #     # gdown.download(url, os.path.abspath(output), quiet=False)
    #     os.system(f"gdown {url} -O {output}")
    subprocess.Popen('gdown https://drive.google.com/uc?id=1Pe7E-7WCCGs9mv7biqxre_lc8WUgMs4T -O /persistent/transformer_weights/keras_transformer_model_weights.ckpt.data-00000-of-00001', shell=True)
    subprocess.Popen('gdown https://drive.google.com/uc?id=1N3BeASouU0Lu68QBmCOnb7n55BPTxYQQ -O /persistent/transformer_weights/keras_transformer_model_weights.ckpt.index', shell=True)
    subprocess.Popen('gdown https://drive.google.com/uc?id=1mIbDPn9PsASBHDwM96JMafcOO4CaVljX -O /persistent/coconet_weights/my_model.data-00000-of-00001', shell=True)
    subprocess.Popen('gdown https://drive.google.com/uc?id=1_DLbqvCj8nsXY65tVavhAx9ZWYbeGeAe -O /persistent/coconet_weights/my_model.index', shell=True)
    subprocess.Popen('gdown https://drive.google.com/uc?id=1fGLDatyEHq12jfdCTHnpWrldgDXaF4J9 -O /persistent/output_json/output.json', shell=True)