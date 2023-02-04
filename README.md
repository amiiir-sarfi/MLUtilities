This is a utilities library for Linear-Probe Transfer Learning, KNN probe on layer activations.

1. For Transfer learning, run the following (and replace the <ins>underlined</ins> text):

    <pre><code>
    python main.py --task=transfer_linear --data_dir=<u>DataRoot</u> --set=<u>DSName</u> --arch=ResNet<u>xx</u> \
                   --ckpt_root <u>RootCKPT</u> --ckpt_paths <u>"X/Y/Z.pt"</u> --ckpt_info <u>Name4ThisCkpt</u> \
                   --lr 0.01 --batch_size 32
    </code></pre>

    Additionally you can indicate train/test batch sizes, whether to linear probe the validation set or not and more by tweaking the parameters in [config.py](./config.py). 
    
    
    
2.  For the KNN probe (replace the <ins>underlined</ins> text):
    <pre><code>
    python main.py --task=knn_probe --data_dir=<u>DataRoot</u> --set=<u>DSName</u> --arch=ResNet<u>xx</u> \
                   --ckpt_root <u>RootCKPT</u> --ckpt_paths <u>"X/Y/Z.pt"</u> --ckpt_info <u>Name4ThisCkpt</u> \
                   --extract_features 1 --features_per_file 1000
    </code></pre>

    Refer to [config.py](./config.py) to change the hook layers if you wish to check out other layers (in Config.parse line 182). Also, you can modify other parameters such as train/test batch sizes for knn, knn_K through modifying the [config.py](./config.py) file. 

    Please note that for KNN probe the checkpoint paths + checkpoint root must be a hierarchy of 4 directories to avoid errors.


