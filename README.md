# MixNet

This is a PyTorch implementation of MixNet

# Train on your data
In order to train the model on your data, follow the steps below 
# 1. data preprocessing 
* prepare your data and make sure the data is formatted in an PSV format as below without the header
```
speaker reference audio path,audio path,text,duration
path/speaker_reference.wav|path/clean_audio_sample.wav|the text in that file|3.2 

# 2. Setup development environment
* create enviroment 
```bash
python -m venv env
```
* activate the enviroment
```bash
source env/bin/activate
```
* install the required dependencies
```bash
pip install -r requirements.txt
```
# 3. Training 
* update the args (for the dataloader) and hyperparams (for the model) file if needed
* train the model 

# 4. Evaluating
* update the args (for the dataloader): set batch size to 1
* run eval<model_name>.py
