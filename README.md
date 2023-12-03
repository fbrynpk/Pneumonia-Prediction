# Covid19-Prediction

Trying to classify from X-Ray images whether it's classified as COVID-19 or Normal X-Ray Images by replicating the Tiny VGG Architecture

Architecture can be found from: "https://poloclub.github.io/cnn-explainer/"

# [Try the model here!](https://huggingface.co/spaces/fbrynpk/Covid-19-Detection)

# Training Results
### 5 Epochs 

TinyVGG: Train Loss = 0.6877 | Train Acc = 0.60% | Test Loss = 0.6805 | Test Acc = 0.97% <br/>
EffNetB0: Train Loss = 0.2432 | Train Acc = 0.97% | Test Loss = 0.3150 | Test Acc = 0.92%
________________
### 10 Epochs


TInyVGG: Train Loss = 0.2147 | Train Acc = 0.93% | Test Loss = 0.1650 | Test Acc = 0.89% <br/>
EffNetB0: Train Loss = 0.1570 | Train Acc = 0.97% | Test Loss = 0.1964 | Test Acc = 0.94%

Best performing model: **EffNetB0 with 10 Epochs**

# After Data Augmentation and Label Smoothing (Avoid Overfitting Problems)

### 5 Epochs

TinyVGG: Train Loss = 0.6877 | Train Acc = 0.61% | Test Loss = 0.68256 | Test Acc = 0.88%<br/>
EffNetB0: Train Loss = 0.3893 | Train Acc = 0.91% | Test Loss = 0.35206 | Test Acc = 0.95%

________________
### 10 Epochs

TInyVGG: Train Loss = 0.6708 | Train Acc = 0.65% | Test Loss = 0.6423 | Test Acc = 0.83%<br/>
EffNetB0: Train Loss = 0.3242 | Train Acc = 0.93% | Test Loss = 0.2793 | Test Acc = 0.97%

Best performing model: **EffNetB0 with 10 Epochs**
> The trend shows that if the model train for longer then maybe it could still reduce the loss and increase the accuracy without overfitting
