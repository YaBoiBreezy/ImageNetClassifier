# ImageNetClassifier
My ImageNet100 classifier, then transfer learning to use both that model and MobileNetV2 to learn a 5-class 2200-image flower dataset

main.py is my classifier for imagenet100, from here: https://www.kaggle.com/datasets/ambityga/imagenet100
it is based on AlexNet, used keras tuner to tune the hyperparameters. Itloads data one batch at a time.
The top5 file was to get my top-5 error rate, which was 25%

I then did transferLearning.py, which used my trained weights on a 5-class 2216-image flower database. That was very difficult, since the flowers are very similar (roses vs red tulips, yellow dandelions vs sunflowers, etc.)
I managed to get 86% accuracy, which is pretty good given the pretrained model type.
Top-5 accuracy was 100% (that's a joke, since the correct answer of a 5-class dataset is always in the top 5 guesses lol)

Lastly I did transferLearning2.py, which does transfer learning with MobileNetV2, a pretrained database for image classification, both with and without ensemble learning. This resulted in a whopping 95% accuracy!
I believe accuracy could be further increased with data augmentation, and a higher-resolution image as input to the network (larger than 224x224x3), as some images are of entire fields of flowers.
