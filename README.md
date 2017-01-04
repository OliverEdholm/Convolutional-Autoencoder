# Neural-convolutional-autoencoder
A convolutional autoencoder made in TFLearn.

### Examples
I trained this "architecture" on selfies (256*256 RGB) and terminated the training procedure after one epoch out of sheer excitement.

Here are some results (selfies are taken from google image search https://www.google.com/search?as_st=y&tbm=isch&as_q=selfie&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=itp:face,sur:fmc):

Image 1:

![Img1](https://raw.githubusercontent.com/OliverEdholm/Neural-convolutional-autoencoder/master/inputs/selfie1.jpg)
![Img1O](https://raw.githubusercontent.com/OliverEdholm/Neural-convolutional-autoencoder/master/outputs/selfie1_output.jpg)

Image 2:

![Img2](https://raw.githubusercontent.com/OliverEdholm/Neural-convolutional-autoencoder/master/inputs/selfie2.jpg)
![Img2O](https://raw.githubusercontent.com/OliverEdholm/Neural-convolutional-autoencoder/master/outputs/selfie2_output.jpg)

Image 3:

![Img3](https://raw.githubusercontent.com/OliverEdholm/Neural-convolutional-autoencoder/master/inputs/selfie3.jpg)
![Img3O](https://raw.githubusercontent.com/OliverEdholm/Neural-convolutional-autoencoder/master/outputs/selfie3_output.jpg)

Image 4:

![Img4](https://raw.githubusercontent.com/OliverEdholm/Neural-convolutional-autoencoder/master/inputs/potato.jpg)
![Img4O](https://raw.githubusercontent.com/OliverEdholm/Neural-convolutional-autoencoder/master/outputs/potato_output.jpg)

### Requirements
* Python 3.*
* TFlearn
* Keras, for evaluation script

### Usage
**Training and dataset preparation:**

1. Create a folder with the name "images", without quotation marks.

2. Inside the "images" folder, create a folder called "0".

3. Put all the images you want to train on there.

4. Create a folder called "checkpoints".

5. Done.

**Training:**

Run this command to train the convolutional autoencoder on the images in the images folder.
```
python3 quote_lstm.py
```
All checkpoints will be stored in the checkpoints folder.

**Evaluation**

To evaluate a checkpoint on an image you can run.
```
python3 evaluate_autoencoder.py <checkpoints/checkpointname> <path_to_image>
```
The output will be saved as "output.jpg".

### Other
Made by Oliver Edholm, 14 years old.
