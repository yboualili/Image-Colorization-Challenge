Image Colorization Competition

This project was part of a study competition where the task was to colorize grayscale images. Participants were required to develop a method that would accurately predict and apply color to grayscale images, with the performance measured by the Mean Squared Error (MSE). I won the competition with a MSE of 5.9 on the test set.

For my approach, I used a GAN architecture with a U-Net as the generator. The U-Net was built on a ResNet backbone, which provided strong feature extraction capabilities and improved the accuracy of the colorization. I also experimented with a ConvNeXt network, but the ResNet-based U-Net outperformed it in terms of MSE, delivering superior results.

Master 3. Semester