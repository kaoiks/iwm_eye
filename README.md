# Detection of retinal dna vessels


The purpose of this application is to automatically detect blood vessels in the bottom of the retina using a given input image. The algorithm employed in this process determines, on a pixel-by-pixel basis, whether a particular pixel represents a blood vessel or not.

To achieve this goal, we have implemented a simplified approach that utilizes the conventional Gaussian image blurring technique with carefully selected parameters. This blurring technique helps in enhancing the image quality for subsequent processing steps.

Building upon the image blurring step, the next stage involves the utilization of the Canny edge detection method. This technique assists in identifying the edges and boundaries of the blood vessels within the retinal image, further contributing to accurate vessel detection.

Through the combination of these techniques, our application provides an automated solution for the detection of retinal DNA vessels, facilitating a comprehensive analysis of the retinal health and aiding in the diagnosis of various ocular conditions.
