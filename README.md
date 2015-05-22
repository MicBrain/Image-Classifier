# Image-Classifier

### What is this project about 

We have a large number of 32x32 images and this project has to find out whether or not each of them contains a cat. In this project, I applied some of the performance optimization techniques to the real-world problem of classifying images using a Convolutional Neural Network (CNN).

### How does a Computer tell a Cat from a Chair?

   Image classification describes a problem where a computer is given an image and has to tell what it represents (usually from a set of possible categories). Image classification has long been considered a difficult challenge, and a lot of work has been done over the years to improve its accuracy (i.e., what fraction of pictures gets classified correctly).

   Lately, Convolutional Neural Networks (CNNs) have emerged as a promising solution to this problem. In contrast to classical algorithms, which explicitly encode some insight about the real world into a function, CNNs instead learn a function by being presented with a large number of examples and adjusting themselves based on these examples -- this is called training. Once the CNN has learned the function (or to be more precise, an approximation of it), it can repeatedly apply it to inputs it has not seen before.
   
   In this project, I already have a pre-trained CNN that can classify 32x32 images into 10 categories (such as dog, cat, car or plane). Using this CNN, I developed an application that takes a large set (i.e., 1000's) of images as input and finds those that contain cats.

### How do Convolutional Neural Networks Work?

