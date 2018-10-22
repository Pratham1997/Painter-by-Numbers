# Painter-by-Numbers
The aim of this project is to develop a model which can identify the authenticity of a painting. We predict the painter of a given painting from our database of famous painters. This can help in the identification of original and forged paintings. The results are obtained for both the standard CNN model, and a novel approach of Hierarchical CNN.

<pre>
1) For training (pre-trained weights have also been included) -
        1. Add training data to the corresponding directory(examples provided) and move the Dataset directory to the directory NormalCNN or Hierarchical CNN.
        ii)  For Normal CNN -
                Navigate to the directory NormalCNN and run -
                        python NormalCNNtrain.py
        iii) For Hierarchical CNN-
                Navigate to the directory HierarchicalCNN and run -
                        python Hierarchical_Train.py

2) For testing -
        i) Add testing data to the corresponding directory(examples provided) and move the Testing Data directory to the directory NormalCNN or Hierarchical CNN.
        ii)  For Normal CNN -
                Navigate to the directory NormalCNN and run -
                        python NormalCNNtest.py
        iii) For Hierarchical CNN-
                Navigate to the directory HierarchicalCNN and run -
                        python Hierarchical_Test.py                   
 </pre>          

