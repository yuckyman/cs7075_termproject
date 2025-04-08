**# PROJECT NOTES**

## TIMELINE

I. [[Gesture Recognition for Human-Robot Collaboration A Review.pdf|Gesture Recognition]] & embedding mapping  
II. 1. Set up gesture recognition  
      [x] find psychology-based library of gestures  
      	- ekman & friesen’s five types of gestures, [[kinesics.html|kinesics]]:  
      		- emblems  
      		- illustrators  
      		- affect displays  
      		- regulators  
      		- adaptors  
      	- kendon’s gesture studies  
      		- phases: preparation, stroke (main motion), retraction.  
      [ ] define core set of gestures  
      	- common across cultures  
      	- relevant to conversational AI  
      	- detectable via existing pose models  
      [ ] collect example data & create labels  
      	- CMU panoptic or MPII human pose for general body language  
      	- [mediapipe] [2] or [openpose] [1,2]   
      	- extract features like joint angles  
III. 2. Convert gestures to embeddings  
      [ ] map keypoints into vectors (PCS or t-SNE)  
      [ ] normalize vectors (L2 normalization)  
IV. 3. Define gesture categories & tokens  
      [ ] manually classify key gestures (e.g., “engaged”, “defensive”, “confused”, etc.)  
      [ ] assign a discrete token for each category  
V. LLM integration & testing  
VI. 1. Integrate with LLM (ollama)  
      [ ] concatenate gesture tokens with text input  
      [ ] adjust prompts to include nonverbal context  
VII. 2. Fine-tune responses & test interaction  
      [ ] conduct real-time testing with video input  
      [ ] adjust mapping if gestures aren’t recognized well  
      [ ] build simple UI for demonstration (optional)

## for next week  
[ ] get enough reference literature (and make sure we are doing something more unique) and start implementing mediapipe with python code -> get livestream demo where it detects a gesture from camera  
[ ] embed body language into conversation to add context for LLM conversation, 

## REFERENCES  

[1] [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
[2] [https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset)
[3] [https://github.com/google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe)

[4] [https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/](https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/)
