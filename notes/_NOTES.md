# PROJECT NOTES
[[CS7075_ProjectProposal_Outline.pdf|PROJECT PROPOSAL]] 

## TIMELINE

1. Gesture recognition & embedding mapping
    

2. Set up [[Gesture Recognition for Human-Robot Collaboration A Review.pdf|Gesture Recognition]]   
    [ ] find psychology-based library of gestures  
    - [[kinesics.html|ekman & friesen’s five types of gestures]]:  
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
    - CMU panoptic or [MPII human pose](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset) for general body language  
    [ ] [mediapipe](https://github.com/google-ai-edge/mediapipe) or openpose  
    [ ] extract features like joint angles
    
3. Convert gestures to embeddings  
    [ ] map keypoints into vectors ([PCS or t-SNE](https://www.geeksforgeeks.org/difference-between-pca-vs-t-sne/))  
    [ ] normalize vectors (L2 normalization)
    
4. Define gesture categories & tokens  
    [ ] manually classify key gestures (e.g., “engaged”, “defensive”, “confused”, etc.)  
    [ ] assign a discrete token for each category
    

5. LLM integration & testing
    

6. Integrate with LLM (ollama)  
    [ ] concatenate gesture tokens with text input  
    [ ] adjust prompts to include nonverbal context
    
7. Fine-tune responses & test interaction  
    [ ] conduct real-time testing with video input  
    [ ] adjust mapping if gestures aren’t recognized well  
    [ ] build simple UI for demonstration (optional)
    

  

## for next week

[ ] get enough reference literature (and make sure we are doing something more unique) and start implementing mediapipe with python code -> get livestream demo where it detects a gesture from camera

[ ] embed body language into conversation to add context for LLM conversation, 

  

## REFERENCES

[https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)