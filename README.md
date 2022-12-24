# Action Analysis Research 視訊動作分析探討

This is a research repository aiming to analyze and create various applications using extracted data.

這是一個以使用者的動作分析與實作多項 app 的程式專案。

## Current Goals 目前目標

```txt

Network: DNN
Type: Classification
Dataset: UR_Fall (Subjected to change in the future, please read the following section.)
Framework: Tensorflow
Hardware Acceleration: Apple M1 Pro, 16GB RAM on Metal API
Language: Python
```

- [x] Refactor the codes
  - [x] Easy filesystem control
  - [x] Easy switch video stream (layer of abstraction)
- [ ] Train the network (Labels > 1)
  - [x] UR_Fall
  - [ ] MICC_Fall
  - [ ] Research on other datasets
    - [ ] UCF101
    - [ ] HMDB51
    - [ ] NTU RGB+D
    - [ ] Kinetics
- [ ] Test the network
  - [ ] Mediapipe + Heuristic Methods
  - [ ] Mediapipe + DNN
  - [ ] OpenPose + DNN
- [ ] Result compilation
