import pickle

with open('cloth/polyfold-square-tasks/train/gt_keypoints/1/1.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)