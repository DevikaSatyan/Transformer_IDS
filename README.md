# Transformer-based Intrusion Detection System for CAN Network
This is the implementation of an encoder-only Transformer used for intrusion detection in CAN Bus network.
## The datasets used for this implementation are :
1. [Car Hacking](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
2. [Survival Analysis](https://ocslab.hksecurity.net/Datasets/survival-ids)

## Code
1. car_hacking.py : utilizes the transformer model to perform intrusion detection on the car hacking dataset
2. survival_analysis.py : utilizes the transformer model to perform intrusion detection on the survival_analysis dataset
3. unseen_attack.py : performs cross dataset evaluation by training on the survival_analysis dataset and testing on car hacking dataset
