import matplotlib.pyplot as plt
import torch

test_set = torch.load('data/SHM_test_set.pt')

test_models = []
for i in [10, 20, 30, 40, 50, 100, 150, 200]:
    all_models = torch.load('model/all_learner_{}sample.pt'.format(i))
    for model in all_models:
        model.eval()
    test_models.append(all_models)

t_span = test_set[0, :, -1]
test_traj = test_models[3][5].trajectory(test_set[:, 0, 0:2], t_span).detach().transpose(0,1)