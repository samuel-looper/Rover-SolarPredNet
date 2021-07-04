from models.SolarPredHybrid import SunPose_CNN, IR_CNN
from models.SolarPredConv import SolarPredConvNet
from data.data_loader import SolarPredDataset
from scipy.spatial.transform import Rotation as R
import torch
from matplotlib import pyplot as plt
import os
import numpy as np


# evaluate.py: Evaluates solar energy generation prediction models on representative test

def evaluate(dataset, e2e_net, sun_pose_net, ir_net):
    # Neural Network Evaluation

    # Split between testing and training set
    discard_size = int(len(dataset) * 0.5)
    tv_size = int((len(dataset) - discard_size) * 0.85)
    test_size = len(dataset) - tv_size - discard_size
    tv_set, test_set, discard = torch.utils.data.random_split(dataset, [tv_size, test_size, discard_size],
                                                              generator=torch.Generator().manual_seed(81))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
    loss = torch.nn.MSELoss()  # Define Mean Square Error Loss
    e2e_test_loss = []
    hybrid_test_loss = []
    i = 0

    with torch.no_grad():
        for data in test_loader:
            image = data["image"].type(torch.FloatTensor)  # Load Input Images
            # Load labels
            label = data["power"].type(torch.FloatTensor)[0]

            # End To End Testing
            pred_z = e2e_net(image)[0]
            pred = pred_z * dataset.power_std + dataset.power_mean  # Forward Pass
            minibatch_loss = torch.sqrt(loss(pred, label))  # Compute loss
            e2e_test_loss.append(minibatch_loss.item())

            # Hybrid Testing
            sun_pose = torch.mul(sun_pose_net(image), dataset.sun_stds) + dataset.sun_means  # Forward Passes
            ir = ir_net(image) * dataset.ir_std + dataset.ir_mean

            r = R.from_quat(sun_pose[0])  # Power Model
            sx = r.as_matrix()[0, 2]
            sy = r.as_matrix()[1, 2]
            area = 9 * max(sx, -sx) + 3 * max(0, -sy)
            pred = torch.FloatTensor([area * ir])
            minibatch_loss = torch.sqrt(loss(pred, label))  # Compute loss
            hybrid_test_loss.append(minibatch_loss.item() / len(test_set))

            i += 1
            if i % 100 == 0:
                print(i)
                print(np.asarray(e2e_test_loss).mean())
                print(np.asarray(hybrid_test_loss).mean())

    return e2e_test_loss, hybrid_test_loss


if __name__ == "__main__":
    input_dir = "data"
    output_dir = "outputs"
    dataset = SolarPredDataset(input_dir)

    # Load models
    solar_pred_net = SolarPredConvNet()
    solar_pred_net.load_state_dict(torch.load(os.path.join(input_dir, "solar_pred_net_best.pth")))
    solar_pred_net.train(False)
    solar_pred_net.eval()

    ir_net = IR_CNN()
    ir_net.load_state_dict(torch.load(os.path.join(input_dir, "ir_best.pth")))
    ir_net.train(False)
    ir_net.eval()

    sun_pose_net = SunPose_CNN()
    sun_pose_net.load_state_dict(torch.load(os.path.join(input_dir, "sun_pose_best.pth")))
    sun_pose_net.train(False)
    sun_pose_net.eval()

    e2e_test_loss, hybrid_test_loss = evaluate(dataset, solar_pred_net, sun_pose_net, ir_net)

    e2e_MSE = np.asarray(e2e_test_loss).mean()
    hybrid_MSE = np.asarray(hybrid_test_loss).mean()
    np.savetxt(os.path.join(output_dir, "e2e_errors.csv"), np.asarray(e2e_test_loss))
    np.savetxt(os.path.join(output_dir, "hybrid_errors.csv"), np.asarray(hybrid_test_loss))

    # Plot Final Training Errors
    plt.barh(["End To End Model", "Hybrid Model"], [e2e_MSE, hybrid_MSE], height=0.4)
    plt.title("Mean Square Test Error Comparison")
    plt.savefig(os.path.join(output_dir, "test_error.png"))
    # plt.show()

    # Plot Training Error Distributions
    plt.hist(e2e_test_loss, bins=25)
    plt.title("End To End Model Error Distribution")
    plt.xlabel("Test Error")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "e2e_test_error.png"))
    # plt.show()

    plt.hist(hybrid_test_loss, bins=25, )
    plt.title("Hybrid Model Error Distribution")
    plt.xlabel("Test Error")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "hybrid_test_error.png"))
    # plt.show()
