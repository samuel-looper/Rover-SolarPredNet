from models.SolarPredHybrid import SunPose_CNN, IR_CNN
from models.SolarPredConv import SolarPredConvNet
from data.data_loader import SolarPredDataset
import torch
from matplotlib import pyplot as plt


# train.py: Trains neural network components on solar energy generation dataset

def net_train(net, dataset, lr, wd, epochs, bs):
    # Parse what type of network is being trained
    if type(net) is IR_CNN:
        net_type = "ir"
    if type(net) is SunPose_CNN:
        net_type = "sun"
    if type(net) is SolarPredConvNet:
        net_type = "e2e"
    else:
        net_type = "other"

    # Split between testing and training set, discard half of the total dataset to improve runtime
    discard_size = int(len(dataset) * 0.5)
    tv_size = int((len(dataset) - discard_size) * 0.85)
    test_size = len(dataset) - tv_size - discard_size
    tv_set, test_set, discard = torch.utils.data.random_split(dataset, [tv_size, test_size, discard_size],
                                                              generator=torch.Generator().manual_seed(81))

    loss = torch.nn.MSELoss()  # Define Mean Square Error Loss
    optimizer = torch.optim.Adam(list(net.parameters()), lr=lr, weight_decay=wd)  # Define Adam optimization algorithm

    # Randomly split between training and validation set
    train_len = int(len(tv_set) * 0.8)
    val_len = len(tv_set) - train_len
    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len], torch.Generator())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    print("Dataset Loaded")

    train_loss = []
    val_loss = []
    best_loss = 0
    best_epoch = 0

    print("Training Length: {}".format(int(train_len / bs)))
    for epoch in range(1, epochs + 1):
        print("Training")
        net.train(True)
        epoch_train_loss = 0
        i = 0

        for data in train_loader:
            image = data["image"].type(torch.FloatTensor)  # Load Input Image

            # Load labels
            label = 0
            if net_type == "e2e":
                label = data["power"].type(torch.FloatTensor)
                label = (label - dataset.power_mean) / dataset.power_std
            if net_type == "sun":
                label = data["sun pose"].type(torch.FloatTensor)
                label = torch.div(label - dataset.sun_means, dataset.sun_stds)
            if net_type == "ir":
                label = data["ir"].type(torch.FloatTensor)
                label = (label - dataset.ir_mean) / dataset.ir_std

            optimizer.zero_grad()  # Reset gradients
            pred = net(image)  # Forward Pass
            minibatch_loss = loss(pred, label)  # Compute loss
            minibatch_loss.backward()  # Backpropagation
            optimizer.step()  # Optimization

            epoch_train_loss += minibatch_loss.item() / train_len

            i += 1
            if i % 60 == 0:
                print(i)

        train_loss.append(epoch_train_loss)
        print("Training Error for this Epoch: {}".format(epoch_train_loss))

        print("Validation")
        net.train(False)
        net.eval()
        epoch_val_loss = 0
        i = 0
        with torch.no_grad():
            for data in val_loader:
                image = data["image"].type(torch.FloatTensor)  # Load Input Images

                # Load labels
                label = 0
                if net_type == "e2e":
                    label = data["power"].type(torch.FloatTensor)
                    label = (label - dataset.power_mean) / dataset.power_std
                if net_type == "sun":
                    label = data["sun pose"].type(torch.FloatTensor)
                    label = torch.div(label - dataset.sun_means, dataset.sun_stds)
                if net_type == "ir":
                    label = data["ir"].type(torch.FloatTensor)
                    label = (label - dataset.ir_mean) / dataset.ir_std

                pred = net(image)  # Forward Pass
                minibatch_loss = loss(pred, label)  # Compute loss

                epoch_val_loss += minibatch_loss.item() / val_len

            val_loss.append(epoch_val_loss)
            print(epoch_val_loss)
            if best_epoch == 0 or epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_epoch = epoch
                torch.save(net.state_dict(), '{}_best.pth'.format(net_type))

            # Plotting
            plt.plot(train_loss, linewidth=2)
            plt.plot(val_loss, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend(["Training Loss", "Validation Loss"])
            plt.savefig("{}_losses_intermediate.png".format(net_type))
            plt.show()

    print("Training Complete")
    print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

    # Plot Final Training Errors
    plt.plot(train_loss, linewidth=2)
    plt.plot(val_loss, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.savefig("{}_losses.png".format(net_type))
    plt.show()


if __name__ == "__main__":
    input_dir = "data"
    dataset = SolarPredDataset(input_dir)

    lr = 0.001
    wd = 0.0005
    epochs = 20
    bs = 16

    solarpred_cnn = SolarPredConvNet()
    ir_cnn = IR_CNN()
    sun_cnn = SunPose_CNN()

    for net in [solarpred_cnn, ir_cnn, sun_cnn]:
        net_train(net, dataset, lr, wd, epochs, bs)
